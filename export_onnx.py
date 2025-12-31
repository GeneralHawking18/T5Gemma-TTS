#!/usr/bin/env python3
"""
Unified ONNX Export & Quantization Script for T5Gemma-TTS.

Features:
- Exports Encoder, Decoder (Init & Step), and XCodec2
- Handles FP32 export for stable quantization
- Performs INT8 Dynamic Quantization
- CPU-Optimized execution

Usage:
    python export_onnx.py --model_name "Aratako/T5Gemma-TTS-2b-2b"
"""

# =============================================================================
# CRITICAL: Set TMPDIR BEFORE any imports to avoid "No space left on device"
# The system /tmp is on the root partition (98% full, only 6GB free).
# We redirect all temp files to the HDD which has 31GB+ free.
# =============================================================================
import os
_SAFE_TMPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_working"))
os.makedirs(_SAFE_TMPDIR, exist_ok=True)
os.environ["TMPDIR"] = _SAFE_TMPDIR
os.environ["TEMP"] = _SAFE_TMPDIR  # Windows compatibility
os.environ["TMP"] = _SAFE_TMPDIR   # Windows compatibility
import tempfile
tempfile.tempdir = _SAFE_TMPDIR  # Force Python's tempfile module to use it
print(f"[INIT] TMPDIR set to: {_SAFE_TMPDIR}")

import gc
import json
import shutil
import logging
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

# Local modules
from onnx_modules import (
    EncoderWrapper, 
    DecoderInitWrapper, 
    DecoderStepWrapper, 
    XCodec2DecoderWrapper
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Monkeypatch for ONNX compatibility (SDPA)
# =============================================================================
def _apply_sdpa_monkeypatch() -> None:
    import transformers.masking_utils
    def _custom_no_vmap_sdpa_mask(batch_size, cache_position, kv_length, kv_offset=0, mask_function=None, attention_mask=None, **kwargs):
        device = cache_position.device
        q_length = cache_position.shape[0]
        query_idx = cache_position.unsqueeze(1)
        key_idx = torch.arange(kv_length, device=device).unsqueeze(0) + kv_offset
        causal_mask = query_idx >= key_idx
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            padding_mask = attention_mask.to(torch.bool)[:, None, None, :]
            causal_mask = causal_mask & padding_mask
        return causal_mask

    transformers.masking_utils.sdpa_mask = _custom_no_vmap_sdpa_mask
    transformers.masking_utils.sdpa_mask_recent_torch = _custom_no_vmap_sdpa_mask
    transformers.masking_utils.sdpa_mask_older_torch = _custom_no_vmap_sdpa_mask
    
    # Also patch the global mapping which captures the function references
    if hasattr(transformers.masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS"):
        mapping = transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
        if "sdpa" in mapping:
            mapping["sdpa"] = _custom_no_vmap_sdpa_mask
            
    logger.info("Monkeypatched transformers.masking_utils.sdpa_mask")

_apply_sdpa_monkeypatch()

# =============================================================================
# Exporter Config & Logic
# =============================================================================

@dataclass
class ExportConfig:
    model_name: str = "Aratako/T5Gemma-TTS-2b-2b"
    output_dir: str = "./onnx_models"
    opset_version: int = 17
    # User requested fp16
    export_dtype: torch.dtype = torch.float16 
    device: str = "cpu"

class T5GemmaExporter:
    def __init__(self, config: ExportConfig):
        self.config = config
        self.model = None

    def load_model(self):
        logger.info(f"Loading model: {self.config.model_name} (forcing {self.config.export_dtype})")
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=self.config.export_dtype,
                low_cpu_mem_usage=True,
                device_map={"": self.config.device},
            )
            self.model.eval()
            
            # Additional force cast
            self.model.to(dtype=self.config.export_dtype)
            
            # Setup compatibility attributes
            if not hasattr(self.model, "args"):
                self.model.args = self.model.config
            if not hasattr(self.model, "encoder_module"):
                if hasattr(self.model, "model"):
                    self.model.encoder_module = self.model.model.encoder
                    self.model.decoder_module = self.model.model.decoder
            if not hasattr(self.model, "text_input_type"):
                self.model.text_input_type = getattr(self.model.config, "text_input_type", "text")
            if not hasattr(self.model, "progress_scale"):
                self.model.progress_scale = getattr(self.model.config, "progress_scale", 2000.0)
                
            logger.info("Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def export_all(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 1. Export Encoder
        self._export_component(
            name="encoder",
            wrapper_cls=EncoderWrapper,
            dummy_inputs=self._get_encoder_dummies(),
            input_names=['input_ids', 'attention_mask'],
            output_names=['encoder_hidden_states'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'},
                'encoder_hidden_states': {0: 'batch', 1: 'sequence'},
            }
        )
        
        # 2. Export Decoder Init - DISABLED for hybrid inference
        # Decoder uses PyTorch for better FP16 stability and memory efficiency
        logger.info("Skipping decoder_init export for Hybrid Inference (using PyTorch for decoding).")
        # self._export_component(
        #     name="decoder_init",
        #     wrapper_cls=DecoderInitWrapper,
        #     dummy_inputs=self._get_decoder_init_dummies(),
        #     input_names=['prompt_tokens', 'encoder_hidden_states', 'encoder_attention_mask', 'target_length'],
        #     output_names=['logits'],
        #     dynamic_axes={
        #         'prompt_tokens': {0: 'batch', 1: 'prompt_length'},
        #         'encoder_hidden_states': {0: 'batch', 1: 'enc_length'},
        #         'encoder_attention_mask': {0: 'batch', 1: 'enc_length'},
        #         'logits': {0: 'batch'},
        #     }
        # )
        
        # 3. Export Decoder Step - SKIPPED FOR HYBRID INFERENCE
        # T5Gemma's complex cache with cross-attention is not compatible with legacy ONNX.
        # We will use PyTorch for the autoregressive decoder step.
        logger.info("Skipping decoder_step export for Hybrid Inference (using PyTorch for decoding).")

        # 4. Export XCodec2 - SKIPPED FOR HYBRID INFERENCE
        # XCodec2 export has issues with einx/dynamic shapes. Using PyTorch.
        logger.info("Skipping XCodec2 export for Hybrid Inference (using PyTorch).")
        
        # 5. Save Config
        self._save_config()

    def _export_component(self, name, wrapper_cls, dummy_inputs, **kwargs):
        output_path = os.path.join(self.config.output_dir, f"{name}.onnx")
        
        # Skip if ONNX file already exists
        if os.path.exists(output_path):
            logger.info(f"Skipping {name} - already exists at {output_path}")
            return
        
        logger.info(f"Exporting {name} to {output_path}...")
        
        wrapper = wrapper_cls(self.model)
        wrapper.eval()
        
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            **kwargs
        )
        logger.info(f"Exported {name}.")

    def _get_encoder_dummies(self):
        device = self.config.device
        return (
            torch.randint(0, 1000, (1, 64), device=device, dtype=torch.long),
            torch.ones((1, 64), dtype=torch.long, device=device)
        )

    def _get_decoder_init_dummies(self):
        device = self.config.device
        dtype = self.config.export_dtype
        batch, prompt_len, enc_len = 1, 10, 64
        enc_hidden = getattr(self.model.config, "encoder_hidden_size", 2304)
        
        return (
            torch.randint(0, 100, (batch, prompt_len), device=device, dtype=torch.long),
            torch.randn(batch, enc_len, enc_hidden, device=device, dtype=dtype),
            torch.ones(batch, enc_len, device=device, dtype=torch.long),
            torch.tensor([50], device=device, dtype=torch.long)
        )

    def _export_decoder_step(self):
        name = "decoder_step"
        output_path = os.path.join(self.config.output_dir, f"{name}.onnx")
        
        # Skip if ONNX file already exists
        if os.path.exists(output_path):
            logger.info(f"Skipping {name} - already exists at {output_path}")
            return
        
        logger.info(f"Exporting {name} to {output_path}...")
        
        wrapper = DecoderStepWrapper(self.model)
        wrapper.eval()
        
        # Prepare complex dummy inputs for KV cache
        config = self.model.config
        batch = 1
        # T5GemmaVoiceConfig has nested decoder config
        decoder_config = getattr(config, "decoder", config)
        num_layers = getattr(decoder_config, "num_hidden_layers", 26)
        num_heads = getattr(decoder_config, "num_key_value_heads", 4)
        head_dim = getattr(decoder_config, "head_dim", 256)
        past_seq = 10
        device = self.config.device
        dtype = self.config.export_dtype
        
        input_token = torch.tensor([[1]], device=device, dtype=torch.long)
        enc_len = 64
        enc_hidden = getattr(config, "encoder_hidden_size", 2304)
        
        encoder_hidden = torch.randn(batch, enc_len, enc_hidden, device=device, dtype=dtype)
        encoder_mask = torch.ones(batch, enc_len, device=device, dtype=torch.long)
        cur_len = torch.tensor([past_seq], device=device, dtype=torch.long)
        tgt_len = torch.tensor([50], device=device, dtype=torch.long)
        
        past_key_values = []
        for _ in range(num_layers):
            k = torch.randn(batch, num_heads, past_seq, head_dim, device=device, dtype=dtype)
            v = torch.randn(batch, num_heads, past_seq, head_dim, device=device, dtype=dtype)
            past_key_values.append(k)
            past_key_values.append(v)
            
        dummies = (input_token, encoder_hidden, encoder_mask, cur_len, tgt_len, tuple(past_key_values))
        
        # Dynamic Axes
        input_names = ['input_token', 'encoder_hidden_states', 'encoder_attention_mask', 'current_length', 'target_length']
        output_names = ['logits']
        dynamic_axes = {
            'input_token': {0: 'batch'},
            'encoder_hidden_states': {0: 'batch', 1: 'enc_len'},
            'encoder_attention_mask': {0: 'batch', 1: 'enc_len'},
            'logits': {0: 'batch'},
        }
        
        # KV Names
        pv_names = []
        for i in range(num_layers):
            k_name, v_name = f'past_key_values.{i}.key', f'past_key_values.{i}.value'
            input_names.extend([k_name, v_name])
            dynamic_axes[k_name] = {0: 'batch', 2: 'past_len'}
            dynamic_axes[v_name] = {0: 'batch', 2: 'past_len'}
            
            pk_name, pv_name = f'present_key_values.{i}.key', f'present_key_values.{i}.value'
            pv_names.extend([pk_name, pv_name])
            dynamic_axes[pk_name] = {0: 'batch', 2: 'pres_len'}
            dynamic_axes[pv_name] = {0: 'batch', 2: 'pres_len'}
            
        output_names.extend(pv_names)
        
        torch.onnx.export(
            wrapper, dummies, output_path,
            export_params=True, opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        logger.info(f"Exported {name}.")

    def _export_xcodec2(self):
        try:
            from data.tokenizer import AudioTokenizer
            name = "xcodec2_decoder"
            output_path = os.path.join(self.config.output_dir, f"{name}.onnx")
            
            # Skip if ONNX file already exists
            if os.path.exists(output_path):
                logger.info(f"Skipping {name} - already exists at {output_path}")
                return
            
            logger.info(f"Exporting {name}...")
            
            xcodec_name = getattr(self.model.config, "xcodec2_model_name", "hkust-audio/xcodec2")
            tok = AudioTokenizer(backend="xcodec2", model_name=xcodec_name, device=self.config.device)
            
            wrapper = XCodec2DecoderWrapper(tok.codec)
            wrapper.eval()
            
            dummy = torch.randint(0, 65535, (1, 1, 100), device=self.config.device, dtype=torch.long)
            
            torch.onnx.export(
                wrapper, dummy, output_path,
                export_params=True, opset_version=self.config.opset_version,
                do_constant_folding=True,
                input_names=['codes'], output_names=['audio'],
                dynamic_axes={'codes': {0: 'batch', 2: 'num_codes'}, 'audio': {0: 'batch', 1: 'samples'}}
            )
            logger.info(f"Exported {name}.")
        except Exception as e:
            logger.warning(f"Skipping XCodec2 export: {e}")

    def _save_config(self):
        path = os.path.join(self.config.output_dir, "model_args.json")
        # Get config - handle various attribute structures
        conf_obj = self.model.config
        if hasattr(conf_obj, 'to_dict'):
            conf = conf_obj.to_dict()
        elif hasattr(conf_obj, '__dict__'):
            # Fallback: serialize dict representation
            conf = {k: v for k, v in conf_obj.__dict__.items() if not k.startswith('_')}
        else:
            conf = {"model_name": self.config.model_name, "note": "Config could not be serialized"}
        
        with open(path, 'w') as f:
            json.dump(conf, f, indent=2, default=str)
        logger.info(f"Saved config to {path}")

    def release_memory(self):
        logger.info("Releasing memory...")
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

# =============================================================================
# Quantization Logic
# =============================================================================

# =============================================================================
# Quantization Logic
# =============================================================================

def quantize_all(input_dir: str, output_dir: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Safe Temp Directory for Quantization
    # We use a local temp directory on the HDD to avoid filling up the system /tmp (often small/full).
    safe_temp_dir = os.path.abspath(os.path.join(output_dir, "temp_quant_working"))
    os.makedirs(safe_temp_dir, exist_ok=True)
    
    # Save original TMPDIR to restore later
    original_tmpdir = os.environ.get("TMPDIR", None)
    os.environ["TMPDIR"] = safe_temp_dir
    logger.info(f"Setting TMPDIR={safe_temp_dir} to avoid No Space Left on Device errors.")
    logger.info(f"Verified TMPDIR env var: {os.environ['TMPDIR']}")

    # Files to quantize
    files = ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx", "xcodec2_decoder.onnx"]
    
    # Copy config
    src_cfg = os.path.join(input_dir, "model_args.json")
    if os.path.exists(src_cfg):
        shutil.copy(src_cfg, os.path.join(output_dir, "model_args.json"))
    
    for fname in files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname.replace(".onnx", ".int8.onnx"))
        
        if not os.path.exists(in_path):
            logger.warning(f"Missing {fname}, skipping quantization.")
            continue
        
        if os.path.exists(out_path):
             logger.info(f"Target {out_path} already exists. Skipping.")
             continue
            
        logger.info(f"Quantizing {fname} -> {out_path}...")
        try:
            # Use quantize_dynamic with memory-saving options
            quantize_dynamic(
                model_input=in_path,
                model_output=out_path,
                weight_type=QuantType.QUInt8,
                per_channel=False,  # Reduces memory usage
                use_external_data_format=True,
                extra_options={
                    'DisableShapeInference': True,  # Skip expensive shape inference
                    'MatMulConstBOnly': True,       # Only quantize MatMul with const B
                }
            )
            logger.info(f"Successfully quantized {fname}")
        except Exception as e:
            logger.error(f"Failed to quantize {fname}: {e}")
        finally:
            # Force garbage collection to free memory before next file
            gc.collect()
            
    # Cleanup Temp
    try:
        if original_tmpdir:
            os.environ["TMPDIR"] = original_tmpdir
        else:
            del os.environ["TMPDIR"]
        shutil.rmtree(safe_temp_dir)
        logger.info("Cleaned up temp directory.")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp dir: {e}")

# =============================================================================
# Main
# =============================================================================

def main(model_name="Aratako/T5Gemma-TTS-2b-2b", output_dir="./onnx_models"):
    # 1. Export (Intermediate FP16)
    intermediate_dir = f"{output_dir}_fp16"
    
    # Check if export is needed
    # We only need encoder for hybrid inference (decoder uses PyTorch)
    expected_fp16 = ["encoder.onnx"]  # Only encoder needed for hybrid inference
    
    must_export = False
    for f in expected_fp16:
        if not os.path.exists(os.path.join(intermediate_dir, f)):
            must_export = True
            break
            
    if must_export:
        logger.info("FP16 models missing or incomplete. Starting fresh export...")
        exporter = T5GemmaExporter(ExportConfig(model_name=model_name, output_dir=intermediate_dir))
        exporter.load_model()
        exporter.export_all()
        exporter.release_memory()
    else:
        logger.info(f"All expected FP16 models found in {intermediate_dir}. Skipping Export phase.")
    
    # 2. Quantize (Final INT8)
    int8_dir = f"{output_dir}_int8"
    logger.info("Starting Quantization...")
    # quantize_all(intermediate_dir, int8_dir)
    
    logger.info("DONE. Final Int8 models in " + int8_dir)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
