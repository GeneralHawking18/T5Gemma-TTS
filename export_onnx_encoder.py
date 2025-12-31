#!/usr/bin/env python3
"""
ONNX Export Script for T5Gemma-TTS Encoder Only.

Usage:
    python export_onnx_encoder.py --model_name "Aratako/T5Gemma-TTS-2b-2b"
"""

# =============================================================================
# CRITICAL: Set TMPDIR BEFORE any imports to avoid "No space left on device"
# =============================================================================
import os
_SAFE_TMPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_working"))
os.makedirs(_SAFE_TMPDIR, exist_ok=True)
os.environ["TMPDIR"] = _SAFE_TMPDIR
os.environ["TEMP"] = _SAFE_TMPDIR
os.environ["TMP"] = _SAFE_TMPDIR
import tempfile
tempfile.tempdir = _SAFE_TMPDIR
print(f"[INIT] TMPDIR set to: {_SAFE_TMPDIR}")

import gc
import json
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from onnx_modules import EncoderWrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Monkeypatch for ONNX compatibility (SDPA) - Non-causal for Encoder
# =============================================================================
def _apply_sdpa_monkeypatch() -> None:
    import transformers.masking_utils
    
    def _custom_no_vmap_sdpa_mask(
        batch_size, 
        cache_position, 
        kv_length, 
        kv_offset=0, 
        mask_function=None, 
        attention_mask=None, 
        **kwargs
    ):
        """
        Custom SDPA mask that supports both causal (decoder) and non-causal (encoder).
        For encoder: bidirectional attention (all positions attend to all).
        """
        device = cache_position.device
        q_length = cache_position.shape[0]
        
        # Check if this is a causal (decoder) context or non-causal (encoder)
        # Encoder uses full bidirectional attention
        is_causal = kwargs.get('is_causal', True)
        
        if is_causal:
            # Causal mask for decoder
            query_idx = cache_position.unsqueeze(1)
            key_idx = torch.arange(kv_length, device=device).unsqueeze(0) + kv_offset
            causal_mask = query_idx >= key_idx
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        else:
            # Non-causal (full attention) for encoder - all True
            causal_mask = torch.ones(
                (batch_size, 1, q_length, kv_length), 
                dtype=torch.bool, 
                device=device
            )
        
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
            
    logger.info("Monkeypatched transformers.masking_utils.sdpa_mask for encoder (non-causal).")

_apply_sdpa_monkeypatch()


# =============================================================================
# Exporter Config & Logic
# =============================================================================

@dataclass
class ExportConfig:
    model_name: str = "Aratako/T5Gemma-TTS-2b-2b"
    output_dir: str = "./onnx_models_fp16"
    opset_version: int = 17
    export_dtype: torch.dtype = torch.float16
    device: str = "cpu"


class EncoderExporter:
    """Exports only the encoder part of T5Gemma to ONNX."""
    
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

    def export_encoder(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        output_path = os.path.join(self.config.output_dir, "encoder.onnx")
        
        # Skip if ONNX file already exists
        if os.path.exists(output_path):
            logger.info(f"Skipping encoder - already exists at {output_path}")
            return
        
        logger.info(f"Exporting encoder to {output_path}...")
        
        # Create wrapper
        wrapper = EncoderWrapper(self.model)
        wrapper.eval()
        
        # Prepare dummy inputs
        device = self.config.device
        dummy_inputs = (
            torch.randint(0, 1000, (1, 64), device=device, dtype=torch.long),
            torch.ones((1, 64), dtype=torch.long, device=device)
        )
        
        # Export to ONNX
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['encoder_hidden_states'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'},
                'encoder_hidden_states': {0: 'batch', 1: 'sequence'},
            }
        )
        logger.info(f"Exported encoder to {output_path}")
        
        # Save config
        self._save_config()

    def _save_config(self):
        path = os.path.join(self.config.output_dir, "model_args.json")
        conf_obj = self.model.config
        
        if hasattr(conf_obj, 'to_dict'):
            conf = conf_obj.to_dict()
        elif hasattr(conf_obj, '__dict__'):
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
# Main
# =============================================================================

def main(model_name="Aratako/T5Gemma-TTS-2b-2b", output_dir="./onnx_models_fp16"):
    """
    Export T5Gemma encoder to ONNX.
    
    Args:
        model_name: HuggingFace model name or local path
        output_dir: Output directory for ONNX files
    """
    config = ExportConfig(
        model_name=model_name,
        output_dir=output_dir
    )
    
    exporter = EncoderExporter(config)
    exporter.load_model()
    exporter.export_encoder()
    exporter.release_memory()
    
    logger.info(f"DONE. Encoder ONNX model saved to {output_dir}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
