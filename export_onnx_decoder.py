#!/usr/bin/env python3
"""
Export T5Gemma decoder to ONNX.
Generates:
1. decoder_init.onnx (Prompt processing / Prefill)
2. decoder_step.onnx (Autoregressive step)
"""
import os
_SAFE_TMPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_working_dec"))
os.makedirs(_SAFE_TMPDIR, exist_ok=True)
os.environ["TMPDIR"] = _SAFE_TMPDIR
os.environ["TEMP"] = _SAFE_TMPDIR
os.environ["TMP"] = _SAFE_TMPDIR

import gc
import json
import logging
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Import wrappers
from onnx_modules import DecoderInitWrapper, DecoderStepWrapper

def export_decoder():
    model_name = "Aratako/T5Gemma-TTS-2b-2b"
    output_dir = "./onnx_models_fp16_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading model: {model_name}...")
    # Use bfloat16 for loading if supported, else float16
    dtype = torch.float16 # ONNX export usually prefers fp16 or fp32
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cpu", # Export on CPU to save GPU memory
    )
    model.eval()
    
    # Ensure config attributes
    cfg = model.config
    if not hasattr(model, "args"):
        model.args = cfg
    if not hasattr(model, "decoder_module"):
        if hasattr(model, "backbone"):
            model.decoder_module = model.backbone.model.decoder
        elif hasattr(model, "model"):
            model.decoder_module = model.model.decoder
    # Audio embedding
    if not hasattr(model, "audio_embedding"):
        # T5GemmaVoice uses model.audio_embedding
        pass # Should be there if loaded correctly
        
    # Get params
    hidden_size = cfg.decoder.hidden_size
    num_layers = cfg.decoder.num_hidden_layers
    num_heads = cfg.decoder.num_attention_heads
    num_kv_heads = getattr(cfg.decoder, "num_key_value_heads", num_heads)
    head_dim = getattr(cfg.decoder, "head_dim", hidden_size // num_heads)
    
    logger.info(f"Model params: Layers={num_layers}, Heads={num_heads}, KV_Heads={num_kv_heads}, Dim={head_dim}")

    # ========================================================================
    # 1. Export Decoder Init (Prefill)
    # ========================================================================
    logger.info("Preparing Decoder Init Wrapper...")
    init_wrapper = DecoderInitWrapper(model)
    
    # Dummy inputs for Init
    batch_size = 1
    prompt_len = 50 # Example prompt length
    enc_len = 64
    
    prompt_tokens = torch.randint(0, 1000, (batch_size, prompt_len), dtype=torch.long)
    encoder_hidden = torch.randn(batch_size, enc_len, hidden_size, dtype=dtype)
    encoder_mask = torch.ones((batch_size, enc_len), dtype=torch.long)
    target_length = torch.tensor([100], dtype=torch.long) # Estimated total length
    
    init_inputs = (prompt_tokens, encoder_hidden, encoder_mask, target_length)
    
    # Define IO names
    init_input_names = [
        "prompt_tokens", "encoder_hidden_states", "encoder_attention_mask", "target_length"
    ]
    
    # Output is logits + past_key_values
    # But DecoderInitWrapper only returns logits in current onnx_modules.py!
    # CHECK onnx_modules.py content again carefully.
    # Result: `return self.predict_layer(last_hidden)` -> ONLY LOGITS!
    # This is a problem. We need past_key_values for the next step.
    
    logger.warning("DecoderInitWrapper in onnx_modules.py only returns logits. Modifying wrapper for export...")
    
    # Patch the forward method to return past_key_values too
    def patched_init_forward(self, prompt_tokens, encoder_hidden_states, encoder_attention_mask, target_length):
        batch_size = prompt_tokens.shape[0]
        device = prompt_tokens.device
        
        # BOS
        bos = torch.full((batch_size, 1), self.args.empty_token, dtype=torch.long, device=device)
        tokens = torch.cat([bos, prompt_tokens], dim=1)
        
        embedded_y = self.audio_dropout(self.audio_embedding(tokens))
        cur_len = embedded_y.shape[1]
        
        decoder_attention_mask = torch.ones((batch_size, cur_len), dtype=torch.long, device=device)
        
        use_pm_rope = getattr(self.args, "use_pm_rope", 1)
        pm_kwargs = {}
        if use_pm_rope:
            pm_kwargs = self._build_pm_rope_kwargs(
                encoder_hidden_states, encoder_attention_mask, cur_len, target_length, device
            )
        else:
            pm_kwargs["position_ids"] = None
            
        decoder_outputs = self.decoder(
            inputs_embeds=embedded_y,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            **pm_kwargs,
        )
        
        last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
        logits = self.predict_layer(last_hidden)
        
        # Unpack cache
        past = decoder_outputs.past_key_values
        flat_past = []
        if hasattr(past, 'key_cache'): # DynamicCache
             for i in range(len(past.key_cache)):
                 flat_past.append(past.key_cache[i])
                 flat_past.append(past.value_cache[i])
        else: # Tuple
             for layer in past:
                 flat_past.append(layer[0])
                 flat_past.append(layer[1])
                 
        return logits, tuple(flat_past)

    # Apply patch
    import types
    init_wrapper.forward = types.MethodType(patched_init_forward, init_wrapper)
    
    # Define output names
    init_output_names = ["logits"]
    for i in range(num_layers):
        init_output_names.append(f"present_key_{i}")
        init_output_names.append(f"present_value_{i}")
        
    init_dynamic_axes = {
        "prompt_tokens": {0: "batch", 1: "seq_len"},
        "encoder_hidden_states": {0: "batch", 1: "enc_len"},
        "encoder_attention_mask": {0: "batch", 1: "enc_len"},
        "target_length": {0: "batch"},
        "logits": {0: "batch", 1: "seq_out"},
    }
    for i in range(num_layers):
        # Cache shape: [batch, heads, seq, dim]
        init_dynamic_axes[f"present_key_{i}"] = {0: "batch", 2: "past_seq_len"}
        init_dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "past_seq_len"}

    init_path = os.path.join(output_dir, "decoder_init.onnx")
    logger.info(f"Exporting Decoder Init to {init_path}...")
    
    try:
        torch.onnx.export(
            init_wrapper,
            init_inputs,
            init_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=init_input_names,
            output_names=init_output_names,
            dynamic_axes=init_dynamic_axes
        )
        logger.info("✅ Decoder Init exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export Decoder Init: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # 2. Export Decoder Step (Autoregressive)
    # ========================================================================
    logger.info("Preparing Decoder Step Wrapper...")
    step_wrapper = DecoderStepWrapper(model)
    
    # Dummy inputs for Step
    input_token = torch.randint(0, 1000, (batch_size, 1), dtype=torch.long)
    current_length = torch.tensor([prompt_len + 1], dtype=torch.long)
    
    # Create past key values
    past_list = []
    for _ in range(num_layers):
        # Shape: [Batch, Heads, Seq, Dim]
        k = torch.randn(batch_size, num_kv_heads, prompt_len, head_dim, dtype=dtype)
        v = torch.randn(batch_size, num_kv_heads, prompt_len, head_dim, dtype=dtype)
        past_list.append(k)
        past_list.append(v)
    past_tuple = tuple(past_list)
    
    step_inputs = (
        input_token,
        encoder_hidden,
        encoder_mask,
        current_length,
        target_length,
        past_tuple
    )
    
    step_input_names = [
        "input_token", "encoder_hidden_states", "encoder_attention_mask", 
        "current_length", "target_length"
    ]
    for i in range(num_layers):
        step_input_names.append(f"past_key_{i}")
        step_input_names.append(f"past_value_{i}")
        
    step_output_names = ["logits"]
    for i in range(num_layers):
        step_output_names.append(f"present_key_{i}")
        step_output_names.append(f"present_value_{i}")
        
    step_dynamic_axes = {
        "input_token": {0: "batch"},
        "encoder_hidden_states": {0: "batch", 1: "enc_len"},
        "encoder_attention_mask": {0: "batch", 1: "enc_len"},
        "current_length": {0: "batch"},
        "target_length": {0: "batch"},
        "logits": {0: "batch"},
    }
    for i in range(num_layers):
        step_dynamic_axes[f"past_key_{i}"] = {0: "batch", 2: "past_seq_len"}
        step_dynamic_axes[f"past_value_{i}"] = {0: "batch", 2: "past_seq_len"}
        step_dynamic_axes[f"present_key_{i}"] = {0: "batch", 2: "total_seq_len"}
        step_dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "total_seq_len"}
        
    step_path = os.path.join(output_dir, "decoder_step.onnx")
    logger.info(f"Exporting Decoder Step to {step_path}...")
    
    try:
        torch.onnx.export(
            step_wrapper,
            step_inputs,
            step_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=step_input_names,
            output_names=step_output_names,
            dynamic_axes=step_dynamic_axes
        )
        logger.info("✅ Decoder Step exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export Decoder Step: {e}")
        import traceback
        traceback.print_exc()

    logger.info("Done.")

if __name__ == "__main__":
    export_decoder()
