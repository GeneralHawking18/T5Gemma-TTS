#!/usr/bin/env python3
"""
Export T5Gemma-TTS model to ONNX format for CPU inference.

This script exports the model in two parts:
1. Encoder - encodes text to memory states
2. Decoder - generates audio tokens step-by-step with KV cache

Usage:
    python export_onnx.py \
        --model_name bundle_step64000_infer \
        --model_root . \
        --output_dir ./onnx_models
"""

import os
import torch
import torch.nn as nn
from argparse import Namespace
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

############################################################
# Wrapper modules for ONNX export
############################################################

class EncoderWrapper(nn.Module):
    """Wrapper for the encoder part of T5Gemma for ONNX export."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.encoder = model.encoder_module
        self.text_input_type = model.text_input_type
        self.text_embedding = model.text_embedding
        self.text_dropout = model.text_dropout
        self.progress_scale = model.progress_scale
        
    def _build_position_ids(self, x_lens: torch.Tensor, max_len: int, device) -> torch.Tensor:
        """Build PM-RoPE position IDs."""
        lengths = x_lens.to(device=device)
        pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
        denom = (lengths.clamp(min=2).to(torch.float32) - 1.0)[:, None]
        position_ids = pos / denom * self.progress_scale
        mask = pos < lengths[:, None]
        return position_ids.masked_fill(~mask, 0.0)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] - tokenized text
            attention_mask: [batch, seq_len] - 1 for valid tokens, 0 for padding
            
        Returns:
            encoder_hidden_states: [batch, seq_len, hidden_size]
        """
        x_lens = attention_mask.sum(dim=1)
        use_pm_rope = getattr(self.model.args, "use_pm_rope", 1)
        
        if use_pm_rope:
            position_ids = self._build_position_ids(x_lens, input_ids.shape[1], input_ids.device)
        else:
            position_ids = None
            
        if self.text_input_type == "text":
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            x_embeds = self.text_dropout(self.text_embedding(input_ids))
            encoder_outputs = self.encoder(
                inputs_embeds=x_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            
        return encoder_outputs.last_hidden_state


class DecoderInitWrapper(nn.Module):
    """Wrapper for initial decoder pass (process prompt and get initial KV cache)."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder = model.decoder_module
        self.audio_embedding = model.audio_embedding[0]
        self.audio_dropout = model.audio_dropout
        self.predict_layer = model.predict_layer[0]
        self.progress_scale = model.progress_scale
        self.args = model.args
        
    def forward(
        self,
        prompt_tokens: torch.Tensor,  # [batch, prompt_len]
        encoder_hidden_states: torch.Tensor,  # [batch, enc_len, hidden]
        encoder_attention_mask: torch.Tensor,  # [batch, enc_len]
        target_length: torch.Tensor,  # [batch] - estimated total length
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Process audio prompt and return initial KV cache.
        
        Returns:
            logits: [batch, 1, vocab_size] - logits for next token
            past_key_values: tuple of KV cache tensors
        """
        batch_size = prompt_tokens.shape[0]
        device = prompt_tokens.device
        
        # Prepend BOS token
        bos = torch.full(
            (batch_size, 1),
            self.args.empty_token,
            dtype=torch.long,
            device=device,
        )
        tokens = torch.cat([bos, prompt_tokens], dim=1)  # [B, T+1]
        
        # Embed audio tokens
        embedded_y = self.audio_embedding(tokens)
        embedded_y = self.audio_dropout(embedded_y)
        
        cur_len = embedded_y.shape[1]
        decoder_attention_mask = torch.ones(
            (batch_size, cur_len), dtype=torch.long, device=device
        )
        
        # Build PM-RoPE position ids
        use_pm_rope = getattr(self.args, "use_pm_rope", 1)
        pm_kwargs = {}
        
        if use_pm_rope:
            x_lens = encoder_attention_mask.sum(dim=1)
            max_len = encoder_hidden_states.shape[1]
            enc_pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
            denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
            encoder_position_ids = enc_pos / denom * self.progress_scale
            mask = enc_pos < x_lens[:, None]
            encoder_position_ids = encoder_position_ids.masked_fill(~mask, 0.0)
            
            # Decoder position ids
            est_total = target_length.float() + 1  # account for BOS
            base = torch.arange(cur_len, device=device, dtype=torch.float32).unsqueeze(0)
            decoder_position_ids = base / (est_total[:, None].clamp(min=2) - 1) * self.progress_scale
            
            pm_kwargs["position_ids"] = decoder_position_ids
            pm_kwargs["pm_decoder_position_ids"] = decoder_position_ids
            pm_kwargs["pm_encoder_position_ids"] = encoder_position_ids
        else:
            pm_kwargs["position_ids"] = None
            
        # Run decoder
        decoder_outputs = self.decoder(
            inputs_embeds=embedded_y,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            **pm_kwargs,
        )
        
        # Get logits for next token prediction
        last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
        logits = self.predict_layer(last_hidden)
        
        return logits, decoder_outputs.past_key_values


class DecoderStepWrapper(nn.Module):
    """Wrapper for single decoder step with KV cache."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder = model.decoder_module
        self.audio_embedding = model.audio_embedding[0]
        self.audio_dropout = model.audio_dropout
        self.predict_layer = model.predict_layer[0]
        self.progress_scale = model.progress_scale
        self.args = model.args
        
    def forward(
        self,
        input_token: torch.Tensor,  # [batch, 1]
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        current_length: torch.Tensor,  # [batch] - current sequence length
        target_length: torch.Tensor,  # [batch] - estimated total length
        past_key_values: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Single autoregressive step.
        
        Returns:
            logits: [batch, 1, vocab_size]
            new_past_key_values: updated KV cache
        """
        device = input_token.device
        batch_size = input_token.shape[0]
        
        # Embed single token
        embedded = self.audio_embedding(input_token)
        embedded = self.audio_dropout(embedded)
        
        # Full attention mask covering all seen tokens
        attention_mask = torch.ones(
            (batch_size, current_length[0].item() + 1),
            dtype=torch.long,
            device=device
        )
        
        # Build position ids
        use_pm_rope = getattr(self.args, "use_pm_rope", 1)
        pm_kwargs = {}
        
        if use_pm_rope:
            x_lens = encoder_attention_mask.sum(dim=1)
            max_len = encoder_hidden_states.shape[1]
            enc_pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
            denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
            encoder_position_ids = enc_pos / denom * self.progress_scale
            mask = enc_pos < x_lens[:, None]
            encoder_position_ids = encoder_position_ids.masked_fill(~mask, 0.0)
            
            # Single position for this step
            cur_len = current_length.float()
            est_total = target_length.float() + 1
            new_pos_value = cur_len / (est_total.clamp(min=2) - 1) * self.progress_scale
            new_pos_value = new_pos_value.clamp(max=self.progress_scale)
            pos_1 = new_pos_value.unsqueeze(-1)  # [B, 1]
            
            pm_kwargs["position_ids"] = pos_1
            pm_kwargs["pm_decoder_position_ids"] = pos_1
            pm_kwargs["pm_encoder_position_ids"] = encoder_position_ids
        else:
            pm_kwargs["position_ids"] = None
            
        decoder_outputs = self.decoder(
            inputs_embeds=embedded,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            **pm_kwargs,
        )
        
        logits = self.predict_layer(decoder_outputs.last_hidden_state)
        
        return logits, decoder_outputs.past_key_values


############################################################
# Export functions
############################################################

def export_encoder(
    model,
    output_path: str,
    opset_version: int = 17,
):
    """Export encoder to ONNX."""
    logging.info("Exporting encoder to ONNX...")
    
    encoder_wrapper = EncoderWrapper(model)
    encoder_wrapper.eval()
    
    # Sample inputs
    batch_size = 1
    seq_len = 64
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    
    # Export
    torch.onnx.export(
        encoder_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['encoder_hidden_states'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'encoder_hidden_states': {0: 'batch', 1: 'sequence'},
        },
    )
    
    logging.info(f"Encoder exported to {output_path}")


def export_xcodec2(
    audio_tokenizer,
    output_dir: str,
    opset_version: int = 17,
):
    """Export XCodec2 encoder and decoder to ONNX."""
    logging.info("Exporting XCodec2 audio tokenizer...")
    
    device = audio_tokenizer.device
    
    # Export encoder (audio -> codes)
    class XCodec2Encoder(nn.Module):
        def __init__(self, codec):
            super().__init__()
            self.codec = codec
            self.sample_rate = codec.config.encoder_sample_rate if hasattr(codec.config, 'encoder_sample_rate') else 16000
            
        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            # waveform: [batch, samples]
            codes = self.codec.encode_code(input_waveform=waveform, sample_rate=self.sample_rate)
            return codes
    
    # Export decoder (codes -> audio)
    class XCodec2Decoder(nn.Module):
        def __init__(self, codec):
            super().__init__()
            self.codec = codec
            
        def forward(self, codes: torch.Tensor) -> torch.Tensor:
            # codes: [batch, num_codes] or [batch, 1, num_codes]
            if codes.ndim == 2:
                codes = codes.unsqueeze(1)
            codes = codes.long()
            recon = self.codec.decode_code(codes)
            return recon
    
    # Export decoder only (usually what we need for inference)
    decoder_wrapper = XCodec2Decoder(audio_tokenizer.codec)
    decoder_wrapper.eval()
    
    dummy_codes = torch.randint(0, 65535, (1, 1, 100), device=device, dtype=torch.long)
    
    decoder_path = os.path.join(output_dir, "xcodec2_decoder.onnx")
    
    try:
        torch.onnx.export(
            decoder_wrapper,
            dummy_codes,
            decoder_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['codes'],
            output_names=['audio'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'num_codes'},
                'audio': {0: 'batch', 1: 'samples'},
            },
        )
        logging.info(f"XCodec2 decoder exported to {decoder_path}")
    except Exception as e:
        logging.warning(f"Failed to export XCodec2: {e}")
        logging.warning("You may need to use the original XCodec2 for audio decoding.")


############################################################
# Main export script
############################################################

def main(
    model_name: str = "bundle_step64000_infer",
    model_root: str = ".",
    output_dir: str = "./onnx_models",
    opset_version: int = 17,
    export_audio_codec: bool = True,
):
    """
    Export T5Gemma-TTS model to ONNX.
    
    Args:
        model_name: Name of the model bundle (without .pth)
        model_root: Directory containing the model bundle
        output_dir: Directory to save ONNX models
        opset_version: ONNX opset version
        export_audio_codec: Whether to also export XCodec2
    """
    import fire
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    torch.serialization.add_safe_globals([Namespace])
    device = "cpu"  # Export on CPU for compatibility
    
    ckpt_fn = os.path.join(model_root, model_name + ".pth")
    if not os.path.exists(ckpt_fn):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_fn}")
        
    logging.info(f"Loading model from {ckpt_fn}")
    bundle = torch.load(ckpt_fn, map_location=device, weights_only=True)
    args = bundle["args"]
    
    # Import model
    from models.t5gemma import T5GemmaVoiceModel
    
    model = T5GemmaVoiceModel(args)
    model.load_state_dict(bundle["model"], strict=False)
    model.to(device)
    model.eval()
    
    del bundle
    
    # Export encoder
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    export_encoder(model, encoder_path, opset_version)
    
    # Export audio codec if requested
    if export_audio_codec:
        from data.tokenizer import AudioTokenizer
        
        audio_tokenizer = AudioTokenizer(
            backend="xcodec2",
            model_name=getattr(args, "xcodec2_model_name", None),
            device=device,
        )
        export_xcodec2(audio_tokenizer, output_dir, opset_version)
    
    # Save model args for inference
    import json
    
    args_dict = {}
    for key in dir(args):
        if not key.startswith('_'):
            val = getattr(args, key)
            if isinstance(val, (int, float, str, bool, list, dict, type(None))):
                args_dict[key] = val
    
    args_path = os.path.join(output_dir, "model_args.json")
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=2)
    logging.info(f"Model args saved to {args_path}")
    
    logging.info("=" * 60)
    logging.info("ONNX export completed!")
    logging.info(f"Output directory: {output_dir}")
    logging.info("")
    logging.info("NOTE: The full decoder export with KV cache is complex and")
    logging.info("may require additional work. For CPU inference, consider:")
    logging.info("1. Using PyTorch with torch.set_num_threads() for CPU optimization")
    logging.info("2. Using torch.compile() for potential speedups")
    logging.info("3. Using ONNX Runtime with the encoder + custom decoder loop")
    logging.info("")
    logging.info("See inference_onnx.py for a hybrid inference example.")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
