"""
ONNX Inference Runtime for T5Gemma-TTS.
Handles loading .onnx (.int8.onnx) models and executing generation loop.
"""

import os
import time
import logging
import numpy as np
import onnxruntime as ort
from typing import Optional, List, Dict, Any, Tuple

# Helper for sampling (can reuse PyTorch for top_k/p sampling efficiently)
import torch

try:
    from data.tokenizer import AudioTokenizer
    from inference_tts_utils import normalize_text_with_lang
except ImportError:
    pass # Will handle if modules invalid

logger = logging.getLogger(__name__)

class T5GemmaONNX:
    def __init__(self, model_dir: str, use_int8: bool = True, device: str = "cpu"):
        """
        Initialize ONNX sessions.
        model_dir: Directory containing .onnx files
        use_int8: If True, look for .int8.onnx files
        """
        self.model_dir = model_dir
        self.device = device
        
        # Suffix
        suffix = ".int8.onnx" if use_int8 else ".onnx"
        
        # Paths
        self.encoder_path = os.path.join(model_dir, "encoder" + suffix)
        self.decoder_init_path = os.path.join(model_dir, "decoder_init" + suffix)
        self.decoder_step_path = os.path.join(model_dir, "decoder_step" + suffix)
        self.xcodec2_path = os.path.join(model_dir, "xcodec2_decoder" + suffix)
        self.config_path = os.path.join(model_dir, "model_args.json")
        
        # Load Config
        import json
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        # Model Params
        self.num_layers = self.config.get("num_decoder_layers", 26)
        self.num_heads = self.config.get("num_key_value_heads", 6)
        self.head_dim = self.config.get("head_dim", 256) # 2304 / 8? Check logic
        # If head_dim not in config, infer from hidden_size / num_heads
        if "head_dim" not in self.config:
            self.head_dim = self.config.get("hidden_size", 2304) // self.config.get("num_attention_heads", 18)

        # ORT Options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        provider = "CPUExecutionProvider" # Force CPU for consistency with User Request
        
        logger.info("Loading ONNX Sessions...")
        self.sess_enc = ort.InferenceSession(self.encoder_path, sess_options, providers=[provider])
        self.sess_dec_init = ort.InferenceSession(self.decoder_init_path, sess_options, providers=[provider])
        self.sess_dec_step = ort.InferenceSession(self.decoder_step_path, sess_options, providers=[provider])
        
        # Load XCodec2 if exists
        self.sess_xcodec = None
        if os.path.exists(self.xcodec2_path):
            self.sess_xcodec = ort.InferenceSession(self.xcodec2_path, sess_options, providers=[provider])

    def generate(
        self,
        text_tokens: np.ndarray,
        target_len: int = 250,
        temp: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate audio codes autoregressively.
        """
        return self._generate_autoregressive(text_tokens, target_len, temp, top_k, top_p)

    def _generate_autoregressive(
        self,
        text_tokens: np.ndarray,
        target_len: int = 250,
        temp: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[np.ndarray, float]:

        
        start_t = time.time()
        
        # 1. Encoder
        text_mask = np.ones_like(text_tokens, dtype=np.int64)
        enc_out = self.sess_enc.run(["encoder_hidden_states"], {
            "input_ids": text_tokens.astype(np.int64),
            "attention_mask": text_mask.astype(np.int64)
        })[0]
        
        # 2. Setup KV Cache (Empty)
        # Exported decoder_step expects:
        # past_key_values.{i}.key: [batch, num_heads, past_len, head_dim]
        # We start with past_len = 0.
        
        batch_size = text_tokens.shape[0]
        past_len = 0
        
        # Initialize Cache Dict
        kv_inputs = {}
        for i in range(self.num_layers):
            # Empty tensor for initial cache? ONNX usually dislikes 0-dim if not dynamic enough.
            # But we defined dynamic axis 2 as 'past_sequence_length'.
            # Let's try passing 0-sized tensor.
            
            k = np.zeros((batch_size, self.num_heads, 0, self.head_dim), dtype=np.float32)
            v = np.zeros((batch_size, self.num_heads, 0, self.head_dim), dtype=np.float32)
            kv_inputs[f'past_key_values.{i}.key'] = k
            kv_inputs[f'past_key_values.{i}.value'] = v

        # Start Token (BOS or similar) - Arg '1' used in export dummy
        # T5Gemma usually uses 1? Checked DecoderInitWrapper: self.args.empty_token
        # Ideally need config value. Defaulting to 1.
        current_token = np.array([[1]], dtype=np.int64) 
        
        generated_codes = []
        
        # Generation Loop
        for step in range(target_len):
            # Inputs
            step_inputs = {
                "input_token": current_token,
                "encoder_hidden_states": enc_out,
                "encoder_attention_mask": text_mask.astype(np.int64),
                "current_length": np.array([past_len], dtype=np.int64),
                "target_length": np.array([target_len], dtype=np.int64)
            }
            # Add KV
            step_inputs.update(kv_inputs)
            
            # Run
            # Output names: logits, present_key_values.0.key, ...
            # We need to capture all outputs
            outputs = self.sess_dec_step.run(None, step_inputs)
            
            logits = outputs[0] # [batch, 1, vocab]
            
            # Update KV Cache from outputs
            # Outputs [1:] are K, V interleaved
            for i in range(self.num_layers):
                idx_k = 1 + i*2
                idx_v = 1 + i*2 + 1
                kv_inputs[f'past_key_values.{i}.key'] = outputs[idx_k]
                kv_inputs[f'past_key_values.{i}.value'] = outputs[idx_v]
                
            past_len += 1
            
            # Sampling (using PyTorch for convenience)
            next_token = self._sample(logits, temp, top_k, top_p)
            generated_codes.append(next_token.item())
            current_token = next_token.reshape(1, 1).numpy().astype(np.int64)
            
        dur = time.time() - start_t
        return np.array(generated_codes), dur

    def _sample(self, logits_np, temp, top_k, top_p):
        logits = torch.from_numpy(logits_np[:, -1, :]) # [batch, vocab]
        logits = logits / temp
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
            
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
            
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

    def decode_audio(self, codes: np.ndarray) -> np.ndarray:
        """Run XCodec2 decoder."""
        if self.sess_xcodec is None:
            raise RuntimeError("XCodec2 ONNX model not found")
            
        # Codes: [seq] -> [1, 1, seq]
        inp = codes.reshape(1, 1, -1).astype(np.int64)
        audio = self.sess_xcodec.run(None, {"codes": inp})[0]
        return audio[0, 0, :]
