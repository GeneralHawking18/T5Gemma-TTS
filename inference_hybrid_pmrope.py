"""
Hybrid T5Gemma-TTS Inference with PM-RoPE Support
- ONNX Encoder (FP16) for fast text encoding
- PyTorch Decoder with PM-RoPE loaded from local weights
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from typing import Optional, List, Union, Tuple
from transformers import AutoConfig, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# Add models to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.t5gemma import T5GemmaVoiceModel


# =============================================================================
# Helper Functions
# =============================================================================

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Return Bool mask [B, T] where True indicates padding."""
    max_len = max(max_len, int(lengths.max().item()))
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    return seq_range.unsqueeze(0).expand(n, max_len) >= lengths.unsqueeze(-1)


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(logits < threshold, filter_value)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


def create_args_from_json(args_path: str):
    """Create args namespace from JSON file."""
    class Args:
        pass
    
    args = Args()
    
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(args, k, v)
    else:
        # Default values
        args.t5gemma_model_name = "google/t5gemma-2b-2b-ul2"
        args.n_codebooks = 1
        args.audio_vocab_size = 65536
        args.n_special = 5
        args.empty_token = 65536
        args.eog = 65537
        args.eos = 65539
        args.audio_pad_token = 65540
        args.use_pm_rope = 1
        args.progress_scale = 2000.0
        args.codec_audio_sr = 44100
        args.encodec_sr = 50
        args.audio_max_length = 30
        args.text_input_type = "text"
        args.text_vocab_size = 0
        args.precision = "bfloat16"
        args.attn_implementation = "eager"
        args.prune_text_modules = 1
        args.freeze_t5gemma = 0
        args.use_lora = 0
        args.t5_gradient_checkpointing = 0
        args.eog_weight = 1.0
        args.special_first = 0
        args.x_sep_token = None
        args.y_sep_token = None
    
    return args


# =============================================================================
# ONNX Encoder Wrapper
# =============================================================================

class ONNXEncoderWrapper(nn.Module):
    """Wrapper that replaces PyTorch encoder with ONNX encoder."""
    
    def __init__(self, onnx_session, device, dtype):
        super().__init__()
        self.onnx_session = onnx_session
        self.device = device
        self.dtype = dtype
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs
    ):
        """Run ONNX encoder and return outputs in same format as PyTorch encoder."""
        if input_ids is not None:
            onnx_inputs = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": attention_mask.cpu().numpy().astype(np.int64) if attention_mask is not None else np.ones_like(input_ids.cpu().numpy()),
            }
        else:
            raise ValueError("inputs_embeds not supported for ONNX encoder")
        
        outputs = self.onnx_session.run(None, onnx_inputs)
        last_hidden_state = torch.from_numpy(outputs[0]).to(device=self.device, dtype=self.dtype)
        
        class EncoderOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
        
        return EncoderOutput(last_hidden_state)


# =============================================================================
# Hybrid TTS Class with Local PM-RoPE Weights
# =============================================================================

class HybridT5GemmaTTS:
    """
    Hybrid T5Gemma-TTS with:
    - ONNX Encoder (fast, low memory)
    - PyTorch Decoder with PM-RoPE loaded from local weights
    """
    
    def __init__(
        self,
        weights_dir: str = "weights",
        onnx_dir: str = "onnx_models_fp16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        print(f"[HybridTTS] Initializing on {device}, dtype: {self.dtype}...")
        
        # =================================================================
        # 1. Load ONNX Encoder
        # =================================================================
        encoder_path = os.path.join(onnx_dir, "encoder.onnx")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder ONNX not found at {encoder_path}")
        
        print(f"[HybridTTS] Loading ONNX Encoder from {encoder_path}...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        print(f"[HybridTTS] Using CPU for ONNX encoder (saves GPU memory for decoder)")
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options, providers=providers)
        
        # =================================================================
        # 2. Load Args and Create T5GemmaVoiceModel
        # =================================================================
        args_path = os.path.join(weights_dir, "model_args.json")
        print(f"[HybridTTS] Loading args from {args_path}...")
        self.args = create_args_from_json(args_path)
        
        print(f"[HybridTTS] Creating T5GemmaVoiceModel with PM-RoPE...")
        self.model = T5GemmaVoiceModel(self.args)
        self.model = self.model.to(dtype=self.dtype)
        
        # =================================================================
        # 3. Load Local PM-RoPE Weights
        # =================================================================
        weights_path = os.path.join(weights_dir, "decoder_pmrope.bin")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Decoder weights not found at {weights_path}")
        
        print(f"[HybridTTS] Loading PM-RoPE weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Load weights with relaxed matching
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[HybridTTS] Loaded weights: {len(missing)} missing, {len(unexpected)} unexpected")
        
        # Move to device
        self.model = self.model.to(device=device, dtype=self.dtype)
        self.model.eval()
        
        # Verify PM-RoPE is enabled
        pm_rope_enabled = getattr(self.model, "_pm_rope_enabled", False)
        print(f"[HybridTTS] PM-RoPE enabled: {pm_rope_enabled}")
        
        # =================================================================
        # 4. Replace Encoder with ONNX Wrapper
        # =================================================================
        print(f"[HybridTTS] Replacing encoder with ONNX wrapper...")
        
        self.onnx_encoder = ONNXEncoderWrapper(
            self.encoder_session,
            device=self.device,
            dtype=self.dtype
        )
        
        # Replace encoder_module
        self.model.encoder_module = self.onnx_encoder
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # =================================================================
        # 5. Load Tokenizer
        # =================================================================
        tokenizer_name = getattr(self.args, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2")
        print(f"[HybridTTS] Loading tokenizer from {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        # Special tokens
        self.empty_token = getattr(self.args, "empty_token", 65536)
        self.eog_token = getattr(self.args, "eog", 65537)
        self.eos_token = getattr(self.args, "eos", 65539)
        
        print(f"[HybridTTS] Initialization complete!")
    
    @torch.inference_mode()
    def generate(
        self,
        text: str,
        prompt_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 2000,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate audio tokens from text.
        
        Args:
            text: Input text
            prompt_tokens: Optional prompt audio tokens [1, T]
            max_new_tokens: Maximum tokens to generate
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            temperature: Sampling temperature
            
        Returns:
            Audio tokens [1, T]
        """
        device = self.device
        
        # 1. Tokenize text
        print(f"[Generate] Tokenizing text: '{text[:50]}...'")
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(device)
        x_lens = torch.tensor([input_ids.shape[1]], device=device)
        
        # 2. Prepare audio prompt
        if prompt_tokens is not None:
            y = prompt_tokens.to(device)
            if y.ndim == 2:
                y = y.unsqueeze(0)
        else:
            # Empty prompt - [B, T, K] format
            y = torch.zeros((1, 0, 1), dtype=torch.long, device=device)
        
        # 3. Target length estimation
        codec_sr = int(getattr(self.args, "encodec_sr", 50))
        estimated_duration = len(text) * 0.15  # seconds
        tgt_y_lens = torch.tensor([int(y.shape[1] + estimated_duration * codec_sr)], device=device)
        
        # 4. Run inference
        print(f"[Generate] Running inference with PM-RoPE...")
        start_time = time.time()
        
        concat_frames, gen_frames = self.model.inference_tts(
            x=input_ids,
            x_lens=x_lens,
            y=y,
            tgt_y_lens=tgt_y_lens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stop_repetition=3,
            silence_tokens=[],
        )
        
        gen_time = time.time() - start_time
        num_tokens = gen_frames.shape[-1]
        tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0
        print(f"[Generate] Generated {num_tokens} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        return gen_frames.squeeze(0)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hybrid T5Gemma-TTS with Local PM-RoPE Weights")
    print("=" * 60)
    
    try:
        tts = HybridT5GemmaTTS(
            weights_dir="weights",
            onnx_dir="onnx_models_fp16",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Test generation
        text = "こんにちは、これはテストです。"
        print(f"\nGenerating audio for: '{text}'")
        
        tokens = tts.generate(
            text=text,
            max_new_tokens=500,
            top_k=50,
            temperature=1.0,
        )
        
        print(f"\n✅ Generated tokens shape: {tokens.shape}")
        print(f"Token values (first 10): {tokens[0, :10].tolist() if tokens.shape[1] > 0 else 'empty'}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Test failed: {e}")
