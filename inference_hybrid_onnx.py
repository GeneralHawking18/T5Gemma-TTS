
import os
import io
import time
import base64
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from typing import Optional, List, Union, Tuple, Callable
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Helper Functions from Modeling Code
# =============================================================================

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Return Bool mask [B, T] where True indicates padding."""
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    return expanded_lengths >= lengths.unsqueeze(-1)

def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
    min_p=0.0,
    filter_value=-float("Inf"),
    min_tokens_to_keep=1,
):
    min_p_enabled = 0.0 < min_p < 1.0
    if min_p_enabled:
        probs = F.softmax(logits, dim=-1)
        indices_to_remove = probs < min_p
        if torch.all(indices_to_remove.sum(-1) < logits.size(-1)):
            logits = logits.masked_fill(indices_to_remove, filter_value)
            top_k = 0
            top_p = 1.0

    if isinstance(top_k, int) and top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value
    elif isinstance(top_k, list):
        assert len(top_k) == logits.size(
            0
        ), f"top_k list length ({len(top_k)}) must match logits.size(0) ({logits.size(0)})"
        for i in range(logits.size(0)):
            k_i = top_k[i]
            if k_i > 0:
                k_i = min(max(k_i, min_tokens_to_keep), logits.size(-1))
                row_threshold = torch.topk(logits[i], k_i, dim=-1)[0][-1]
                indices_to_remove_i = logits[i] < row_threshold
                logits[i, indices_to_remove_i] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, min_p=0.0, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, min_p=min_p)
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token

# =============================================================================
# Hybrid Inference Class
# =============================================================================

class HybridT5Gemma:
    def __init__(
        self,
        model_name: str = "Aratako/T5Gemma-TTS-2b-2b",
        onnx_dir: str = "./onnx_models_fp16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_int8: bool = True,
    ):
        self.device = device
        
        print(f"[HybridT5Gemma] Initializing on {device}...")
        
        # 1. Load PyTorch Decoder & Helper Configs
        print(f"[HybridT5Gemma] Loading PyTorch Backbone ({model_name})...")
        
        # Load to CPU first to avoid allocating Encoder on GPU
        # 1. Load Config & Verify
        from transformers import AutoConfig
        from models.t5gemma import T5GemmaVoiceModel
        
        print(f"[HybridT5Gemma] Loading Config for {model_name}...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        try:
             self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except ValueError as e:
             print(f"[Warn] AutoTokenizer failed for {model_name}: {e}")
             print(f"[Info] Retrying with backbone tokenizer...")
             backbone_name = getattr(config, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2")
             try:
                 self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)
             except Exception:
                 from transformers import T5Tokenizer
                 print(f"[Info] Fallback to generic T5TokenizerFast...")
                 # Default to t5-v1_1-xl or similiar if Aratako uses that vocab
                 self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", legacy=False)
        
        try:
             self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except ValueError as e:
             # ... (fallback code) ...
             pass
        
        # DEBUG: Check for layer_types
        print(f"[HybridT5Gemma] Config keys: {list(config.to_dict().keys())}")
        if hasattr(config, "layer_types"):
             print(f"[HybridT5Gemma] Found layer_types in config!")
        else:
             print(f"[HybridT5Gemma] MISSING layer_types in config!")

        # 2. Instantiate Empty Model (Skeleton) to save RAM
        print(f"[HybridT5Gemma] Instantiating Skeleton Model...")
        
        try:
            from unittest.mock import patch
            
            real_from_pretrained = AutoModelForSeq2SeqLM.from_pretrained
            
            def mock_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
                if pretrained_model_name_or_path == config.t5gemma_model_name:
                    print(f"[HybridT5Gemma] Intercepted backbone load: {pretrained_model_name_or_path}")
                    print(f"[HybridT5Gemma] Creating EMPTY backbone from LOCAL config to skip Encoder load...")
                    
                    # Create config for backbone using LOCAL keys to avoid hitting Gated Repo
                    from transformers import T5Config
                    import sys
    
                    cfg_dict = config.to_dict()
                    vocab_size = cfg_dict.get("vocab_size", 32128)
                    d_model = cfg_dict.get("d_model", cfg_dict.get("hidden_size", 2304))
                    
                    # Manually map keys because T5Config.from_dict might miss inherited keys or fail due to mismatch
                    backbone_config = T5Config(
                        vocab_size=vocab_size,
                        d_model=d_model,
                        d_kv=getattr(config, "d_kv", d_model // getattr(config, "num_heads", 1)),
                        d_ff=getattr(config, "d_ff", 2048),
                        num_layers=getattr(config, "num_layers", 6),
                        num_decoder_layers=getattr(config, "num_decoder_layers", getattr(config, "num_layers", 6)),
                        num_heads=getattr(config, "num_heads", 8),
                        dropout_rate=getattr(config, "dropout_rate", 0.1),
                        is_encoder_decoder=True,
                        pad_token_id=getattr(config, "pad_token_id", 0),
                        eos_token_id=getattr(config, "eos_token_id", 1),
                        feed_forward_proj=getattr(config, "feed_forward_proj", "relu"),
                        initializer_factor=getattr(config, "initializer_factor", 1.0),
                        relative_attention_num_buckets=getattr(config, "relative_attention_num_buckets", 32),
                    )
                    
                    for k, v in cfg_dict.items():
                        if not hasattr(backbone_config, k):
                            setattr(backbone_config, k, v)
                            
                    # CRITICAL FIX: Ensure layer_types exists and sliding_window logic works
                    if not hasattr(backbone_config, "layer_types"):
                        num_dec = getattr(backbone_config, "num_decoder_layers", 6)
                        backbone_config.layer_types = ["attention"] * num_dec
                        print(f"[HybridT5Gemma] Injected default layer_types for {num_dec} layers.")
                    
                    if not hasattr(backbone_config, "sliding_window"):
                        backbone_config.sliding_window = 4096 
                        
                    if not hasattr(backbone_config, "attn_logit_softcapping"):
                         backbone_config.attn_logit_softcapping = None

                    if not hasattr(backbone_config, "num_key_value_heads"):
                         backbone_config.num_key_value_heads = getattr(backbone_config, "num_heads", 8)
                    
                    if not hasattr(backbone_config, "head_dim"):
                         d_model_val = getattr(backbone_config, "d_model", 2304)
                         n_heads_val = getattr(backbone_config, "num_heads", 8)
                         backbone_config.head_dim = getattr(config, "d_kv", d_model_val // n_heads_val)

                    if not hasattr(backbone_config, "query_pre_attn_scalar"):
                         hd = getattr(backbone_config, "head_dim", 256)
                         backbone_config.query_pre_attn_scalar = hd ** -0.5
                         
                    if not hasattr(backbone_config, "attention_dropout"):
                         backbone_config.attention_dropout = 0.0

                    if not hasattr(backbone_config, "attention_bias"):
                         backbone_config.attention_bias = False

                    # Create empty backbone
                    with torch.device("meta"):
                        db_backbone = AutoModelForSeq2SeqLM.from_config(backbone_config)
                    db_backbone = db_backbone.to_empty(device="cpu") 
                    
                    if hasattr(db_backbone, "decoder") and hasattr(db_backbone.decoder, "block"):
                        db_backbone.decoder.layers = db_backbone.decoder.block
                        for i, layer in enumerate(db_backbone.decoder.block):
                            layer.config = backbone_config
                            layer.layer_idx = i
                    elif hasattr(db_backbone, "model") and hasattr(db_backbone.model.decoder, "block"):
                        db_backbone.model.decoder.layers = db_backbone.model.decoder.block
                        for i, layer in enumerate(db_backbone.model.decoder.block):
                            layer.config = backbone_config
                            layer.layer_idx = i
                    
                    return db_backbone
                return real_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

            # Apply the patch
            with patch('transformers.AutoModelForSeq2SeqLM.from_pretrained', side_effect=mock_from_pretrained): 
                self.model = T5GemmaVoiceModel(config) # config acts as args here per T5Gemma pattern
                print("[HybridT5Gemma] Skeleton Model instantiated.")

            # 4. Manual Weight Loading (The "Partial Load")
            print(f"[HybridT5Gemma] Loading weights manually into Skeleton...")
            import safetensors.torch
            from transformers.utils import cached_file
            
            # Determine weight file
            weight_files = []
            try:
                index_file = cached_file(model_name, "model.safetensors.index.json")
                if index_file:
                    import json
                    with open(index_file, "r") as f:
                         idx = json.load(f)
                    for fname in idx["weight_map"].values():
                         path = cached_file(model_name, fname)
                         if path not in weight_files: weight_files.append(path)
                else:
                    sf = cached_file(model_name, "model.safetensors")
                    if sf: weight_files.append(sf)
            except:
                pass
                
            if not weight_files:
                 try:
                     bin_f = cached_file(model_name, "pytorch_model.bin")
                     if bin_f: weight_files.append(bin_f)
                 except: pass

            if weight_files:
                 for w_file in weight_files:
                     if w_file.endswith(".safetensors"):
                         state_dict = safetensors.torch.load_file(w_file, device="cpu")
                     else:
                         state_dict = torch.load(w_file, map_location="cpu")
                     
                     if use_int8: 
                         keys = list(state_dict.keys())
                         skipped = 0
                         for k in keys:
                             if ("encoder" in k and "decoder" not in k) or "backbone.encoder" in k:
                                 del state_dict[k]
                                 skipped += 1
                         if skipped > 0:
                             print(f"   - Skipped {skipped} encoder keys in {os.path.basename(w_file)}")
                     
                     self.model.load_state_dict(state_dict, strict=False)
                 print("[HybridT5Gemma] Partial weights loaded.")
            else:
                 print("[Warn] No weights found to load manually.")

        except Exception as e:
            print(f"[Info] Optimization (Partial Loading) failed: {e}")
            print("[Info] Falling back to standard full load.")
            self.model = T5GemmaVoiceModel(config)
            
        # 5. Load ONNX Encoder
        encoder_path = os.path.join(onnx_dir, "encoder.int8.onnx" if use_int8 else "encoder.onnx")
        if not os.path.exists(encoder_path):
             print(f"[Warn] {encoder_path} not found. Trying FP16/FP32 version.")
             encoder_path = os.path.join(onnx_dir, "encoder.onnx")
            
        if os.path.exists(encoder_path):
            try:
                print(f"[HybridT5Gemma] Loading ONNX Encoder from {encoder_path}...")
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                available_providers = ort.get_available_providers()
                if device == 'cuda' and 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
                    
                self.encoder_session = ort.InferenceSession(encoder_path, sess_options, providers=providers)
                self.use_onnx_encoder = True
                print(f"[HybridT5Gemma] ONNX Encoder loaded with providers: {providers}")
            except Exception as e:
                print(f"[Warn] Failed to load ONNX Encoder: {e}")
                self.use_onnx_encoder = False
        else:
            print("[Warn] ONNX Encoder not found.")
            self.use_onnx_encoder = False

        # 6. Cleanup PyTorch Encoder
        self.decoder_module = self.model.decoder_module
        if self.use_onnx_encoder:
            print("[HybridT5Gemma] Removing PyTorch Encoder from memory...")
            self.encoder_module = None
            if hasattr(self.model, "encoder_module"): self.model.encoder_module = None
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "encoder"):
                 self.model.backbone.encoder = None
                 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            self.encoder_module = self.model.encoder_module
            
        self.audio_embedding = self.model.audio_embedding
        self.audio_dropout = self.model.audio_dropout
        self.predict_layer = self.model.predict_layer
        self.progress_scale = self.model.progress_scale
        self._build_position_ids = self.model._build_position_ids

        # Move decoder to device
        self.decoder_module.to(device)
        self.model.eval()


    @torch.inference_mode()
    def inference_tts(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        tgt_y_lens: torch.Tensor,
        top_k: Union[int, List[int]] = -100,
        top_p: float = 1.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
        stop_repetition: int = 3,
        silence_tokens: List[int] = None,
        multi_trial: List[int] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hybrid TTS Inference:
        - Encoder: ONNX (if available)
        - Decoder: PyTorch (Native)
        """
        if getattr(self.args, "n_codebooks", 1) != 1:
            raise ValueError("Hybrid inference only supports n_codebooks=1 for now.")
            
        bsz = x.shape[0]
        device = self.device
        
        # 1. ENCODER STEP
        if self.use_onnx_encoder:
            # Prepare ONNX inputs
            # T5Gemma exports usually take 'input_ids' and 'attention_mask'
            onnx_inputs = {
                "input_ids": x.cpu().numpy().astype(np.int64),
                "attention_mask": (~make_pad_mask(x_lens)).long().cpu().numpy().astype(np.int64),
            }
            
            # Run ONNX (assume first output is last_hidden_state)
            try:
                encoder_outputs = self.encoder_session.run(None, onnx_inputs)
                memory = torch.from_numpy(encoder_outputs[0]).to(device)
                
                # Safe casting to decoder dtype
                # Find first parameter of decoder to determine expected dtype
                target_dtype = next(self.decoder_module.parameters()).dtype
                memory = memory.to(target_dtype)
                
                # Create masks for decoder use
                x_padding_mask = make_pad_mask(x_lens).to(device)
                encoder_attention_mask = (~x_padding_mask).long()
                
                # PM-RoPE position IDs
                if getattr(self.args, "use_pm_rope", 1):
                     encoder_position_ids = self._build_position_ids(x_lens, x.shape[1], device)
                else:
                     encoder_position_ids = None
                     
            except Exception as e:
                print(f"[Error] ONNX Inference failed: {e}")
                raise e
                
        else:
             # PyTorch Encoder Fallback
             x_padding_mask = make_pad_mask(x_lens).to(device)
             encoder_attention_mask = (~x_padding_mask).long()
             
             if getattr(self.args, "use_pm_rope", 1):
                encoder_position_ids = self._build_position_ids(x_lens, x.shape[1], device)
             else:
                encoder_position_ids = None
                
             # Handle text input type logic from original model
             if self.model.text_input_type == "text":
                encoder_outputs = self.encoder_module(
                    input_ids=x,
                    attention_mask=encoder_attention_mask,
                    position_ids=encoder_position_ids,
                )
             else:
                x_embeds = self.model.text_dropout(self.model.text_embedding(x))
                encoder_outputs = self.encoder_module(
                    inputs_embeds=x_embeds,
                    attention_mask=encoder_attention_mask,
                    position_ids=encoder_position_ids,
                )
             memory = encoder_outputs.last_hidden_state

        # Expand encoder outputs for batch
        if batch_size > 1:
            memory = memory.expand(batch_size, -1, -1).contiguous()
            encoder_attention_mask = encoder_attention_mask.expand(batch_size, -1).contiguous()
            if encoder_position_ids is not None:
                encoder_position_ids = encoder_position_ids.expand(batch_size, -1).contiguous()
        
        # logging.info(f"Encoder Time: {time.time() - start_enc:.4f}s")

        # ---- DECODER STEP (PyTorch Native) ----
        
        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2, 1).contiguous()  # [1, 1, T]
        y_len = y.shape[-1]
        prompt_frames = kwargs.get("prompt_frames", y_len)

        if batch_size > 1:
            y = y.expand(batch_size, -1, -1).contiguous()

        target_total = None
        cutoff_limit = None
        if tgt_y_lens is not None:
            target_total = int(tgt_y_lens[0].item())
            extra_cutoff = getattr(self.args, "extra_cutoff", 5.0)
            codec_sr = int(getattr(self.args, "encodec_sr", 50))
            cutoff_limit = target_total + int(codec_sr * extra_cutoff)

        bos = torch.full(
            (batch_size, 1, 1),
            self.args.empty_token,
            dtype=torch.long,
            device=device,
        )
        cated_y = torch.cat([bos, y], dim=2)

        embedded_y = self.audio_embedding[0](cated_y[:, 0])
        embedded_y = self.audio_dropout(embedded_y)

        y_padding_mask = torch.full(
            (batch_size, embedded_y.shape[1]), False, device=device
        )
        current_length = embedded_y.shape[1]
        prompt_offset = prompt_frames + 1  # +BOS
        decoder_attention_mask = (~y_padding_mask).long()

        if target_total is not None:
            est_total = int(target_total) + 1
        elif cutoff_limit is not None:
            est_total = int(cutoff_limit)
        else:
            lookahead = getattr(self.args, "progress_lookahead_secs", 2.0)
            est_total = int(current_length + int(self.args.encodec_sr) * lookahead)
        est_total = max(est_total, current_length)

        pm_kwargs = {}
        decoder_position_ids_full = None
        if getattr(self.args, "use_pm_rope", 1):
            base = torch.arange(cur_len := embedded_y.shape[1], device=device, dtype=torch.float32).unsqueeze(0)
            decoder_position_ids_full = (
                base / max(1, est_total - 1) * self.progress_scale
            )
            if batch_size > 1:
                decoder_position_ids_full = decoder_position_ids_full.expand(batch_size, -1).contiguous()
            pm_kwargs["position_ids"] = decoder_position_ids_full
            pm_kwargs["pm_decoder_position_ids"] = decoder_position_ids_full
            pm_kwargs["pm_encoder_position_ids"] = encoder_position_ids
        else:
            pm_kwargs["position_ids"] = None

        # --- Decoder Init ---
        decoder_outputs = self.decoder_module(
            inputs_embeds=embedded_y,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            **pm_kwargs,
        )
        last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
        past_key_values = decoder_outputs.past_key_values

        generated_tokens: List[torch.Tensor] = []
        cur_num_gen = 0
        prev_tokens = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        consec_silence_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        first_input_len = int(x_lens[0].item())
        text_mode = getattr(self.args, "text_input_type", "text") == "text"
        frames_per_token_cap = getattr(self.args, "text_guard_frames_per_token", 0)
        extra_cutoff_val = getattr(self.args, "extra_cutoff", 5)

        # --- Decoder Loop ---
        while not finished.all():
            logits = self.predict_layer[0](last_hidden).squeeze(1)

            effective_length = max(0, current_length - prompt_offset)

            if effective_length == 0:
                logits[:, eog_inference] = -1e9
            
            if isinstance(top_k, list):
                kk = top_k[min(len(top_k) - 1, cur_num_gen)]
            else:
                kk = top_k

            if cur_num_gen <= self.args.encodec_sr // 5:
                logits[:, eog_inference] = -10000.0

            # Repetition Penalty
            if stop_repetition > 0 and silence_tokens:
                for sil_tok in silence_tokens:
                    mask = (prev_tokens == sil_tok) & (consec_silence_counts > stop_repetition)
                    if mask.any():
                        penalty = (consec_silence_counts[mask] - (stop_repetition - 1)).float()
                        neg_mask = logits[mask, sil_tok] < 0
                        logits[mask, sil_tok] = torch.where(
                            neg_mask,
                            logits[mask, sil_tok] * penalty,
                            logits[mask, sil_tok] / penalty,
                        )

            tokens = topk_sampling(
                logits,
                top_k=kk,
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
            ).squeeze(-1)

            should_force_stop = (tokens == eog_inference) | (logits.argmax(dim=-1) == eog_inference)

            if not text_mode:
                token_budget = first_input_len * max(1, int(self.args.encodec_sr) // 4)
                should_force_stop |= (effective_length > token_budget)
            elif frames_per_token_cap > 0:
                token_budget = max(1, first_input_len) * frames_per_token_cap
                should_force_stop |= (effective_length > token_budget)

            if target_total is not None:
                time_budget = target_total - prompt_offset + int(self.args.encodec_sr) * extra_cutoff_val
                if cur_num_gen > time_budget:
                    should_force_stop[:] = True

            tokens = torch.where(should_force_stop, torch.full_like(tokens, eog_inference), tokens)

            for sil_tok in silence_tokens:
                is_same_silence = (tokens == sil_tok) & (prev_tokens == sil_tok)
                consec_silence_counts = torch.where(
                    is_same_silence,
                    consec_silence_counts + 1,
                    torch.where(tokens == sil_tok, torch.ones_like(consec_silence_counts), torch.zeros_like(consec_silence_counts))
                )

            prev_tokens = tokens.clone()
            newly_finished = tokens == eog_inference
            finished |= newly_finished
            store_tokens = torch.where(finished & ~newly_finished, torch.full_like(tokens, eog_inference), tokens)
            generated_tokens.append(store_tokens)

            cur_num_gen += 1
            current_length += 1

            if finished.all():
                break

            samples_emb = self.audio_embedding[0](tokens.unsqueeze(1))
            samples_emb = self.audio_dropout(samples_emb)

            if getattr(self.args, "use_pm_rope", 1):
                new_pos_value = (
                    float(current_length - 1) / max(1, est_total - 1) * self.progress_scale
                )
                new_pos_value = min(new_pos_value, self.progress_scale)
                pos_1 = torch.full(
                    (batch_size, 1), new_pos_value, device=device, dtype=torch.float32
                )
                pm_kwargs = {
                    "position_ids": pos_1,
                    "pm_decoder_position_ids": pos_1,
                    "pm_encoder_position_ids": encoder_position_ids,
                }
            else:
                pm_kwargs = {"position_ids": None}

            decoder_outputs = self.decoder_module(
                inputs_embeds=samples_emb,
                # attention_mask=None, # Only causal mask needed, implicit in decoder?
                # Actually original code uses 2D attention mask for initial, 
                # but for step, we rely on cache. 
                # T5GemmaDecoderLayer uses cache_position but if not provided it appends.
                past_key_values=past_key_values,
                use_cache=True,
                encoder_hidden_states=memory,
                encoder_attention_mask=encoder_attention_mask,
                **pm_kwargs,
            )
            last_hidden = decoder_outputs.last_hidden_state
            past_key_values = decoder_outputs.past_key_values

        # Post-process
        gen_frames = torch.stack(generated_tokens, dim=-1).unsqueeze(1) # [B, 1, T]
        concat_frames = torch.cat([cated_y, gen_frames], dim=2)
        
        return concat_frames, gen_frames

    @torch.inference_mode()
    def generate_audio(
        self,
        text: str,
        ref_audio: Optional[torch.Tensor] = None, # [1, T, D] or [T, D]
        speed: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Convenience method for generating audio from text.
        text: Input text
        ref_audio: Reference audio embedding or tokens. If None, uses zeros (unconditional / default voice).
        """
        # 1. Tokenize Text
        # T5Gemma might use "user\n{text}\nmodel\n" format or similar? 
        # For now assume direct text or prompt engineering happens outside.
        # But actually T5Gemma-TTS usually takes just text.
        
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        x = inputs.input_ids.to(self.device)
        x_lens = torch.tensor([x.shape[1]], device=self.device)
        
        # 2. Prepare Reference Audio (y)
        # If no reference, simulate silence/start
        # y shape expected: [B, T, D] -> inference_tts does transpose to [B, D, T] ? 
        # Check inference_tts: "y = y.transpose(2, 1)" => Input y is [B, T, D] or [B, D, T]??
        # Line 486: y = y.transpose(2, 1).contiguous()  # [1, 1, T] ... wait?
        # If D is embedding dim, it should be [B, T, D]. Transpose makes it [B, D, T].
        
        if ref_audio is None:
            # Create minimal tensor. 
            # If using VQ audio tokens, y might be indices? 
            # inference_tts line 509: embedded_y = self.audio_embedding[0](cated_y[:, 0])
            # cated_y is [batch, 1, T]??
            # Line 507: cated_y = torch.cat([bos, y], dim=2)
            # bos is [batch, 1, 1]
            # So y must be [batch, 1, T] (indices).
            # So 'y' passed to inference_tts is AUDIO TOKENS (indices), not embeddings.
            
            # Default to 1 frame of silence/padding
            y = torch.zeros((1, 1, 1), dtype=torch.long, device=self.device)
        else:
             y = ref_audio.to(self.device)
             if y.ndim == 2:
                 y = y.unsqueeze(0) # [1, T, D]??
        
        # inference_tts expects y to be indices?
        # Let's check line 509: self.audio_embedding[0](cated_y[:, 0])
        # cated_y[:, 0] implies taking the first channel?
        # If x_codecs uses 1 codebook, we expect [B, 1, T].
        
        tgt_y_lens = None # generated length driven by text/stop tokens
        
        # 3. Inference
        concat_frames, gen_frames = self.inference_tts(
            x=x,
            x_lens=x_lens,
            y=y,
            tgt_y_lens=tgt_y_lens,
            **kwargs
        )
        
        # gen_frames is [B, 1, T] tokens.
        # Need to decode to audio?
        # This class only does Model Inference (Tokens). 
        # Audio decoding (VQ-Diffusion / Vocoder) happens outside or if we have vq_model here.
        # The prompt didn't ask for Vocoder, just "inference".
        # But returning tokens is the first step.
        
        return gen_frames

if __name__ == "__main__":
    print("Testing HybridT5Gemma module...")
    # Use FP16 ONNX since INT8 is missing/empty
    tts = HybridT5Gemma(
        model_name="Aratako/T5Gemma-TTS-2b-2b",
        onnx_dir="./onnx_models_fp16",
        use_int8=False,
        device="cuda"
    )
    print("Model loaded successfully!")
    
    # Simple generation test
    try:
        print("Generating warm-up audio tokens...")
        # Japanese test text
        text = "こんにちは、これはテストです。"
        tokens = tts.generate_audio(text)
        print(f"Generated tokens shape: {tokens.shape}")
        print("Inference finished successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Generation failed: {e}")
