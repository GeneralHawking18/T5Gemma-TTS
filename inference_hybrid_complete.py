"""
Hybrid T5Gemma-TTS Inference
- ONNX Encoder for fast text encoding
- PyTorch Decoder (T5GemmaVoiceModel) with PM-RoPE for audio generation
- Uses same inference logic as original T5GemmaVoiceModel.inference_tts
"""
import os
import json
import time
import torch
import numpy as np
import onnxruntime as ort
from typing import Optional, List, Union
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# Import T5GemmaVoiceModel (has PM-RoPE enabled)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.t5gemma import T5GemmaVoiceModel
from models.utils import topk_sampling


class HybridT5GemmaTTS:
    """
    Hybrid T5Gemma-TTS with:
    - ONNX Encoder (fast, low memory)
    - PyTorch Decoder from T5GemmaVoiceModel (PM-RoPE enabled)
    """

    def __init__(
        self,
        model_name: str = "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
        onnx_dir: str = "onnx_models_fp16_fixed",  # Use fixed encoder (corrected monkeypatch)
        decoder_weights: str = "weights/decoder_pmrope.bin",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        # NOTE: Use bfloat16 to match the 4bit model's dtype for decoder
        # ONNX encoder outputs float16, but we convert it to bfloat16 for decoder
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"[HybridTTS] Initializing on {device}, dtype: {self.dtype}...")

        # Load model args
        args_path = os.path.join(os.path.dirname(decoder_weights), "model_args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                args_dict = json.load(f)
            self.args = type("Args", (), args_dict)()
            print(f"[HybridTTS] Loaded args from {args_path}")
        else:
            raise FileNotFoundError(f"model_args.json not found at {args_path}")

        # Load tokenizer
        tokenizer_name = getattr(self.args, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2")
        print(f"[HybridTTS] Loading tokenizer from {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load ONNX Encoder
        encoder_path = os.path.join(onnx_dir, "encoder.onnx")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder ONNX not found at {encoder_path}")

        print(f"[HybridTTS] Loading ONNX Encoder from {encoder_path}...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CPUExecutionProvider']
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options, providers=providers)

        # Create T5GemmaVoiceModel (this enables PM-RoPE!)
        print(f"[HybridTTS] Creating T5GemmaVoiceModel with PM-RoPE...")
        self.model = T5GemmaVoiceModel(self.args)
        self.model = self.model.to(dtype=self.dtype)
        self.model.eval()

        # Verify PM-RoPE is enabled
        pm_rope_enabled = getattr(self.model, "_pm_rope_enabled", False)
        print(f"[HybridTTS] PM-RoPE enabled: {pm_rope_enabled}")

        # Load decoder weights
        if not os.path.exists(decoder_weights):
            raise FileNotFoundError(f"Decoder weights not found at {decoder_weights}")

        print(f"[HybridTTS] Loading decoder weights from {decoder_weights}...")
        state_dict = torch.load(decoder_weights, map_location="cpu")

        # Load weights into model (handles key mapping)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[HybridTTS] Loaded weights: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing and len(missing) < 20:
            print(f"[HybridTTS] Missing keys: {missing}")

        # Move model to device
        self.model = self.model.to(device=device)

        # Store references to key components
        self.decoder_module = self.model.decoder_module
        self.audio_embedding = self.model.audio_embedding
        self.predict_layer = self.model.predict_layer
        self.progress_scale = self.model.progress_scale

        # Special tokens
        self.empty_token = self.args.empty_token
        self.eog_token = self.args.eog
        self.eos_token = getattr(self.args, "eos", -1)
        self.encodec_sr = int(self.args.encodec_sr)

        print(f"[HybridTTS] Initialization complete!")
        print(f"[HybridTTS] Special tokens: empty={self.empty_token}, eog={self.eog_token}, eos={self.eos_token}")

    def _run_encoder(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Run ONNX encoder."""
        onnx_inputs = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        }
        outputs = self.encoder_session.run(None, onnx_inputs)
        return outputs[0]

    def _build_position_ids(self, lengths: torch.Tensor, max_len: int, device) -> torch.Tensor:
        """Build PM-RoPE position IDs (same as T5GemmaVoiceModel)."""
        lengths = lengths.to(device=device)
        pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
        denom = (lengths.clamp(min=2).float() - 1.0)[:, None]
        position_ids = pos / denom * self.progress_scale
        mask = pos < lengths[:, None]
        return position_ids.masked_fill(~mask, 0.0)

    def _load_audio_tokenizer(self):
        """Lazy load audio tokenizer on first use."""
        if not hasattr(self, '_audio_tokenizer') or self._audio_tokenizer is None:
            from data.tokenizer import AudioTokenizer
            xcodec2_model = getattr(self.args, "xcodec2_model_name", "NandemoGHS/Anime-XCodec2-44.1kHz-v2")
            print(f"[HybridTTS] Loading XCodec2 audio tokenizer: {xcodec2_model}...")
            self._audio_tokenizer = AudioTokenizer(
                backend="xcodec2",
                model_name=xcodec2_model,
                device=self.device,
            )
        return self._audio_tokenizer

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        target_duration: Optional[float] = None,
        top_k: Union[int, List[int]] = 30,
        top_p: float = 0.9,
        min_p: float = 0.0,
        temperature: float = 0.7,
        stop_repetition: int = 3,
    ) -> tuple:
        """
        High-level synthesis: text -> audio waveform.
        Returns (sample_rate, audio_numpy).

        This method handles the full pipeline:
        1. Text normalization and language detection
        2. Duration estimation
        3. Token generation
        4. Audio decoding
        """
        from inference_tts_utils import normalize_text_with_lang
        from duration_estimator import estimate_duration

        # Normalize text and detect language
        lang = None if language in {None, "", "none", "null"} else str(language)
        normalized_text, lang_code = normalize_text_with_lang(text, lang)

        print(f"[Synthesize] Text: '{normalized_text}' (lang: {lang_code})")

        # Estimate duration if not provided
        if target_duration is None:
            target_duration = estimate_duration(
                target_text=normalized_text,
                reference_speech=None,
                reference_transcript=None,
                target_lang=lang_code,
                reference_lang=lang_code,
            )
            print(f"[Synthesize] Estimated duration: {target_duration:.2f}s")
        else:
            target_duration = float(target_duration)

        # Calculate max tokens from estimated duration
        # target_total: duration * encodec_sr (no buffer) - used for PM-RoPE, same as 4bit's tgt_y_lens
        target_total_tokens = int(target_duration * self.encodec_sr)
        # effective_max_tokens: with 1-second buffer for safety margin
        effective_max_tokens = target_total_tokens + self.encodec_sr
        effective_max_tokens = min(effective_max_tokens, 2000)

        # Generate tokens
        tokens = self.generate(
            text=normalized_text,
            max_new_tokens=effective_max_tokens,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            stop_repetition=stop_repetition,
            target_total=target_total_tokens,  # Pass to generate() for correct PM-RoPE
        )

        print(f"[Synthesize] Generated {tokens.shape[1]} tokens")

        # Prepare tokens for decoding
        gen_frames = tokens.unsqueeze(1)  # [1, 1, T]

        # Strip special tokens before decoding
        eos_token = getattr(self.args, "eos", 65539)
        eog_token = getattr(self.args, "eog", 65537)
        y_sep_token = getattr(self.args, "y_sep_token", 65540)
        audio_vocab_size = getattr(self.args, "audio_vocab_size", 65536)

        # Create mask for valid audio tokens only
        mask = gen_frames < audio_vocab_size
        mask &= gen_frames.ne(eos_token)
        mask &= gen_frames.ne(eog_token)
        if y_sep_token is not None:
            mask &= gen_frames.ne(y_sep_token)

        # Find valid length and clean tokens
        valid_len = int(mask[0, 0].sum().item())
        cleaned_tokens = gen_frames[0, 0][mask[0, 0]][:valid_len]
        gen_frames_clean = cleaned_tokens.view(1, 1, -1)

        print(f"[Synthesize] Cleaned tokens: {gen_frames_clean.shape[2]} (removed {gen_frames.shape[2] - valid_len} special)")

        # Load audio tokenizer and decode
        audio_tokenizer = self._load_audio_tokenizer()
        gen_audio = audio_tokenizer.decode(gen_frames_clean)

        # Convert to numpy
        gen_audio = gen_audio[0].cpu().numpy()

        # Flatten if multi-dimensional
        if gen_audio.ndim > 1:
            gen_audio = gen_audio.flatten()

        return audio_tokenizer.sample_rate, gen_audio

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        prompt_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 2000,
        top_k: Union[int, List[int]] = 0,
        top_p: float = 0.8,
        min_p: float = 0.0,
        temperature: float = 1.0,
        stop_repetition: int = 3,
        silence_tokens: Optional[List[int]] = None,
        target_total: Optional[int] = None,  # Target tokens for PM-RoPE (same as 4bit's tgt_y_lens)
    ) -> torch.Tensor:
        """
        Generate audio tokens from text (matching original inference_tts logic).

        Args:
            target_total: Target total tokens for PM-RoPE position calculation.
                         Should be duration * encodec_sr (no buffer). If None, uses max_new_tokens.
        """
        device = self.device
        dtype = self.dtype
        silence_tokens = silence_tokens or []
        silence_set = set(silence_tokens)

        # Determine EOG token (same logic as original)
        eog_inference = self.eos_token if self.eos_token > 0 else self.eog_token

        # 1. Encode text with ONNX
        # IMPORTANT: Match 4bit tokenization - use add_special_tokens=False
        # and manually add EOS/BOS based on model config (same as inference_tts_utils.py)
        print(f"[Generate] Encoding text: '{text[:50]}...'")

        # Tokenize without special tokens (same as 4bit: encode with add_special_tokens=False)
        text_tokens = self.tokenizer.encode(text.strip(), add_special_tokens=False)

        # Add EOS/BOS based on model config (same as inference_tts_utils.py lines 285-289)
        add_eos_token = getattr(self.args, "add_eos_to_text", 0)
        add_bos_token = getattr(self.args, "add_bos_to_text", 0)

        if add_eos_token:
            text_tokens.append(add_eos_token)
        if add_bos_token:
            text_tokens = [add_bos_token] + text_tokens

        input_ids = np.array([text_tokens], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        start_time = time.time()
        encoder_output = self._run_encoder(input_ids, attention_mask)
        encoder_time = time.time() - start_time
        print(f"[Generate] Encoder time: {encoder_time:.3f}s, output shape: {encoder_output.shape}")

        # Convert to PyTorch
        memory = torch.from_numpy(encoder_output).to(device=device, dtype=dtype)
        x_lens = torch.tensor([input_ids.shape[1]], device=device)
        encoder_attention_mask = torch.from_numpy(attention_mask).to(device=device).long()

        # PM-RoPE position IDs for encoder (cross-attention)
        encoder_position_ids = self._build_position_ids(x_lens, input_ids.shape[1], device)

        # 2. Prepare decoder initial state
        batch_size = 1

        if prompt_tokens is not None:
            y = prompt_tokens.to(device=device)
            if y.ndim == 2:
                y = y.unsqueeze(1)  # [B, 1, T]
            y_len = y.shape[-1]
        else:
            y = torch.empty((batch_size, 1, 0), dtype=torch.long, device=device)
            y_len = 0

        prompt_frames = y_len

        # Prepend BOS (same as original)
        bos = torch.full((batch_size, 1, 1), self.empty_token, dtype=torch.long, device=device)
        cated_y = torch.cat([bos, y], dim=2)  # [B, 1, T+1]

        # Embed initial tokens (using first codebook embedding)
        embedded_y = self.audio_embedding[0](cated_y[:, 0])  # [B, T+1, hidden]
        embedded_y = self.model.audio_dropout(embedded_y)  # Match 4bit: apply audio_dropout

        current_length = embedded_y.shape[1]
        prompt_offset = prompt_frames + 1  # +BOS

        decoder_attention_mask = torch.ones((batch_size, current_length), dtype=torch.long, device=device)

        # Estimate total length for PM-RoPE
        # IMPORTANT: Match 4bit logic - use target_total if provided (same as tgt_y_lens)
        # 4bit uses: est_total = target_total + 1 (accounting for BOS)
        if target_total is not None:
            est_total = target_total + 1  # +1 for BOS, same as 4bit
        else:
            est_total = current_length + max_new_tokens  # Fallback
        est_total = max(est_total, current_length)  # Safety: ensure >= current
        max_gen_length = est_total + int(self.encodec_sr * 10)
        full_dec_attention_mask = torch.ones((batch_size, max_gen_length), dtype=torch.long, device=device)

        # PM-RoPE position IDs for decoder
        cur_len = embedded_y.shape[1]
        base = torch.arange(cur_len, device=device, dtype=torch.float32).unsqueeze(0)
        decoder_position_ids = base / max(1, est_total - 1) * self.progress_scale

        pm_kwargs = {
            "position_ids": decoder_position_ids,
            "pm_decoder_position_ids": decoder_position_ids,
            "pm_encoder_position_ids": encoder_position_ids,
        }

        # 3. Initial decoder forward (prefill)
        print(f"[Generate] Starting generation loop...")
        start_gen = time.time()

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

        # 4. Autoregressive generation loop (matching original sample_helper logic)
        generated_tokens: List[torch.Tensor] = []
        cur_num_gen = 0
        prev_token = -1
        consec_silence_count = 0

        while True:
            # Predict logits using predict_layer[0] (single codebook)
            logits = self.predict_layer[0](last_hidden).squeeze(0).squeeze(0)  # [vocab_size]

            # Apply same logic as original sample_helper
            effective_length = max(0, current_length - prompt_offset)

            # Prevent EOG at start
            if effective_length == 0:
                logits[eog_inference] = -1e9

            # Prevent EOG in early tokens (same as original: encodec_sr // 5)
            if cur_num_gen <= self.encodec_sr // 5:
                logits[eog_inference] = -10000.0

            # Handle stop_repetition for silence tokens (same as original)
            if (stop_repetition > 0 and prev_token in silence_set and consec_silence_count > stop_repetition):
                if logits[prev_token] < 0:
                    logits[prev_token] = logits[prev_token] * (consec_silence_count - (stop_repetition - 1))
                else:
                    logits[prev_token] = logits[prev_token] / (consec_silence_count - (stop_repetition - 1))

            # Get top_k for current step (support list like original)
            if isinstance(top_k, list):
                kk = top_k[min(len(top_k) - 1, cur_num_gen)]
            else:
                kk = top_k

            # Sample using same function as original
            token = topk_sampling(logits, top_k=kk, top_p=top_p, min_p=min_p, temperature=temperature)
            token_id = int(token.item())

            # Check for forced stop (same as original)
            should_force_stop = (token_id == eog_inference or int(torch.argmax(logits).item()) == eog_inference)

            # Time budget check
            time_budget_exceeded = cur_num_gen > max_new_tokens

            if should_force_stop or time_budget_exceeded:
                token_id = eog_inference

            # Update silence tracking
            if token_id in silence_set and token_id == prev_token:
                consec_silence_count += 1
            else:
                consec_silence_count = 0
            prev_token = token_id

            token_tensor = torch.tensor([[token_id]], device=device, dtype=torch.long)
            generated_tokens.append(token_tensor.squeeze(0))
            cur_num_gen += 1
            current_length += 1

            if token_id == eog_inference:
                break

            # Embed next token
            samples_emb = self.audio_embedding[0](token_tensor)
            samples_emb = self.model.audio_dropout(samples_emb)  # Match 4bit: apply audio_dropout

            # Update PM-RoPE position (same as original)
            new_pos_value = float(current_length - 1) / max(1, est_total - 1) * self.progress_scale
            new_pos_value = min(new_pos_value, self.progress_scale)
            pos_1 = torch.tensor([[new_pos_value]], device=device, dtype=torch.float32)

            pm_kwargs = {
                "position_ids": pos_1,
                "pm_decoder_position_ids": pos_1,
                "pm_encoder_position_ids": encoder_position_ids,
            }

            # Decoder step with KV cache
            decoder_outputs = self.decoder_module(
                inputs_embeds=samples_emb,
                attention_mask=full_dec_attention_mask[:, :current_length],
                encoder_hidden_states=memory,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                **pm_kwargs,
            )
            past_key_values = decoder_outputs.past_key_values
            last_hidden = decoder_outputs.last_hidden_state

            # Progress
            if (cur_num_gen) % 100 == 0:
                print(f"[Generate] Step {cur_num_gen}/{max_new_tokens}")

        gen_time = time.time() - start_gen
        tokens_per_sec = len(generated_tokens) / gen_time if gen_time > 0 else 0
        print(f"[Generate] Generated {len(generated_tokens)} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")

        if generated_tokens:
            generated_tensor = torch.stack(generated_tokens, dim=1)  # [1, T_gen]
        else:
            generated_tensor = torch.zeros((1, 0), dtype=torch.long, device=device)

        return generated_tensor


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hybrid T5Gemma-TTS Inference")
    print("=" * 60)

    try:
        tts = HybridT5GemmaTTS(
            model_name="Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
            onnx_dir="onnx_models_fp16_fixed",  # Use fixed encoder
            decoder_weights="weights/decoder_pmrope.bin",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Test generation
        text = "こんにちは、これはテストです。"
        print(f"\nGenerating audio for: '{text}'")

        tokens = tts.generate(
            text=text,
            max_new_tokens=500,
            top_k=0,
            top_p=0.8,
            temperature=1.0,
        )

        print(f"\nGenerated tokens shape: {tokens.shape}")
        print(f"Token values (first 10): {tokens[0, :10].tolist() if tokens.shape[1] > 0 else 'empty'}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTest failed: {e}")
