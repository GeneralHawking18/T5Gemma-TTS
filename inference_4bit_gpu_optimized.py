"""TTS inference script for T5Gemma-TTS using 4-bit quantization (GPU) + torch.compile Optimization.

This script loads the model in 4-bit using bitsandbytes and optimizes the decoder using torch.compile
for faster inference on compatible GPUs (Ampere+).
"""

import os
import random
import time
import fire
import numpy as np
import torch

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import (
    inference_one_sample,
    normalize_text_with_lang,
    get_audio_info,
    get_sample_rate,
    save_audio,
    transcribe_audio,
)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    raise ImportError("transformers is not installed. Please install it with `pip install transformers bitsandbytes accelerate`.")

# ============================================================================
# Optimization Setup
# ============================================================================
if torch.cuda.is_available():
    # Allow TF32 on Ampere+ GPUs (usually faster, negligible precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 'high' means TF32 if available, or FP32. 'medium' enables bfloat16 for matmul if available.
    torch.set_float32_matmul_precision('high')
    # Enable TorchInductor cache to speed up subsequent runs
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    
    # Add CUDA to PATH if not present
    cuda_path = "/usr/local/cuda-12.8"
    if os.path.exists(cuda_path):
        os.environ["PATH"] = f"{cuda_path}/bin:" + os.environ.get("PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

def seed_everything(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run_inference_optimized(
    reference_speech=None,
    target_text="iPhoneの新しいmodelが発売されました。",
    model_dir="Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
    reference_text=None,
    target_duration=None,
    codec_audio_sr=16000,
    codec_sr=50,
    top_k=30,
    top_p=0.9,
    min_p=0,  # default: disabled
    temperature=0.7,
    silence_tokens=None,
    multi_trial=None,
    repeat_prompt=0,
    stop_repetition=3,
    sample_batch_size=1,
    seed=1,
    output_dir="./generated_tts_4bit",
    cut_off_sec=100,
    dump_tokens=False,
    lang=None,
    # New flags - Aggressive optimization without unstable CUDA Graphs
    compile_mode="max-autotune-no-cudagraphs", 
    warmup_steps=1,
    use_flash_attn=True, # Keep Flash Attention 2
):
    seed_everything(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. 4-bit quantization and torch.compile require a GPU.")

    device = "cuda"

    print(f"[Info] Loading pre-quantized 4-bit model from {model_dir}...")
    
    # Check for Flash Attention 2 availability
    attn_impl = "eager"
    if use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("[Info] Flash Attention 2 enabled.")
        except ImportError:
            print("[Warn] flash_attn not installed. Falling back to default attention.")

    # Load pre-quantized 4-bit model with bfloat16 compute dtype for numerical stability
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # BFloat16 for better numerical stability
        attn_implementation=attn_impl,
    )
    
    model.eval()
    cfg = model.config

    # ========================================================================
    # OPTIMIZATION: torch.compile
    # ========================================================================
    if hasattr(torch, "compile") and os.environ.get("DISABLE_COMPILE", "0") != "1":
        print(f"[Info] Compiling Decoder Module with mode='{compile_mode}'...")
        try:
            # We target the 'decoder_module' specifically because compiling the entire
            # encoder-decoder model often fails due to complex control flow or is slower.
            # The autoregressive decoder loop is the bottleneck.
            if hasattr(model, "decoder_module"):
                model.decoder_module = torch.compile(
                    model.decoder_module, 
                    mode=compile_mode,
                    fullgraph=False
                )
                print("[Info] Compilation scheduled (lazy). First run will be slower.")
            else:
                # Fallback for models where structure might differ slightly
                print("[Warn] 'decoder_module' attribute not found. Attempting to locate...")
                if hasattr(model, "model") and hasattr(model.model, "decoder"):
                    model.model.decoder = torch.compile(model.model.decoder, mode=compile_mode)
                    print("[Info] model.model.decoder compiled.")
                elif hasattr(model, "decoder"):
                     model.decoder = torch.compile(model.decoder, mode=compile_mode)
                     print("[Info] model.decoder compiled.")
                else:
                    print("[Warn] Could not locate decoder for compilation. Skipping.")
        except Exception as e:
            print(f"[Warn] torch.compile failed: {e}")
            print("[Info] Continuing with standard inference.")
    else:
        print("[Info] torch.compile skipped or unavailable.")

    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Audio tokenizer (supports xcodec2 only)
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(cfg, "xcodec2_model_name", "xcodec2"),
    )
    
    # Update sample rates from config if available
    codec_audio_sr = getattr(cfg, "codec_audio_sr", codec_audio_sr)
    codec_sr = getattr(cfg, "encodec_sr", codec_sr)
    codec_audio_sr = audio_tokenizer.sample_rate

    if silence_tokens is None:
        silence_tokens = []
    if isinstance(silence_tokens, str):
        silence_tokens = eval(silence_tokens)

    multi_trial = multi_trial or []

    no_reference_audio = str(reference_speech).lower() in {"none", "", "null"} or reference_speech is None
    has_reference_text = not (
        reference_text is None or str(reference_text).strip().lower() in {"", "none", "null"}
    )

    if no_reference_audio and has_reference_text:
        raise ValueError(
            "reference_text was provided but reference_speech is missing. "
            "Please supply a reference_speech or omit reference_text."
        )

    if no_reference_audio:
        prefix_transcript = ""
    elif not has_reference_text:
        print("[Info] Transcribing reference audio...")
        prefix_transcript = transcribe_audio(reference_speech, device)
        print(f"[Info] Whisper transcribed text: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    lang = None if lang in {None, "", "none", "null"} else str(lang)
    target_text, lang_code = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang_code)

    if target_duration is None:
        target_generation_length = estimate_duration(
            target_text=target_text,
            reference_speech=None if no_reference_audio else reference_speech,
            reference_transcript=None if no_reference_audio else prefix_transcript,
            target_lang=lang_code,
            reference_lang=lang_code,
        )
        print(f"[Info] target_duration not provided, estimated as {target_generation_length:.2f} seconds.")
    else:
        target_generation_length = float(target_duration)

    if not no_reference_audio:
        info = get_audio_info(reference_speech)
        prompt_end_frame = int(cut_off_sec * get_sample_rate(info))
    else:
        prompt_end_frame = 0

    decode_config = {
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "temperature": temperature,
        "stop_repetition": stop_repetition,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "silence_tokens": silence_tokens,
        "sample_batch_size": sample_batch_size,
    }

    # ========================================================================
    # Warmup (Optional)
    # ========================================================================
    if warmup_steps > 0:
        print(f"[Info] Running {warmup_steps} warmup step(s) to compile kernels...")
        warmup_start = time.time()
        # Use a short text for warmup
        warmup_text = "Warmup."
        warmup_len = 1.0 # 1 second
        for i in range(warmup_steps):
            try:
                inference_one_sample(
                    model=model,
                    model_args=cfg,
                    text_tokenizer=text_tokenizer,
                    audio_tokenizer=audio_tokenizer,
                    audio_fn=None,
                    target_text=warmup_text,
                    lang=lang_code,
                    device=device,
                    decode_config=decode_config,
                    prompt_end_frame=0,
                    target_generation_length=warmup_len,
                    prefix_transcript="",
                    multi_trial=[],
                    repeat_prompt=0,
                    return_frames=False,
                )
            except Exception as e:
                print(f"[Warn] Warmup step {i+1} failed: {e}")
        print(f"[Info] Warmup complete in {time.time() - warmup_start:.2f}s")

    # ========================================================================
    # Main Inference
    # ========================================================================
    print(f"[Info] Starting inference for: '{target_text}'")
    start_time = time.time()
    
    res = inference_one_sample(
        model=model,
        model_args=cfg,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        audio_fn=None if no_reference_audio else reference_speech,
        target_text=target_text,
        lang=lang_code,
        device=device,
        decode_config=decode_config,
        prompt_end_frame=prompt_end_frame,
        target_generation_length=target_generation_length,
        prefix_transcript=prefix_transcript,
        multi_trial=multi_trial,
        repeat_prompt=repeat_prompt,
        return_frames=dump_tokens,
    )
    
    inference_time = time.time() - start_time
    print(f"[Info] Inference finished in {inference_time:.2f}s")

    if dump_tokens:
        concat_audio, gen_audio, concat_frames, gen_frames = res
    else:
        concat_audio, gen_audio = res

    concat_audio = concat_audio[0].cpu()
    gen_audio = gen_audio[0].cpu()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated_4bit_optimized.wav")
    save_audio(out_path, gen_audio, codec_audio_sr)

    # Calculate stats
    max_abs = torch.max(gen_audio.abs()).item()
    rms = torch.sqrt((gen_audio ** 2).mean()).item()
    print(f"[Info] Generated audio stats -> max_abs: {max_abs:.6f}, rms: {rms:.6f}")

    if dump_tokens:
        np.save(os.path.join(output_dir, "generated_frames.npy"), gen_frames.squeeze(0).cpu().numpy())
        np.save(os.path.join(output_dir, "concat_frames.npy"), concat_frames.squeeze(0).cpu().numpy())
        print(f"[Info] Saved token arrays to {output_dir}")

    print(f"[Success] Generated audio saved to {out_path}")


def main():
    fire.Fire(run_inference_optimized)


if __name__ == "__main__":
    main()
