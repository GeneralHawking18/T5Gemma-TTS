#!/usr/bin/env python3
"""
T5Gemma-TTS CPU-Optimized Inference Script.

Tối ưu cho:
- Tốc độ inference nhanh nhất trên CPU
- RAM usage thấp nhất

Kỹ thuật sử dụng:
- INT8 Dynamic Quantization (~50% RAM, ~2x speed)
- torch.compile() optional (~1.5-2x speed, warm-up 30s)
- Thread tuning
- Low memory loading

Usage:
    python inference_pytorch_cpu.py \\
        --target_text "Xin chào, đây là test"
"""

import os
import gc
import random
import time


# ============================================================
# CPU OPTIMIZATIONS - MUST BE SET BEFORE IMPORTING TORCH
# ============================================================
_DEFAULT_THREADS = 4

os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_DEFAULT_THREADS))

import torch
import numpy as np

# Set threads
torch.set_num_threads(_DEFAULT_THREADS)
torch.set_num_interop_threads(1)  # Reduce inter-op overhead

# Disable gradients globally
torch.set_grad_enabled(False)

import fire
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# IMPORTS
# ============================================================
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
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def seed_everything(seed: int = 1):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_cpu_info() -> dict:
    """Get CPU optimization info."""
    return {
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "mkl_available": torch.backends.mkl.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "openmp_available": torch.backends.openmp.is_available(),
    }


def print_memory_usage(label: str = ""):
    """Print current memory usage."""
    import psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[Memory] {label}: {mem_mb:.1f} MB")


# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def run_inference_cpu(
    # Text input
    target_text: str = "Hello, this is a test of text to speech.",
    
    # Model config - HuggingFace Hub
    model_dir: str = "Aratako/T5Gemma-TTS-2b-2b",
    
    # CPU Optimizations
    num_threads: int = None,          # Auto-detect if None
    use_quantization: bool = True,    # INT8 dynamic quantization
    use_compile: bool = False,        # torch.compile (warm-up ~30s)
    
    # Reference audio (optional)
    reference_speech: str = None,
    reference_text: str = None,
    target_duration: float = None,
    
    # Decoding parameters
    top_k: int = 30,
    top_p: float = 0.9,
    min_p: float = 0,
    temperature: float = 0.8,
    stop_repetition: int = 3,
    repeat_prompt: int = 0,
    
    # Output
    output_dir: str = "./generated_tts_cpu",
    seed: int = 1,
    lang: str = None,
    
    # Debug
    verbose: bool = True,
    dump_tokens: bool = False,
):
    """
    CPU-optimized TTS inference using HuggingFace model.
    
    Args:
        target_text: Text to synthesize
        model_dir: HuggingFace model ID or local path
        num_threads: Number of CPU threads (auto-detect if None)
        use_quantization: Apply INT8 dynamic quantization (50% RAM, 2x speed)
        use_compile: Use torch.compile (warm-up 30s, then 1.5-2x faster)
        reference_speech: Path to reference audio for voice cloning
        reference_text: Transcript of reference audio (auto-transcribe if None)
        
    Example:
        python inference_pytorch_cpu.py \\
            --target_text "Xin chào thế giới"
    """
    total_start = time.time()
    
    # ============================================================
    # 1. CONFIGURE THREADS
    # ============================================================
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
    
    if verbose:
        info = get_cpu_info()
        print(f"[CPU] Threads: {info['num_threads']}, MKL: {info['mkl_available']}, "
              f"MKLDNN: {info['mkldnn_available']}")
    
    seed_everything(seed)
    device = "cpu"
    
    # ============================================================
    # 2. LOAD MODEL FROM HUGGINGFACE
    # ============================================================
    if verbose:
        print_memory_usage("Before loading")
    
    try:
        from transformers import AutoModelForSeq2SeqLM
    except ImportError:
        raise ImportError("transformers required. Run: pip install transformers")
    
    if verbose:
        print(f"[Load] Loading model from {model_dir}...")
    
    load_start = time.time()
    
    # Load with CPU-optimized settings
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # float32 is most stable on CPU
        device_map=None,            # No auto device map for CPU
        low_cpu_mem_usage=True,     # Use less memory during loading
    )
    model = model.to(device)
    model.eval()
    
    cfg = model.config
    
    if verbose:
        print(f"[Load] Model loaded in {time.time() - load_start:.2f}s")
        print_memory_usage("After model init")
    
    # Cleanup
    gc.collect()
    
    # ============================================================
    # 3. APPLY INT8 QUANTIZATION (saves ~50% RAM, ~2x faster)
    # ============================================================
    if use_quantization:
        if verbose:
            print("[Optim] Applying INT8 dynamic quantization...")
        
        quant_start = time.time()
        try:
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            if verbose:
                print(f"[Optim] Quantization done in {time.time() - quant_start:.2f}s")
                print_memory_usage("After quantization")
        except Exception as e:
            print(f"[Warning] Quantization failed: {e}")
    
    # ============================================================
    # 4. OPTIONAL: torch.compile (warm-up ~30s, then faster)
    # ============================================================
    if use_compile:
        if hasattr(torch, "compile"):
            if verbose:
                print("[Optim] Compiling model (this may take ~30 seconds)...")
            compile_start = time.time()
            try:
                model = torch.compile(
                    model,
                    backend="inductor",
                    mode="max-autotune",
                    fullgraph=False,
                )
                if verbose:
                    print(f"[Optim] Compile setup done in {time.time() - compile_start:.2f}s")
            except Exception as e:
                print(f"[Warning] torch.compile failed: {e}")
        else:
            print("[Warning] torch.compile not available (requires PyTorch 2.0+)")
    
    # ============================================================
    # 5. LOAD TOKENIZERS
    # ============================================================
    if AutoTokenizer is None:
        raise ImportError("transformers required. Run: pip install transformers")
    
    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or \
                     getattr(cfg, "t5gemma_model_name", "google/t5gemma-b-b-ul2")
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if verbose:
        print("[Load] Loading audio tokenizer (XCodec2)...")
    
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(cfg, "xcodec2_model_name", None),
        device=device,
    )
    
    codec_audio_sr = audio_tokenizer.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)
    
    if verbose:
        print_memory_usage("After tokenizers")
    
    # ============================================================
    # 6. PREPARE INPUT
    # ============================================================
    no_reference = reference_speech is None or str(reference_speech).lower() in {"none", "", "null"}
    has_ref_text = reference_text is not None and str(reference_text).strip().lower() not in {"", "none", "null"}
    
    if no_reference and has_ref_text:
        raise ValueError("reference_text provided but reference_speech is missing")
    
    if no_reference:
        prefix_transcript = ""
    elif not has_ref_text:
        if verbose:
            print("[Whisper] Transcribing reference audio (may take a while on CPU)...")
        prefix_transcript = transcribe_audio(reference_speech, device)
        if verbose:
            print(f"[Whisper] Transcribed: {prefix_transcript}")
    else:
        prefix_transcript = reference_text
    
    # Normalize text
    lang = None if lang in {None, "", "none", "null"} else str(lang)
    target_text, lang_code = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang_code)
    
    # Estimate duration
    if target_duration is None:
        target_generation_length = estimate_duration(
            target_text=target_text,
            reference_speech=None if no_reference else reference_speech,
            reference_transcript=None if no_reference else prefix_transcript,
            target_lang=lang_code,
            reference_lang=lang_code,
        )
        if verbose:
            print(f"[Duration] Estimated: {target_generation_length:.2f}s")
    else:
        target_generation_length = float(target_duration)
    
    # Get prompt end frame
    if not no_reference:
        info = get_audio_info(reference_speech)
        prompt_end_frame = int(100 * get_sample_rate(info))  # cut_off_sec=100
    else:
        prompt_end_frame = 0
    
    # ============================================================
    # 7. RUN INFERENCE
    # ============================================================
    decode_config = {
        'top_k': top_k,
        'top_p': top_p,
        'min_p': min_p,
        'temperature': temperature,
        'stop_repetition': stop_repetition,
        'codec_audio_sr': codec_audio_sr,
        'codec_sr': codec_sr,
        'silence_tokens': [],
        'sample_batch_size': 1,
    }
    
    if verbose:
        print("[Inference] Running TTS inference...")
    
    infer_start = time.time()
    
    with torch.inference_mode():
        result = inference_one_sample(
            model=model,
            model_args=cfg,
            text_tokenizer=text_tokenizer,
            audio_tokenizer=audio_tokenizer,
            audio_fn=None if no_reference else reference_speech,
            target_text=target_text,
            lang=lang_code,
            device=device,
            decode_config=decode_config,
            prompt_end_frame=prompt_end_frame,
            target_generation_length=target_generation_length,
            prefix_transcript=prefix_transcript,
            multi_trial=[],
            repeat_prompt=repeat_prompt,
            return_frames=dump_tokens,
        )
    
    infer_time = time.time() - infer_start
    
    if dump_tokens:
        concat_audio, gen_audio, concat_frames, gen_frames = result
    else:
        concat_audio, gen_audio = result
    
    concat_audio, gen_audio = concat_audio[0].cpu(), gen_audio[0].cpu()
    
    # ============================================================
    # 8. SAVE OUTPUT
    # ============================================================
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated.wav")
    save_audio(out_path, gen_audio, codec_audio_sr)
    
    if dump_tokens:
        np.save(os.path.join(output_dir, "gen_frames.npy"), gen_frames.squeeze(0).cpu().numpy())
    
    # ============================================================
    # 9. STATS
    # ============================================================
    audio_duration = gen_audio.shape[-1] / codec_audio_sr
    rtf = infer_time / audio_duration if audio_duration > 0 else float('inf')
    total_time = time.time() - total_start
    
    print("=" * 60)
    print(f"[Success] Audio saved to: {out_path}")
    print(f"[Stats] Audio duration: {audio_duration:.2f}s")
    print(f"[Stats] Inference time: {infer_time:.2f}s")
    print(f"[Stats] RTF: {rtf:.2f}x (speed: {1/rtf:.2f}x real-time)")
    print(f"[Stats] Total time: {total_time:.2f}s")
    print_memory_usage("Final")
    print("=" * 60)
    
    # Cleanup
    gc.collect()
    
    return out_path


def main():
    fire.Fire(run_inference_cpu)


if __name__ == "__main__":
    main()
