"""TTS inference script optimized for CPU.

Based on inference_commandline_hf.py but with CPU-specific optimizations:
- Forces CPU device
- Uses float32 precision (most stable on CPU)
- Optimized thread count
- Optional torch.compile for PyTorch 2.0+
"""

import os
import random
import time

import fire
import numpy as np
import torch

# CPU Configuration - set before other imports
# Limit threads to avoid thrashing on high-core systems if memory is tight
torch.set_num_threads(4) 
torch.set_num_interop_threads(1)

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
from dotenv import load_dotenv

load_dotenv()


try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


def seed_everything(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_inference_cpu(
    reference_speech=None,
    target_text="Hello, this is a test of the text to speech system on CPU.",
    # Model from HuggingFace Hub
    model_dir="Aratako/T5Gemma-TTS-2b-2b",
    # CPU specific options
    num_threads=None,  # Defaults to os.cpu_count()
    use_compile=False,  # Try torch.compile (experimental on CPU)
    # Additional optional
    reference_text=None,
    target_duration=None,
    # Decoding parameters
    codec_audio_sr=16000,
    codec_sr=50,
    top_k=30,
    top_p=0.9,
    min_p=0,
    temperature=0.8,
    silence_tokens=None,
    multi_trial=None,
    repeat_prompt=0,
    stop_repetition=3,
    sample_batch_size=1,
    seed=1,
    output_dir="./generated_tts_cpu",
    cut_off_sec=100,
    dump_tokens=False,
    lang=None,
):
    """
    CPU-Optimized TTS Inference using HuggingFace model.
    
    Model will be automatically downloaded from HuggingFace Hub.
    
    Example:
        uv run python inference_cpu.py \\
            --target_text "Hello world, this is a test."
    """
    # CPU thread configuration
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    
    print(f"[Info] Running on CPU with {torch.get_num_threads()} threads")
    
    seed_everything(seed)

    # Force CPU device
    device = "cpu"

    if AutoModelForSeq2SeqLM is None:
        raise ImportError("transformers is not installed. Run: pip install transformers")

    print(f"[Info] Loading model from {model_dir}...")
    start_load = time.time()
    
    # Load model with CPU-optimized settings
    # Use float32 for best CPU compatibility (bfloat16 can be slow on some CPUs)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.float32,  # float32 is most stable on CPU
        device_map=None,  # No auto device map for CPU
        low_cpu_mem_usage=True,  # Use less memory during loading
    )
    model = model.to(device)
    model.eval()
    
    print(f"[Info] Model loaded in {time.time() - start_load:.2f}s")
    
    # Dynamic Quantization for CPU (float32 -> int8)
    # Reduces model size by ~50% and speeds up CPU inference
    if use_compile and hasattr(torch, "compile"):
        # Compile mode
        try:
             print("[Info] Attempting torch.compile()...")
             model = torch.compile(model, backend="inductor", mode="reduce-overhead")
             print("[Info] Model compiled successfully.")
        except Exception as e:
             print(f"[Warning] torch.compile failed: {e}")
    elif True: # Default to dynamic quantization if not compiling
        print("[Info] Applying Dynamic Quantization (int8) for CPU speedup...")
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("[Info] Model quantized successfully.")
        except Exception as e:
            print(f"[Warning] Quantization failed: {e}")

    cfg = model.config

    if AutoTokenizer is None:
        raise ImportError("transformers is not installed")
    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Audio tokenizer - also force CPU
    print("[Info] Loading audio tokenizer (XCodec2)...")
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(cfg, "xcodec2_model_name", None),
        device=device,  # Force CPU
    )
    
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
        print("[Info] Transcribing reference speech with Whisper (this may take a while on CPU)...")
        prefix_transcript = transcribe_audio(reference_speech, device)
        print(f"[Info] Transcribed text: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    # Language + normalization
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
        print(f"[Info] Estimated duration: {target_generation_length:.2f} seconds")
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

    print("[Info] Running TTS inference...")
    start_infer = time.time()

    with torch.inference_mode():
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

    infer_duration = time.time() - start_infer

    if dump_tokens:
        concat_audio, gen_audio, concat_frames, gen_frames = res
    else:
        concat_audio, gen_audio = res

    concat_audio, gen_audio = concat_audio[0].cpu(), gen_audio[0].cpu()
    
    # Calculate RTF (Real Time Factor)
    generated_seconds = gen_audio.shape[-1] / codec_audio_sr
    rtf = infer_duration / generated_seconds if generated_seconds > 0 else float('inf')
    
    print(f"[Success] Inference completed in {infer_duration:.2f}s")
    print(f"[Info] Generated {generated_seconds:.2f}s of audio")
    print(f"[Info] RTF: {rtf:.2f}x (speed: {1/rtf:.2f}x real-time)")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated_cpu.wav")
    save_audio(out_path, gen_audio, codec_audio_sr)

    max_abs = torch.max(gen_audio.abs()).item()
    rms = torch.sqrt((gen_audio ** 2).mean()).item()
    print(f"[Info] Audio stats -> max_abs: {max_abs:.6f}, rms: {rms:.6f}")

    if dump_tokens:
        np.save(os.path.join(output_dir, "generated_frames.npy"), gen_frames.squeeze(0).cpu().numpy())
        np.save(os.path.join(output_dir, "concat_frames.npy"), concat_frames.squeeze(0).cpu().numpy())
        print(f"[Info] Saved token arrays to {output_dir}")

    print(f"[Success] Audio saved to {out_path}")


def main():
    fire.Fire(run_inference_cpu)


if __name__ == "__main__":
    main()
