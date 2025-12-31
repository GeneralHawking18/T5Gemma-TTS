#!/usr/bin/env python3
"""
So sánh kết quả inference cuối cùng giữa Hybrid và 4-bit.

QUAN TRỌNG: ONNX encoder từ full model, 4-bit encoder từ quantized model
=> Encoder outputs sẽ khác nhau, nhưng FINAL output mới quan trọng!

Script này so sánh:
1. Generated audio từ cả 2 pipeline
2. Nghe để so sánh chất lượng
"""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("=" * 70)
print("SO SÁNH FULL INFERENCE: Hybrid vs 4-bit")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Test input
text = "こんにちは、今日はいい天気ですね"
print(f"\nTest text: {text}")

# ========================================
# 1. Load 4-bit model
# ========================================
print("\n[1] Loading 4-bit model...")
start = time.time()
model_4bit = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model_4bit.eval()
cfg = model_4bit.config
print(f"    Loaded in {time.time()-start:.1f}s")

tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

# ========================================
# 2. Run 4-bit Inference (using inference_one_sample like inference_4bit_gpu.py)
# ========================================
print("\n[2] Running 4-bit inference...")
from data.tokenizer import AudioTokenizer
from inference_tts_utils import inference_one_sample, normalize_text_with_lang, save_audio
from duration_estimator import estimate_duration

# Load audio tokenizer
audio_tokenizer = AudioTokenizer(
    backend="xcodec2",
    model_name=getattr(cfg, "xcodec2_model_name", "xcodec2"),
)
codec_audio_sr = audio_tokenizer.sample_rate
codec_sr = int(getattr(cfg, "encodec_sr", 50))

normalized_text, lang_code = normalize_text_with_lang(text, None)
print(f"    Normalized: '{normalized_text}' (lang: {lang_code})")

target_duration = estimate_duration(
    target_text=normalized_text,
    reference_speech=None,
    reference_transcript=None,
    target_lang=lang_code,
    reference_lang=lang_code,
)
print(f"    Estimated duration: {target_duration:.2f}s")

# Set seed for reproducibility
def seed_everything(seed=1):
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(1)

decode_config = {
    "top_k": 30,
    "top_p": 0.9,
    "min_p": 0.0,
    "temperature": 0.7,
    "stop_repetition": 3,
    "codec_audio_sr": codec_audio_sr,
    "codec_sr": codec_sr,
    "silence_tokens": [],
    "sample_batch_size": 1,
}

start = time.time()
concat_audio_4bit, gen_audio_4bit = inference_one_sample(
    model=model_4bit,
    model_args=cfg,
    text_tokenizer=text_tokenizer,
    audio_tokenizer=audio_tokenizer,
    audio_fn=None,  # No reference audio
    target_text=normalized_text,
    lang=lang_code,
    device=device,
    decode_config=decode_config,
    prompt_end_frame=0,
    target_generation_length=target_duration,
    prefix_transcript="",
    multi_trial=[],
    repeat_prompt=0,
    return_frames=False,
)
time_4bit = time.time() - start

gen_audio_4bit = gen_audio_4bit[0].cpu()
print(f"    Generated audio in {time_4bit:.2f}s")
print(f"    Audio length: {gen_audio_4bit.shape[-1]} samples = {gen_audio_4bit.shape[-1]/codec_audio_sr:.2f}s")

# Free GPU memory before loading hybrid
del model_4bit
torch.cuda.empty_cache()

# ========================================
# 3. Load Hybrid Model (ONNX encoder + PyTorch decoder)
# ========================================
print("\n[3] Loading Hybrid model...")
from inference_hybrid_complete import HybridT5GemmaTTS

start = time.time()
hybrid_tts = HybridT5GemmaTTS(
    onnx_dir="onnx_models_fp16",
    decoder_weights="weights/decoder_pmrope.bin",
    device=device,
)
print(f"    Loaded in {time.time()-start:.1f}s")

# ========================================
# 4. Run Hybrid Inference
# ========================================
print("\n[4] Running Hybrid inference...")

# Set same seed
torch.manual_seed(1)
np.random.seed(1)

start = time.time()
sample_rate, audio_hybrid = hybrid_tts.synthesize(
    text=text,
    language=None,
    target_duration=None,  # auto-estimate
    top_k=30,
    top_p=0.9,
    temperature=0.7,
    min_p=0.0,
    stop_repetition=3,
)
time_hybrid = time.time() - start

print(f"    Generated audio in {time_hybrid:.2f}s")
print(f"    Sample rate: {sample_rate}Hz")
print(f"    Audio length: {len(audio_hybrid)} samples = {len(audio_hybrid)/sample_rate:.2f}s")

# ========================================
# 5. Save outputs for manual comparison
# ========================================
print("\n[5] Saving outputs...")

os.makedirs("outputs/comparison", exist_ok=True)

# Save 4-bit audio
import scipy.io.wavfile as wav
wav.write("outputs/comparison/audio_4bit.wav", codec_audio_sr, gen_audio_4bit.numpy().astype(np.float32))
print("    Saved: outputs/comparison/audio_4bit.wav")

# Save hybrid audio
wav.write("outputs/comparison/audio_hybrid.wav", sample_rate, audio_hybrid.astype(np.float32))
print("    Saved: outputs/comparison/audio_hybrid.wav")

# ========================================
# 6. Summary
# ========================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  4-bit inference time:  {time_4bit:.2f}s")
print(f"  Hybrid inference time: {time_hybrid:.2f}s")
print(f"  4-bit audio: {gen_audio_4bit.shape[-1]/codec_audio_sr:.2f}s")
print(f"  Hybrid audio: {len(audio_hybrid)/sample_rate:.2f}s")

print("\n⚠️ LƯU Ý: ONNX encoder từ FULL model (FP16), 4-bit từ quantized model.")  
print("   Encoder outputs sẽ KHÁC NHAU, nhưng cả hai đều có thể sinh audio đúng!")
print("   Hãy nghe cả 2 file audio để so sánh chất lượng:")
print("   - outputs/comparison/audio_4bit.wav")
print("   - outputs/comparison/audio_hybrid.wav")

