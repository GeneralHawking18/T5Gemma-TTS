#!/usr/bin/env python3
"""Run 4-bit inference and save audio."""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

text = "こんにちは、今日はいい天気ですね"
print(f"Text: {text}")

model = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()
cfg = model.config

tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

from data.tokenizer import AudioTokenizer
from inference_tts_utils import inference_one_sample, normalize_text_with_lang
from duration_estimator import estimate_duration

audio_tokenizer = AudioTokenizer(
    backend="xcodec2",
    model_name=getattr(cfg, "xcodec2_model_name", "xcodec2"),
)
codec_audio_sr = audio_tokenizer.sample_rate
codec_sr = int(getattr(cfg, "encodec_sr", 50))

normalized_text, lang_code = normalize_text_with_lang(text, None)
target_duration = estimate_duration(
    target_text=normalized_text,
    reference_speech=None,
    reference_transcript=None,
    target_lang=lang_code,
    reference_lang=lang_code,
)
print(f"Duration: {target_duration:.2f}s")

torch.manual_seed(1)
np.random.seed(1)

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

_, gen_audio = inference_one_sample(
    model=model,
    model_args=cfg,
    text_tokenizer=text_tokenizer,
    audio_tokenizer=audio_tokenizer,
    audio_fn=None,
    target_text=normalized_text,
    lang=lang_code,
    device="cuda",
    decode_config=decode_config,
    prompt_end_frame=0,
    target_generation_length=target_duration,
    prefix_transcript="",
    multi_trial=[],
    repeat_prompt=0,
    return_frames=False,
)

gen_audio = gen_audio[0].cpu()
# Flatten to 1D if multi-dimensional
audio_np = gen_audio.numpy()
if audio_np.ndim > 1:
    audio_np = audio_np.flatten()
os.makedirs("outputs/comparison", exist_ok=True)

import soundfile as sf
sf.write("outputs/comparison/audio_4bit.wav", audio_np, codec_audio_sr)
print(f"Saved: outputs/comparison/audio_4bit.wav ({len(audio_np)/codec_audio_sr:.2f}s)")

