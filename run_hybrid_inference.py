#!/usr/bin/env python3
"""Run hybrid (ONNX encoder + PyTorch decoder) inference and save audio."""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np

text = "こんにちは、今日はいい天気ですね"
print(f"Text: {text}")

from inference_hybrid_complete import HybridT5GemmaTTS

hybrid_tts = HybridT5GemmaTTS(
    onnx_dir="onnx_models_fp16",
    decoder_weights="weights/decoder_pmrope.bin",
    device="cuda",
)

torch.manual_seed(1)
np.random.seed(1)

sample_rate, audio = hybrid_tts.synthesize(
    text=text,
    language=None,
    target_duration=None,
    top_k=30,
    top_p=0.9,
    temperature=0.7,
    min_p=0.0,
    stop_repetition=3,
)

os.makedirs("outputs/comparison", exist_ok=True)

import soundfile as sf
# Flatten to 1D if multi-dimensional
if audio.ndim > 1:
    audio = audio.flatten()
sf.write("outputs/comparison/audio_hybrid.wav", audio, sample_rate)
print(f"Saved: outputs/comparison/audio_hybrid.wav ({len(audio)/sample_rate:.2f}s)")
