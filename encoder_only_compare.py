#!/usr/bin/env python3
"""Minimal encoder comparison - PyTorch vs ONNX."""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModel, AutoTokenizer
import gc

print("=" * 70)
print("ENCODER COMPARISON: PyTorch vs ONNX")
print("=" * 70)

# Load only the T5Gemma backbone (encoder-decoder), not the full TTS model
print("\n[1] Loading T5Gemma backbone encoder only...")
backbone = AutoModel.from_pretrained(
    "google/t5gemma-2b-2b-ul2",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
backbone.eval()
encoder = backbone.encoder

tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2b-2b-ul2")
print("   Loaded successfully")

# Load ONNX
print("\n[2] Loading ONNX encoder...")
onnx_session = ort.InferenceSession(
    'onnx_models_fp16/encoder.onnx',
    providers=['CPUExecutionProvider']
)
print("   Loaded successfully")

# Test cases
test_cases = [
    "こんにちは",
    "Hello world",
    "This is a test sentence",
]

progress_scale = 2000.0

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for text in test_cases:
    print(f"\nText: \"{text}\"")

    # Tokenize (add EOS token = 1)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens.append(1)  # EOS

    input_ids = torch.tensor([tokens], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    print(f"  Tokens: {len(tokens)}")

    # PyTorch encoder with PM-RoPE
    with torch.no_grad():
        input_ids_gpu = input_ids.to(backbone.device)
        attention_mask_gpu = attention_mask.to(backbone.device)

        # PM-RoPE position IDs
        x_lens = attention_mask_gpu.sum(dim=1)
        max_len = input_ids_gpu.shape[1]
        pos = torch.arange(max_len, device=backbone.device, dtype=torch.float32)[None, :]
        denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
        position_ids = pos / denom * progress_scale
        mask = pos < x_lens[:, None]
        position_ids = position_ids.masked_fill(~mask, 0.0)

        pt_out = encoder(
            input_ids=input_ids_gpu,
            attention_mask=attention_mask_gpu,
            position_ids=position_ids,
        ).last_hidden_state.float().cpu().numpy()

    # ONNX encoder
    onnx_out = onnx_session.run(None, {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    })[0]

    # Compare
    cos_sim = np.dot(pt_out.flatten(), onnx_out.flatten()) / (
        np.linalg.norm(pt_out.flatten()) * np.linalg.norm(onnx_out.flatten()) + 1e-8
    )
    max_diff = np.abs(pt_out - onnx_out).max()

    print(f"  PyTorch: mean={pt_out.mean():.4f}, std={pt_out.std():.4f}")
    print(f"  ONNX:    mean={onnx_out.mean():.4f}, std={onnx_out.std():.4f}")
    print(f"  Cosine Similarity: {cos_sim:.6f}")
    print(f"  Max Diff: {max_diff:.4f}")
    print(f"  Status: {'✅ MATCH' if cos_sim > 0.99 else '❌ DIFFER'}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
If cosine similarity < 0.95, the ONNX encoder differs significantly from PyTorch.
This is caused by the SDPA monkeypatch applied during ONNX export.

SOLUTION: Use PyTorch encoder instead of ONNX encoder for correct output.
""")
