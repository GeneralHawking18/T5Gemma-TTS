#!/usr/bin/env python3
"""Debug NaN issue by checking encoder output range and decoder input requirements."""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
import onnxruntime as ort

text = "こんにちは"
print(f"Text: {text}")

# 1. Check ONNX encoder output
print("\n=== ONNX Encoder Output ===")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2b-2b-ul2")

tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
tokens.append(1)  # EOS
input_ids = np.array([tokens], dtype=np.int64)
attention_mask = np.ones_like(input_ids)

sess = ort.InferenceSession('onnx_models_fp16/encoder.onnx', providers=['CPUExecutionProvider'])
onnx_out = sess.run(None, {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
})[0]

print(f"Shape: {onnx_out.shape}")
print(f"Dtype: {onnx_out.dtype}")
print(f"Range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
print(f"Mean: {onnx_out.mean():.6f}, Std: {onnx_out.std():.6f}")
print(f"Has NaN: {np.isnan(onnx_out).any()}")
print(f"Has Inf: {np.isinf(onnx_out).any()}")

# 2. Load decoder and check what it expects
print("\n=== Decoder Weight Check ===")
decoder_weights = torch.load("weights/decoder_pmrope.bin", map_location="cpu")

# Check some decoder weight dtypes
for i, (k, v) in enumerate(decoder_weights.items()):
    if i < 5:
        print(f"  {k}: {v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]")

# Check if any decoder weights are NaN
nan_keys = []
for k, v in decoder_weights.items():
    if torch.isnan(v).any():
        nan_keys.append(k)
if nan_keys:
    print(f"WARNING: {len(nan_keys)} keys have NaN!")
else:
    print("✅ No NaN in decoder weights")

# 3. Test conversion to different dtypes
print("\n=== Dtype Conversion Test ===")
onnx_tensor = torch.from_numpy(onnx_out)

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    converted = onnx_tensor.to(dtype=dtype)
    has_nan = torch.isnan(converted).any().item()
    has_inf = torch.isinf(converted).any().item()
    print(f"  {dtype}: NaN={has_nan}, Inf={has_inf}, range=[{converted.min():.4f}, {converted.max():.4f}]")

print("\n=== Conclusion ===")
print(f"ONNX output dtype: {onnx_out.dtype}")
if onnx_out.dtype == np.float32:
    print("ONNX outputs float32, which is safe for conversion")
elif onnx_out.dtype == np.float16:
    print("⚠️ ONNX outputs float16 - check for overflow issues")
