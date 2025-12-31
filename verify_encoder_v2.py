#!/usr/bin/env python3
"""
Test if corrected monkeypatch produces same output as original encoder.
"""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("=" * 70)
print("Test: Does corrected monkeypatch match original?")
print("=" * 70)

# Load model
print("\n[1] Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)
model.eval()
cfg = model.config
progress_scale = getattr(cfg, "progress_scale", 2000.0)

if hasattr(model, 'backbone'):
    encoder = model.backbone.model.encoder
else:
    encoder = model.model.encoder

# Prepare test input
tokenizer = AutoTokenizer.from_pretrained(
    getattr(cfg, "text_tokenizer_name", None), trust_remote_code=True
)
tokens = tokenizer.encode("こんにちは", add_special_tokens=False) + [1]
input_ids = torch.tensor([tokens], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

x_lens = attention_mask.sum(dim=1)
pos = torch.arange(len(tokens), dtype=torch.float32)[None, :]
position_ids = (pos / (x_lens.float() - 1) * progress_scale).to(torch.float16)

# Get ORIGINAL output
print("\n[2] Getting original output...")
with torch.no_grad():
    original_out = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ).last_hidden_state.float().numpy()

print(f"Original: mean={original_out.mean():.6f}, std={original_out.std():.6f}")
print(f"First 5: {original_out[0, 0, :5]}")

# Apply CORRECTED monkeypatch
print("\n[3] Applying corrected monkeypatch...")
import transformers.masking_utils

def corrected_sdpa_mask(batch_size, cache_position, kv_length, kv_offset=0,
                        mask_function=None, attention_mask=None, **kwargs):
    device = cache_position.device
    q_length = cache_position.shape[0]
    # Full bidirectional attention
    causal_mask = torch.ones((batch_size, 1, q_length, kv_length), dtype=torch.bool, device=device)
    if attention_mask is not None:
        padding_mask = attention_mask.to(torch.bool)[:, None, None, :]
        causal_mask = causal_mask & padding_mask
    return causal_mask

transformers.masking_utils.sdpa_mask = corrected_sdpa_mask
transformers.masking_utils.sdpa_mask_recent_torch = corrected_sdpa_mask
transformers.masking_utils.sdpa_mask_older_torch = corrected_sdpa_mask
if hasattr(transformers.masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS"):
    mapping = transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    if "sdpa" in mapping:
        mapping["sdpa"] = corrected_sdpa_mask

# Get patched output
with torch.no_grad():
    patched_out = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ).last_hidden_state.float().numpy()

print(f"Corrected: mean={patched_out.mean():.6f}, std={patched_out.std():.6f}")

cos_sim = np.dot(original_out.flatten(), patched_out.flatten()) / (
    np.linalg.norm(original_out.flatten()) * np.linalg.norm(patched_out.flatten()) + 1e-8
)
print(f"\nCorrected vs Original cosine sim: {cos_sim:.6f}")

# Apply BROKEN monkeypatch (from original export)
print("\n[4] Testing broken monkeypatch...")

def broken_sdpa_mask(batch_size, cache_position, kv_length, kv_offset=0,
                     mask_function=None, attention_mask=None, **kwargs):
    device = cache_position.device
    q_length = cache_position.shape[0]
    is_causal = kwargs.get('is_causal', True)
    if is_causal:
        query_idx = cache_position.unsqueeze(1)
        key_idx = torch.arange(kv_length, device=device).unsqueeze(0) + kv_offset
        causal_mask = query_idx >= key_idx
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    else:
        causal_mask = torch.ones((batch_size, 1, q_length, kv_length), dtype=torch.bool, device=device)
    if attention_mask is not None:
        padding_mask = attention_mask.to(torch.bool)[:, None, None, :]
        causal_mask = causal_mask & padding_mask
    return causal_mask

transformers.masking_utils.sdpa_mask = broken_sdpa_mask
transformers.masking_utils.sdpa_mask_recent_torch = broken_sdpa_mask
transformers.masking_utils.sdpa_mask_older_torch = broken_sdpa_mask
if hasattr(transformers.masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS"):
    mapping = transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    if "sdpa" in mapping:
        mapping["sdpa"] = broken_sdpa_mask

with torch.no_grad():
    broken_out = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ).last_hidden_state.float().numpy()

print(f"Broken: mean={broken_out.mean():.6f}, std={broken_out.std():.6f}")

cos_sim_broken = np.dot(original_out.flatten(), broken_out.flatten()) / (
    np.linalg.norm(original_out.flatten()) * np.linalg.norm(broken_out.flatten()) + 1e-8
)
print(f"Broken vs Original cosine sim: {cos_sim_broken:.6f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"Corrected monkeypatch: {cos_sim:.4f}")
print(f"Broken monkeypatch:    {cos_sim_broken:.4f}")

if cos_sim > 0.99:
    print("\n✅ CORRECTED monkeypatch matches original!")
    print("   We can export ONNX with this monkeypatch.")
else:
    print("\n❌ Neither monkeypatch matches original.")
    print("   The T5Gemma attention has complex behavior that can't be easily replicated.")
