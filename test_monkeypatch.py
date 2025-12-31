"""Test if SDPA monkeypatch affects encoder output."""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# First, get output WITHOUT monkeypatch
print("Loading model WITHOUT monkeypatch...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()
cfg = model.config

tokenizer = AutoTokenizer.from_pretrained(
    getattr(cfg, "text_tokenizer_name", None),
    trust_remote_code=True
)

text = "こんにちは"
tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
tokens.append(1)  # EOS

input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
attention_mask = torch.ones_like(input_ids)

# PM-RoPE position IDs
progress_scale = getattr(cfg, "progress_scale", 2000.0)
x_lens = attention_mask.sum(dim=1)
max_len = input_ids.shape[1]
pos = torch.arange(max_len, device=model.device, dtype=torch.float32)[None, :]
denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
position_ids = pos / denom * progress_scale
mask = pos < x_lens[:, None]
position_ids = position_ids.masked_fill(~mask, 0.0)

with torch.no_grad():
    # Find the encoder
    if hasattr(model, 'backbone'):
        encoder = model.backbone.model.encoder
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
    else:
        encoder = model.encoder_module

    out_before = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ).last_hidden_state.float().cpu().numpy()

print(f"WITHOUT monkeypatch: mean={out_before.mean():.6f}, std={out_before.std():.6f}")
print(f"  First 5: {out_before[0, 0, :5]}")

# Now apply the monkeypatch (same as export script)
print("\nApplying SDPA monkeypatch...")
import transformers.masking_utils

def _custom_no_vmap_sdpa_mask(
    batch_size,
    cache_position,
    kv_length,
    kv_offset=0,
    mask_function=None,
    attention_mask=None,
    **kwargs
):
    device = cache_position.device
    q_length = cache_position.shape[0]
    is_causal = kwargs.get('is_causal', True)

    if is_causal:
        query_idx = cache_position.unsqueeze(1)
        key_idx = torch.arange(kv_length, device=device).unsqueeze(0) + kv_offset
        causal_mask = query_idx >= key_idx
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    else:
        causal_mask = torch.ones(
            (batch_size, 1, q_length, kv_length),
            dtype=torch.bool,
            device=device
        )

    if attention_mask is not None:
        padding_mask = attention_mask.to(torch.bool)[:, None, None, :]
        causal_mask = causal_mask & padding_mask

    return causal_mask

transformers.masking_utils.sdpa_mask = _custom_no_vmap_sdpa_mask
transformers.masking_utils.sdpa_mask_recent_torch = _custom_no_vmap_sdpa_mask
transformers.masking_utils.sdpa_mask_older_torch = _custom_no_vmap_sdpa_mask

if hasattr(transformers.masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS"):
    mapping = transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    if "sdpa" in mapping:
        mapping["sdpa"] = _custom_no_vmap_sdpa_mask

print("Monkeypatch applied.")

# Get output WITH monkeypatch
with torch.no_grad():
    out_after = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ).last_hidden_state.float().cpu().numpy()

print(f"WITH monkeypatch:    mean={out_after.mean():.6f}, std={out_after.std():.6f}")
print(f"  First 5: {out_after[0, 0, :5]}")

# Compare
cos_sim = np.dot(out_before.flatten(), out_after.flatten()) / (
    np.linalg.norm(out_before.flatten()) * np.linalg.norm(out_after.flatten()) + 1e-8
)
diff = np.abs(out_before - out_after)
print(f"\nBefore vs After monkeypatch:")
print(f"  Max diff: {diff.max():.6f}")
print(f"  Cosine similarity: {cos_sim:.6f}")

if cos_sim > 0.9999:
    print("  ✅ Monkeypatch has no effect on encoder output")
else:
    print("  ❌ Monkeypatch CHANGES encoder output!")
