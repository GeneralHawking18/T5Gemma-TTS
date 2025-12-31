"""Test EncoderWrapper outputs match the raw encoder."""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from onnx_modules import EncoderWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()
cfg = model.config

# Setup model attributes (same as export script)
if not hasattr(model, "args"):
    model.args = model.config
if not hasattr(model, "encoder_module"):
    if hasattr(model, "model"):
        model.encoder_module = model.model.encoder
        model.decoder_module = model.model.decoder
if not hasattr(model, "text_input_type"):
    model.text_input_type = getattr(model.config, "text_input_type", "text")
if not hasattr(model, "progress_scale"):
    model.progress_scale = getattr(model.config, "progress_scale", 2000.0)

# Create wrapper (same as export)
wrapper = EncoderWrapper(model)
wrapper.eval()

tokenizer_name = getattr(cfg, "text_tokenizer_name", None)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

print(f"\ntext_input_type: {model.text_input_type}")
print(f"progress_scale: {model.progress_scale}")

# Test text
text = "こんにちは"
add_eos = getattr(cfg, "add_eos_to_text", 1)
tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
if add_eos:
    tokens.append(add_eos)

input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
attention_mask = torch.ones_like(input_ids)

print(f"\nText: {text}")
print(f"Tokens: {tokens}")

with torch.no_grad():
    # Method 1: EncoderWrapper (what ONNX exports)
    wrapper_out = wrapper(input_ids, attention_mask)

    # Method 2: Direct encoder call with manual position IDs (what I was comparing against)
    progress_scale = model.progress_scale
    x_lens = attention_mask.sum(dim=1)
    max_len = input_ids.shape[1]
    pos = torch.arange(max_len, device=model.device, dtype=torch.float32)[None, :]
    denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
    position_ids = pos / denom * progress_scale
    mask = pos < x_lens[:, None]
    position_ids = position_ids.masked_fill(~mask, 0.0)

    direct_out = model.encoder_module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ).last_hidden_state

print(f"\nWrapper output shape: {wrapper_out.shape}")
print(f"Direct output shape: {direct_out.shape}")

wrapper_np = wrapper_out.float().cpu().numpy()
direct_np = direct_out.float().cpu().numpy()

cos_sim = np.dot(wrapper_np.flatten(), direct_np.flatten()) / (
    np.linalg.norm(wrapper_np.flatten()) * np.linalg.norm(direct_np.flatten()) + 1e-8
)
diff = np.abs(wrapper_np - direct_np)

print(f"\nWrapper: mean={wrapper_np.mean():.6f}, std={wrapper_np.std():.6f}")
print(f"Direct:  mean={direct_np.mean():.6f}, std={direct_np.std():.6f}")
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
print(f"Cosine similarity: {cos_sim:.6f}")

if cos_sim > 0.9999:
    print("\n✅ EncoderWrapper matches direct encoder call!")
else:
    print("\n❌ EncoderWrapper differs from direct encoder call!")
    print("   The ONNX export might have an issue with the wrapper.")
