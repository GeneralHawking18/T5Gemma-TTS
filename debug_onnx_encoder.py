#!/usr/bin/env python3
"""
Deep Debug: Compare ONNX Encoder vs PyTorch Encoder step-by-step.

This script identifies exactly where the ONNX encoder diverges from PyTorch.
Key areas to check:
1. Position IDs computation
2. Embedding output 
3. Layer-by-layer comparison (if possible)
4. Attention patterns
"""
import os

# Load .env
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("=" * 70)
print("DEEP DEBUG: ONNX Encoder vs PyTorch Encoder")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================
# 1. Load Models
# ========================================
print("\n[1] Loading 4bit PyTorch model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()
cfg = model.config

tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
print(f"[2] Loading tokenizer from: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

print("\n[3] Loading ONNX Encoder...")
onnx_session = ort.InferenceSession(
    'onnx_models_fp16/encoder.onnx',
    providers=['CPUExecutionProvider']
)

# ========================================
# 2. Prepare Input - Match exactly what inference_tts_utils does
# ========================================
text = "こんにちは"

# Tokenize exactly like 4bit version
add_eos = getattr(cfg, "add_eos_to_text", 1)
add_bos = getattr(cfg, "add_bos_to_text", 0)
tokens = tokenizer.encode(text.strip(), add_special_tokens=False)

print(f"\n[4] Tokenization check:")
print(f"  Text: {text}")
print(f"  Tokens (before EOS/BOS): {tokens}")

if add_eos:
    tokens.append(add_eos)
if add_bos:
    tokens = [add_bos] + tokens

print(f"  Tokens (after adding eos={add_eos}, bos={add_bos}): {tokens}")

input_ids = torch.tensor([tokens], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

print(f"  Input shape: {input_ids.shape}")

# ========================================
# 3. Compute PM-RoPE Position IDs (manually)
# ========================================
progress_scale = getattr(cfg, "progress_scale", 2000.0)
use_pm_rope = getattr(cfg, "use_pm_rope", True)

print(f"\n[5] PM-RoPE Configuration:")
print(f"  use_pm_rope: {use_pm_rope}")
print(f"  progress_scale: {progress_scale}")

x_lens = attention_mask.sum(dim=1)  # [batch]
max_len = input_ids.shape[1]

pos = torch.arange(max_len, dtype=torch.float32)[None, :]  # [1, seq_len]
denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]  # [batch, 1]
position_ids = pos / denom * progress_scale
mask = pos < x_lens[:, None]
position_ids = position_ids.masked_fill(~mask, 0.0)

print(f"\n[6] Position IDs (manual calculation):")
print(f"  Shape: {position_ids.shape}")
print(f"  Values: {position_ids[0].tolist()}")

# ========================================
# 4. Run ONNX Encoder
# ========================================
print("\n[7] Running ONNX Encoder...")
onnx_input_ids = input_ids.numpy().astype(np.int64)
onnx_attention_mask = attention_mask.numpy().astype(np.int64)

onnx_output = onnx_session.run(None, {
    "input_ids": onnx_input_ids,
    "attention_mask": onnx_attention_mask,
})[0]

print(f"  Output shape: {onnx_output.shape}")
print(f"  Output stats: mean={onnx_output.mean():.6f}, std={onnx_output.std():.6f}")
print(f"  First token hidden (first 10 dims): {onnx_output[0, 0, :10]}")
print(f"  Last token hidden (first 10 dims): {onnx_output[0, -1, :10]}")

# ========================================
# 5. Run PyTorch Encoder (Direct call like 4bit)
# ========================================
print("\n[8] Running PyTorch Encoder (with PM-RoPE)...")
with torch.no_grad():
    input_ids_gpu = input_ids.to(model.device)
    attention_mask_gpu = attention_mask.to(model.device)
    position_ids_gpu = position_ids.to(model.device) if use_pm_rope else None
    
    # Get encoder
    if hasattr(model, 'backbone'):
        encoder = model.backbone.model.encoder
    elif hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
    else:
        encoder = model.model.model.encoder
    
    pt_output = encoder(
        input_ids=input_ids_gpu,
        attention_mask=attention_mask_gpu,
        position_ids=position_ids_gpu,
    )
    pt_hidden = pt_output.last_hidden_state.float().cpu().numpy()

print(f"  Output shape: {pt_hidden.shape}")
print(f"  Output stats: mean={pt_hidden.mean():.6f}, std={pt_hidden.std():.6f}")
print(f"  First token hidden (first 10 dims): {pt_hidden[0, 0, :10]}")
print(f"  Last token hidden (first 10 dims): {pt_hidden[0, -1, :10]}")

# ========================================
# 6. Compare Outputs
# ========================================
print("\n[9] Comparing ONNX vs PyTorch:")
diff = np.abs(onnx_output - pt_hidden)
cos_sim = np.dot(onnx_output.flatten(), pt_hidden.flatten()) / (
    np.linalg.norm(onnx_output.flatten()) * np.linalg.norm(pt_hidden.flatten()) + 1e-8
)

print(f"  Max diff: {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")
print(f"  Cosine similarity: {cos_sim:.6f}")

if cos_sim > 0.99:
    print("  ✅ GOOD: Encoders match well!")
elif cos_sim > 0.95:
    print("  ⚠️ WARNING: Some differences exist")
else:
    print("  ❌ CRITICAL: Encoders differ significantly!")

# ========================================
# 7. Test WITHOUT PM-RoPE (position_ids=None)
# ========================================
print("\n[10] Testing PyTorch Encoder WITHOUT PM-RoPE (position_ids=None)...")
with torch.no_grad():
    pt_output_no_rope = encoder(
        input_ids=input_ids_gpu,
        attention_mask=attention_mask_gpu,
        position_ids=None,  # No PM-RoPE
    )
    pt_hidden_no_rope = pt_output_no_rope.last_hidden_state.float().cpu().numpy()

print(f"  Output shape: {pt_hidden_no_rope.shape}")
print(f"  Output stats: mean={pt_hidden_no_rope.mean():.6f}, std={pt_hidden_no_rope.std():.6f}")

diff_no_rope = np.abs(onnx_output - pt_hidden_no_rope)
cos_sim_no_rope = np.dot(onnx_output.flatten(), pt_hidden_no_rope.flatten()) / (
    np.linalg.norm(onnx_output.flatten()) * np.linalg.norm(pt_hidden_no_rope.flatten()) + 1e-8
)

print(f"\n  Comparison ONNX vs PyTorch (NO PM-RoPE):")
print(f"  Max diff: {diff_no_rope.max():.6f}")
print(f"  Mean diff: {diff_no_rope.mean():.6f}")
print(f"  Cosine similarity: {cos_sim_no_rope:.6f}")

if cos_sim_no_rope > cos_sim:
    print("\n  🔍 INSIGHT: ONNX output is CLOSER to PyTorch WITHOUT PM-RoPE!")
    print("     This suggests the ONNX export may not be applying position_ids correctly.")
else:
    print("\n  ONNX output is closer to PyTorch WITH PM-RoPE (expected).")

# ========================================
# 8. Test EncoderWrapper directly (same as ONNX export used)
# ========================================
print("\n[11] Testing EncoderWrapper (same as used during ONNX export)...")
from onnx_modules import EncoderWrapper

# Need to set up model attributes that EncoderWrapper expects
if not hasattr(model, "args"):
    model.args = model.config
if not hasattr(model, "encoder_module"):
    if hasattr(model, "model"):
        model.encoder_module = model.model.encoder
    elif hasattr(model, "backbone"):
        model.encoder_module = model.backbone.model.encoder
if not hasattr(model, "text_input_type"):
    model.text_input_type = getattr(model.config, "text_input_type", "text")
if not hasattr(model, "progress_scale"):
    model.progress_scale = getattr(model.config, "progress_scale", 2000.0)
if not hasattr(model, "text_embedding"):
    # Try to find text embedding
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        model.text_embedding = model.model.encoder.embed_tokens
    elif hasattr(model, "backbone"):
        model.text_embedding = model.backbone.model.encoder.embed_tokens
if not hasattr(model, "text_dropout"):
    model.text_dropout = torch.nn.Identity()

wrapper = EncoderWrapper(model)
wrapper.eval()

with torch.no_grad():
    wrapper_output = wrapper(input_ids_gpu, attention_mask_gpu)
    wrapper_hidden = wrapper_output.float().cpu().numpy()

print(f"  Output shape: {wrapper_hidden.shape}")
print(f"  Output stats: mean={wrapper_hidden.mean():.6f}, std={wrapper_hidden.std():.6f}")

diff_wrapper = np.abs(wrapper_hidden - pt_hidden)
cos_sim_wrapper = np.dot(wrapper_hidden.flatten(), pt_hidden.flatten()) / (
    np.linalg.norm(wrapper_hidden.flatten()) * np.linalg.norm(pt_hidden.flatten()) + 1e-8
)

print(f"\n  Comparison EncoderWrapper vs PyTorch (with PM-RoPE):")
print(f"  Max diff: {diff_wrapper.max():.6f}")
print(f"  Mean diff: {diff_wrapper.mean():.6f}")
print(f"  Cosine similarity: {cos_sim_wrapper:.6f}")

# ========================================
# 9. Summary
# ========================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  ONNX vs PyTorch (with PM-RoPE):    cosine={cos_sim:.6f}")
print(f"  ONNX vs PyTorch (without PM-RoPE): cosine={cos_sim_no_rope:.6f}")
print(f"  Wrapper vs PyTorch (with PM-RoPE): cosine={cos_sim_wrapper:.6f}")

if cos_sim_no_rope > cos_sim + 0.1:
    print("\n❌ DIAGNOSIS: ONNX encoder is NOT applying PM-RoPE position_ids!")
    print("   The ONNX export likely couldn't trace the position_ids correctly.")
    print("   FIX: Re-export the encoder with explicit position_ids as input.")
elif cos_sim_wrapper < 0.95:
    print("\n❌ DIAGNOSIS: EncoderWrapper itself produces different results.")
    print("   Check EncoderWrapper implementation.")
else:
    print("\n⚠️ Need more investigation...")
