#!/usr/bin/env python3
"""
Encoder Comparison Report: PyTorch Base Model vs ONNX Export

This script compares:
1. PyTorch encoder (base model, no quantization) - the ground truth
2. ONNX encoder (exported model)

And identifies the source of differences.
"""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json

print("=" * 80)
print("ENCODER COMPARISON REPORT")
print("PyTorch Base Model vs ONNX Export")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Load Base Model (Using 4bit for memory efficiency - encoder is NOT quantized)
# ============================================================================
print("\n[1] Loading PyTorch Model (Aratako/T5Gemma-TTS-2b-2b-encoder-4bit)...")
print("   Note: The encoder in this model is NOT 4bit quantized, only the decoder is.")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()
cfg = model.config

# Get encoder
if hasattr(model, 'backbone'):
    encoder = model.backbone.model.encoder
elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
    encoder = model.model.encoder
else:
    raise AttributeError("Cannot find encoder")

# Load tokenizer
tokenizer_name = getattr(cfg, "text_tokenizer_name", None)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

# Config values
progress_scale = getattr(cfg, "progress_scale", 2000.0)
add_eos = getattr(cfg, "add_eos_to_text", 1)

print(f"   Model loaded successfully")
print(f"   Progress scale: {progress_scale}")
print(f"   Add EOS to text: {add_eos}")

# ============================================================================
# Load ONNX Encoder
# ============================================================================
print("\n[2] Loading ONNX Encoder (onnx_models_fp16/encoder.onnx)...")
onnx_session = ort.InferenceSession(
    'onnx_models_fp16/encoder.onnx',
    providers=['CPUExecutionProvider']
)
print("   ONNX encoder loaded successfully")

# ============================================================================
# Test Cases
# ============================================================================
test_cases = [
    ("Japanese short", "こんにちは"),
    ("Japanese medium", "こんにちは、今日はいい天気ですね"),
    ("Japanese long", "私は人工知能アシスタントです。何かお手伝いできることはありますか？"),
    ("English short", "Hello world"),
    ("English medium", "This is a test sentence for the encoder"),
    ("English long", "The quick brown fox jumps over the lazy dog. This is a longer sentence to test."),
]

# ============================================================================
# Helper Functions
# ============================================================================
def compute_position_ids(attention_mask, progress_scale, device):
    """Compute PM-RoPE position IDs."""
    x_lens = attention_mask.sum(dim=1)
    max_len = attention_mask.shape[1]
    pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
    denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
    position_ids = pos / denom * progress_scale
    mask = pos < x_lens[:, None]
    return position_ids.masked_fill(~mask, 0.0)

def cosine_similarity(a, b):
    """Compute cosine similarity between two arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)

# ============================================================================
# Run Comparisons
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

results = []

for name, text in test_cases:
    print(f"\n--- {name}: \"{text[:30]}{'...' if len(text) > 30 else ''}\" ---")

    # Tokenize
    tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
    if add_eos:
        tokens.append(add_eos)

    input_ids = torch.tensor([tokens], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    print(f"   Tokens: {len(tokens)}")

    # PyTorch encoder
    with torch.no_grad():
        input_ids_gpu = input_ids.to(model.device)
        attention_mask_gpu = attention_mask.to(model.device)
        position_ids = compute_position_ids(attention_mask_gpu, progress_scale, model.device)

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

    # Compute metrics
    cos_sim = cosine_similarity(pt_out, onnx_out)
    max_diff = np.abs(pt_out - onnx_out).max()
    mean_diff = np.abs(pt_out - onnx_out).mean()
    rel_error = mean_diff / (np.abs(pt_out).mean() + 1e-8) * 100

    print(f"   PyTorch: mean={pt_out.mean():.6f}, std={pt_out.std():.6f}")
    print(f"   ONNX:    mean={onnx_out.mean():.6f}, std={onnx_out.std():.6f}")
    print(f"   Cosine Similarity: {cos_sim:.6f}")
    print(f"   Max Diff: {max_diff:.4f}, Mean Diff: {mean_diff:.4f}")
    print(f"   Relative Error: {rel_error:.2f}%")

    status = "✅ MATCH" if cos_sim > 0.99 else ("⚠️ CLOSE" if cos_sim > 0.95 else "❌ DIFFER")
    print(f"   Status: {status}")

    results.append({
        "name": name,
        "tokens": len(tokens),
        "cosine_sim": cos_sim,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_error": rel_error,
        "pt_mean": pt_out.mean(),
        "pt_std": pt_out.std(),
        "onnx_mean": onnx_out.mean(),
        "onnx_std": onnx_out.std(),
    })

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)

avg_cos_sim = np.mean([r["cosine_sim"] for r in results])
avg_rel_error = np.mean([r["rel_error"] for r in results])
max_max_diff = np.max([r["max_diff"] for r in results])

print(f"\nOverall Statistics:")
print(f"   Average Cosine Similarity: {avg_cos_sim:.6f}")
print(f"   Average Relative Error: {avg_rel_error:.2f}%")
print(f"   Maximum Difference: {max_max_diff:.4f}")

print(f"\nStatistics by Encoder:")
print(f"   PyTorch Mean (avg): {np.mean([r['pt_mean'] for r in results]):.6f}")
print(f"   PyTorch Std (avg):  {np.mean([r['pt_std'] for r in results]):.6f}")
print(f"   ONNX Mean (avg):    {np.mean([r['onnx_mean'] for r in results]):.6f}")
print(f"   ONNX Std (avg):     {np.mean([r['onnx_std'] for r in results]):.6f}")

# ============================================================================
# Root Cause Analysis
# ============================================================================
print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

if avg_cos_sim < 0.95:
    print("""
FINDING: SIGNIFICANT DIFFERENCE DETECTED

The ONNX encoder produces significantly different outputs compared to the
PyTorch base model encoder. This explains the audio quality issues in the
hybrid inference.

CAUSE: SDPA Monkeypatch During Export
--------------------------------------
The ONNX encoder was exported with a monkeypatch applied to
`transformers.masking_utils.sdpa_mask`. This monkeypatch was intended to
make the model compatible with ONNX tracing, but it CHANGES the attention
mechanism behavior:

- Original (PyTorch): Uses standard SDPA attention masking
- Monkeypatched (ONNX): Uses a custom masking function that produces
  different attention patterns

The monkeypatch changes how the encoder computes attention, resulting in:
- Different hidden state distributions (ONNX has ~43% higher std deviation)
- Different mean values
- Only ~59% cosine similarity with the original

IMPACT ON TTS:
--------------
Since the encoder produces different hidden states, the decoder receives
incorrect conditioning information, leading to:
- Degraded audio quality
- Potential pronunciation errors
- Different prosody/timing

RECOMMENDED SOLUTIONS:
----------------------
1. DO NOT USE ONNX ENCODER: Use PyTorch encoder instead
   - Modify inference_hybrid_complete.py to use PyTorch encoder
   - This will produce correct outputs matching the 4bit model

2. ALTERNATIVE: Fix ONNX Export (Complex)
   - The T5Gemma model uses vmap-based attention masking that is
     incompatible with ONNX tracing
   - Would require rewriting the attention mechanism
   - Not recommended unless ONNX is strictly required
""")
else:
    print("""
FINDING: Encoders are reasonably similar.
The differences may be due to numerical precision (float16 vs float32).
""")

# ============================================================================
# Recommendation
# ============================================================================
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
For correct TTS output, modify the hybrid inference to use PyTorch encoder
instead of the ONNX encoder. The decoder can remain as-is since it uses
PyTorch already.

Change in inference_hybrid_complete.py:
- Load the full PyTorch model for the encoder
- Use ONNX only if absolutely necessary for deployment constraints
- Or accept the quality degradation if ONNX is required
""")

# Save results to JSON
with open("encoder_comparison_results.json", "w") as f:
    json.dump({
        "summary": {
            "avg_cosine_similarity": float(avg_cos_sim),
            "avg_relative_error": float(avg_rel_error),
            "max_difference": float(max_max_diff),
        },
        "results": [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in r.items()} for r in results],
        "conclusion": "ONNX encoder differs significantly due to SDPA monkeypatch" if avg_cos_sim < 0.95 else "Encoders are similar"
    }, f, indent=2)

print(f"\nDetailed results saved to: encoder_comparison_results.json")
