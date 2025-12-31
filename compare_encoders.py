"""Compare ONNX Encoder vs PyTorch Encoder hidden states."""
import os
# Load HF_TOKEN
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("=" * 60)
print("CRITICAL: Comparing Encoder Hidden States")
print("ONNX Encoder vs PyTorch Encoder")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model first (tokenizer will be derived from it)
print("\n[1] Loading PyTorch model (4-bit)...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()
cfg = model.config

# Get tokenizer name from config
tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
print(f"[2] Loading tokenizer from: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

# Test text
text = "こんにちは"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print(f"\nInput text: {text}")
print(f"Input IDs: {input_ids[0].tolist()}")

# ========================================
# 1. ONNX Encoder
# ========================================
print("\n[3] Running ONNX Encoder...")
session = ort.InferenceSession('onnx_models_fp16/encoder.onnx', providers=['CPUExecutionProvider'])
onnx_output = session.run(None, {
    "input_ids": input_ids.numpy().astype(np.int64),
    "attention_mask": attention_mask.numpy().astype(np.int64),
})[0]

print(f"ONNX output shape: {onnx_output.shape}")
print(f"ONNX output sample: {onnx_output[0, 0, :5]}")
print(f"ONNX stats: mean={onnx_output.mean():.6f}, std={onnx_output.std():.6f}")
print(f"ONNX range: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")

# ========================================
# 2. PyTorch Encoder
# ========================================
print("\n[4] Running PyTorch Encoder...")
with torch.no_grad():
    input_ids_pt = input_ids.to(model.device)
    attention_mask_pt = attention_mask.to(model.device)
    
    # T5Gemma uses progress monitoring RoPE - compute position IDs
    use_pm_rope = getattr(cfg, "use_pm_rope", True)
    progress_scale = getattr(cfg, "progress_scale", 2000.0)
    
    position_ids = None
    if use_pm_rope:
        x_lens = attention_mask_pt.sum(dim=1)
        max_len = input_ids_pt.shape[1]
        pos = torch.arange(max_len, device=model.device, dtype=torch.float32)[None, :]
        denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
        position_ids = pos / denom * progress_scale
        mask = pos < x_lens[:, None]
        position_ids = position_ids.masked_fill(~mask, 0.0)
        print(f"Position IDs (PM-RoPE): {position_ids[0, :].tolist()}")
    
    # Run encoder - T5GemmaVoice uses backbone.model.encoder
    if hasattr(model, 'backbone'):
        encoder = model.backbone.model.encoder
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'encoder'):
            encoder = model.model.encoder
        else:
            encoder = model.model.model.encoder
    else:
        raise AttributeError("Cannot find encoder in model structure")
    pytorch_output = encoder(
        input_ids=input_ids_pt,
        attention_mask=attention_mask_pt,
        position_ids=position_ids,
    )
    hidden_states = pytorch_output.last_hidden_state.float().cpu().numpy()

print(f"PyTorch output shape: {hidden_states.shape}")
print(f"PyTorch output sample: {hidden_states[0, 0, :5]}")
print(f"PyTorch stats: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")
print(f"PyTorch range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

# ========================================
# 3. Compare
# ========================================
print("\n[5] Comparison:")
diff = np.abs(onnx_output - hidden_states)
print(f"Max difference: {diff.max():.6f}")
print(f"Mean difference: {diff.mean():.6f}")
print(f"Relative error: {diff.mean() / (np.abs(hidden_states).mean() + 1e-8) * 100:.4f}%")

# Cosine similarity
onnx_flat = onnx_output.flatten()
pt_flat = hidden_states.flatten()
cos_sim = np.dot(onnx_flat, pt_flat) / (np.linalg.norm(onnx_flat) * np.linalg.norm(pt_flat))
print(f"Cosine similarity: {cos_sim:.6f}")

if cos_sim > 0.99:
    print("\n✅ Encoders are highly similar!")
elif cos_sim > 0.95:
    print("\n⚠️ Encoders are somewhat similar but may have issues")
else:
    print("\n❌ CRITICAL: Encoders have significant differences!")
    print("   This could be causing the audio quality issues.")
