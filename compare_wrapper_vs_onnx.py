"""Compare PyTorch EncoderWrapper vs ONNX exported model."""
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
from onnx_modules import EncoderWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading PyTorch model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()
cfg = model.config

# Setup model attributes
if not hasattr(model, "args"):
    model.args = model.config
if not hasattr(model, "encoder_module"):
    model.encoder_module = model.model.encoder
if not hasattr(model, "text_input_type"):
    model.text_input_type = getattr(model.config, "text_input_type", "text")
if not hasattr(model, "progress_scale"):
    model.progress_scale = getattr(model.config, "progress_scale", 2000.0)

wrapper = EncoderWrapper(model)
wrapper.eval()

print("Loading ONNX model...")
onnx_session = ort.InferenceSession(
    'onnx_models_fp16/encoder.onnx',
    providers=['CPUExecutionProvider']
)

tokenizer_name = getattr(cfg, "text_tokenizer_name", None)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

# Test
text = "こんにちは"
add_eos = getattr(cfg, "add_eos_to_text", 1)
tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
if add_eos:
    tokens.append(add_eos)

input_ids = torch.tensor([tokens], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

print(f"\nText: {text}")
print(f"Tokens: {tokens}")

with torch.no_grad():
    # PyTorch EncoderWrapper
    wrapper_out = wrapper(input_ids.to(model.device), attention_mask.to(model.device))
    wrapper_np = wrapper_out.float().cpu().numpy()

# ONNX
onnx_out = onnx_session.run(None, {
    "input_ids": input_ids.numpy().astype(np.int64),
    "attention_mask": attention_mask.numpy().astype(np.int64),
})[0]

cos_sim = np.dot(wrapper_np.flatten(), onnx_out.flatten()) / (
    np.linalg.norm(wrapper_np.flatten()) * np.linalg.norm(onnx_out.flatten()) + 1e-8
)
diff = np.abs(wrapper_np - onnx_out)

print(f"\nPyTorch Wrapper: mean={wrapper_np.mean():.6f}, std={wrapper_np.std():.6f}")
print(f"ONNX:            mean={onnx_out.mean():.6f}, std={onnx_out.std():.6f}")
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
print(f"Cosine similarity: {cos_sim:.6f}")

# Now let's investigate WHY they differ
print("\n" + "="*60)
print("INVESTIGATING THE DIFFERENCE")
print("="*60)

# Check specific values
print(f"\nFirst 5 values comparison:")
print(f"PyTorch: {wrapper_np[0, 0, :5]}")
print(f"ONNX:    {onnx_out[0, 0, :5]}")

# The issue might be with how ONNX handles certain operations
# Let's check if there's a constant folding or optimization issue

# Check input/output shapes
print(f"\nInput shape: {input_ids.shape}")
print(f"Output shapes: PyTorch={wrapper_np.shape}, ONNX={onnx_out.shape}")

# Let's see if the ONNX model was exported with the correct settings
# by checking if a fresh export produces different results
print("\n" + "="*60)
print("TESTING FRESH ONNX EXPORT")
print("="*60)

import tempfile

# Export fresh
wrapper.to("cpu")  # Move to CPU for export
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    temp_onnx_path = f.name

print(f"Exporting fresh ONNX to: {temp_onnx_path}")

dummy_inputs = (
    torch.randint(0, 1000, (1, 2), dtype=torch.long),  # Same as our test input length
    torch.ones((1, 2), dtype=torch.long)
)

torch.onnx.export(
    wrapper,
    dummy_inputs,
    temp_onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input_ids', 'attention_mask'],
    output_names=['encoder_hidden_states'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'encoder_hidden_states': {0: 'batch', 1: 'sequence'},
    }
)

# Load and test
fresh_session = ort.InferenceSession(temp_onnx_path, providers=['CPUExecutionProvider'])
fresh_out = fresh_session.run(None, {
    "input_ids": input_ids.numpy().astype(np.int64),
    "attention_mask": attention_mask.numpy().astype(np.int64),
})[0]

cos_sim_fresh = np.dot(wrapper_np.flatten(), fresh_out.flatten()) / (
    np.linalg.norm(wrapper_np.flatten()) * np.linalg.norm(fresh_out.flatten()) + 1e-8
)

print(f"\nFresh ONNX: mean={fresh_out.mean():.6f}, std={fresh_out.std():.6f}")
print(f"Cosine similarity (PyTorch vs Fresh ONNX): {cos_sim_fresh:.6f}")

if cos_sim_fresh > 0.99:
    print("\n✅ Fresh export matches PyTorch!")
    print("   The issue is with the existing ONNX file, not the export process.")
else:
    print("\n❌ Fresh export also differs!")
    print("   There might be an issue with ONNX export of this model.")

# Compare existing vs fresh ONNX
cos_sim_onnx = np.dot(onnx_out.flatten(), fresh_out.flatten()) / (
    np.linalg.norm(onnx_out.flatten()) * np.linalg.norm(fresh_out.flatten()) + 1e-8
)
print(f"\nCosine similarity (Existing ONNX vs Fresh ONNX): {cos_sim_onnx:.6f}")

# Cleanup
os.remove(temp_onnx_path)
