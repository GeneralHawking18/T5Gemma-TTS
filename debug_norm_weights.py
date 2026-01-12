import torch
from transformers import AutoModelForSeq2SeqLM

model_name = "Aratako/T5Gemma-TTS-2b-2b"
print(f"Loading {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")
state_dict = model.state_dict()

# Look for norm weights
norm_weights = [k for k in state_dict.keys() if "norm" in k and "weight" in k]
for k in norm_weights[:5]:
    w = state_dict[k]
    print(f"{k}: mean={w.mean().item():.4f}, std={w.std().item():.4f}, min={w.min().item():.4f}, max={w.max().item():.4f}")
