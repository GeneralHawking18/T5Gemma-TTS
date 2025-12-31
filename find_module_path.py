
import sys
import torch
from transformers import AutoModelForSeq2SeqLM

model_name = "Aratako/T5Gemma-TTS-2b-2b"
try:
    print(f"Loading {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, device_map="cpu", torch_dtype=torch.float16)
    
    import transformers.models.t5gemma.modeling_t5gemma as t5gemma_module
    print(f"Module file: {t5gemma_module.__file__}")
    
    # Also check what T5GemmaRotaryEmbedding is
    if hasattr(t5gemma_module, "T5GemmaRotaryEmbedding"):
        print(f"T5GemmaRotaryEmbedding found in module")
    else:
        print("T5GemmaRotaryEmbedding NOT found in module")

except Exception as e:
    print(f"Error: {e}")
