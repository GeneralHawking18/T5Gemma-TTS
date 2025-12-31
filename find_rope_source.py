
import torch
from transformers import AutoModelForSeq2SeqLM

model_name = "Aratako/T5Gemma-TTS-2b-2b"
try:
    print(f"Loading {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, device_map="cpu", torch_dtype=torch.float16)
    
    # Locate encoder
    encoder = None
    if hasattr(model, "encoder_module"):
        encoder = model.encoder_module
    elif hasattr(model, "backbone") and hasattr(model.backbone, "model"):
        encoder = model.backbone.model.encoder
    elif hasattr(model, "model") and hasattr(model.model, "encoder"):
        encoder = model.model.encoder
    
    if encoder:
        print(f"Encoder found: {type(encoder)}")
        # Try to find layers and RoPE
        layers = encoder.layers if hasattr(encoder, "layers") else []
        if layers:
             print(f"Layer 0 type: {type(layers[0])}")
             # Check for self_attn
             if hasattr(layers[0], "self_attn"):
                 attn = layers[0].self_attn
                 print(f"Attention type: {type(attn)}")
                 if hasattr(attn, "rotary_emb"):
                     rope = attn.rotary_emb
                     print(f"RoPE module found: {type(rope)}")
                     print(f"RoPE module location: {rope.__module__}")
                 else:
                     print("No 'rotary_emb' in self_attn")
        else:
            print("No layers found")

except Exception as e:
    print(f"Error: {e}")
