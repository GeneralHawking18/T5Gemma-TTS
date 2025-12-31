from transformers import AutoConfig
import sys

model_name = "Aratako/T5Gemma-TTS-2b-2b"
print(f"Loading config for {model_name}...")
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print("Config loaded.")
    print(f"Keys: {list(config.to_dict().keys())}")
    
    if hasattr(config, "layer_types"):
        print(f"layer_types: {config.layer_types}")
    else:
        print("layer_types MISSING")
        
    # Check t5gemma_model_name
    if hasattr(config, "t5gemma_model_name"):
        print(f"backbone_name: {config.t5gemma_model_name}")

except Exception as e:
    print(f"Error: {e}")
