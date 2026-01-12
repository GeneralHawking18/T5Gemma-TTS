from transformers import AutoConfig

model_name = "Aratako/T5Gemma-TTS-2b-2b"
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print("Config loaded successfully.")
    print(f"Config keys: {list(config.__dict__.keys())}")
    # Also check if it wraps another config
    if hasattr(config, 'to_dict'):
         import json
         print(json.dumps(config.to_dict(), indent=2))

        
except Exception as e:
    print(f"Error loading config: {e}")
