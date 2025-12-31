from transformers import AutoConfig, AutoModelForSeq2SeqLM
import torch

model_name = "Aratako/T5Gemma-TTS-2b-2b"
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # Load skeleton
    with torch.device("meta"):
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True)

    print("Top level modules:")
    for name, _ in model.named_children():
        print(name)
        
    print("\nARGS:")
    if hasattr(model, "args"):
        print(model.args)
    else:
        print("No .args found on model")
        
    print("\nCONFIG:")
    print(config)
except Exception as e:
    print(e)
