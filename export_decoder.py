import torch
import os
import argparse
import safetensors.torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

try:
    from models.t5gemma import T5GemmaVoiceModel
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    try:
        from models.t5gemma import T5GemmaVoiceModel
    except ImportError:
        pass # Might not need explicit class if AutoModel works

def export_decoder(model_name, output_path, use_fp32=True):
    print(f"Instantiating model via AutoModelForSeq2SeqLM from {model_name}...")
    # Use BFloat16 for stable inference (same memory as FP16 but FP32-like range)
    # BF16 max ~3.4e38 vs FP16 max ~65504 (avoids overflow in decoder)
    dtype = torch.bfloat16
    print(f"Loading model with dtype: {dtype}")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=dtype
        )
    except Exception as e:
        print(f"AutoModel failed: {e}")
        raise e

    print("Model loaded via transformers. Verifying structure...")
    
    print("Filtering keys for Decoder-only export...")
    full_state = model.state_dict()
    decoder_state = {}
    
    count = 0
    dropped = 0
    
    for k, v in full_state.items():
        # Drop encoder keys
        if "backbone.encoder" in k:
            dropped += 1
            continue
        # Added checking for backbone.model.encoder
        if "backbone.model.encoder" in k:
            dropped += 1
            continue
        if "encoder_module" in k:
            dropped += 1
            continue
        if k.startswith("encoder."):
            dropped += 1
            continue
            
        # Keep embeddings, decoders, heads
        decoder_state[k] = v
        count += 1
        
    print(f"Kept {count} keys. Dropped {dropped} encoder keys.")
    
    # Force bin to handle shared tensors automatically
    if output_path.endswith(".safetensors"):
        output_path = output_path.replace(".safetensors", ".bin")
        print(f"Switched to .bin format for shared tensor support: {output_path}")

    print(f"Saving to {output_path}...")
    torch.save(decoder_state, output_path)
        
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Aratako/T5Gemma-TTS-2b-2b")
    parser.add_argument("--output_path", type=str, default="weights/t5gemma_decoder_only.bin")
    args = parser.parse_args()
    
    export_decoder(args.model_name, args.output_path)
