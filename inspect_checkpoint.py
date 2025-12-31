from transformers import AutoModelForSeq2SeqLM, AutoConfig
from transformers.utils import cached_file
import json
import os
import torch

model_name = "Aratako/T5Gemma-TTS-2b-2b"

# Try to find the index file
try:
    index_file = cached_file(model_name, "model.safetensors.index.json")
    if not index_file:
        index_file = cached_file(model_name, "pytorch_model.bin.index.json")
    
    if index_file:
        print(f"Found index file: {index_file}")
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        weight_map = index.get("weight_map", {})
        
        encoder_keys = [k for k in weight_map.keys() if "encoder" in k]
        decoder_keys = [k for k in weight_map.keys() if "decoder" in k]
        
        print(f"Total keys: {len(weight_map)}")
        print(f"Encoder keys: {len(encoder_keys)}")
        print(f"Decoder keys: {len(decoder_keys)}")
        
        # Check if shards separate them
        encoder_shards = set(weight_map[k] for k in encoder_keys)
        decoder_shards = set(weight_map[k] for k in decoder_keys)
        
        print(f"Encoder shards: {encoder_shards}")
        print(f"Decoder shards: {decoder_shards}")
        
        pure_encoder_shards = encoder_shards - decoder_shards
        print(f"Pure encoder shards (can be skipped?): {pure_encoder_shards}")
        
    else:
        print("No index file found (maybe not sharded?).")

except Exception as e:
    print(f"Error: {e}")
