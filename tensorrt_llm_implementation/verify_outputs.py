import torch
import numpy as np
import tensorrt_llm
from transformers import AutoConfig
import os
import sys
import argparse

# Add project root to path to import hf_export
sys.path.append("/app")

try:
    from hf_export.modeling_t5gemma_voice import T5GemmaVoiceForConditionalGeneration
    from hf_export.configuration_t5gemma_voice import T5GemmaVoiceConfig
except ImportError:
    print("Could not import T5GemmaVoice classes. Ensure /app is mapped correctly.")
    sys.exit(1)

def verify(weights_dir, engine_dir):
    print("\n[Verification] Initializing...")
    device = "cuda"
    
    # --- 1. Load PyTorch Model ---
    print("[Verification] Loading PyTorch model from", weights_dir)
    # Load config (assuming config.json is in weights_dir)
    try:
        config = T5GemmaVoiceConfig.from_pretrained(weights_dir)
    except Exception as e:
        print(f"Warning: Could not load config from {weights_dir}, using default. Error: {e}")
        config = T5GemmaVoiceConfig()
        # Hardcode 2B params if defaults differ
        config.d_model = 2048
        config.d_ff = 5120
        config.num_layers = 24
        config.num_heads = 32
        config.d_kv = 64

    # Force float16
    model = T5GemmaVoiceForConditionalGeneration(config)
    
    # Load Weights
    bin_path = os.path.join(weights_dir, "t5gemma_decoder_only.bin")
    if not os.path.exists(bin_path):
        bin_path = os.path.join(weights_dir, "pytorch_model.bin")
    
    print(f"[Verification] Loading state dict from {bin_path}")
    if os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    else:
        print("[Error] Weights file not found!")
        sys.exit(1)
        
    model.to(device).half().eval()
    decoder = model.decoder_module
    
    # --- 2. Load TRT Engine ---
    engine_path = os.path.join(engine_dir, "t5gemma_decoder_new.engine")
    if not os.path.exists(engine_path):
        engine_path = os.path.join(engine_dir, "t5gemma_decoder.engine")
    print(f"[Verification] Loading TRT Engine from {engine_path}")
    
    with open(engine_path, "rb") as f:
        engine_buffer = f.read()
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    
    # --- 3. Prepare Common Inputs ---
    print("[Verification] Preparing inputs...")
    torch.manual_seed(42)
    
    batch_size = 1
    enc_seq_len = 64
    dec_seq_len = 32 # Match build.py fixed shape for now to avoid complexity
    hidden_size = config.d_model # 2048 or 2304
    
    # Decoder Input IDs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, dec_seq_len), dtype=torch.int32).to(device)
    
    # Encoder Hidden States (Mock)
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, hidden_size, dtype=torch.float16).to(device)
    # Match build.py int32
    encoder_attention_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.int32).to(device)
    
    # Position IDs (PM-RoPE progress 0-1)
    pos_ids = torch.linspace(0, 1, dec_seq_len, dtype=torch.float32).unsqueeze(0).to(device)
    enc_pos_ids = torch.linspace(0, 1, enc_seq_len, dtype=torch.float32).unsqueeze(0).to(device)
    
    # --- 4. Run PyTorch Inference ---
    print("[Verification] Running PyTorch Forward...")
    with torch.no_grad():
        try:
            pt_out = decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask.long(),
                pm_decoder_position_ids=pos_ids,
                pm_encoder_position_ids=enc_pos_ids
            )
            pt_hidden = pt_out.last_hidden_state
        except Exception as e:
            print(f"[PyTorch Error] {e}")
            print("Trying inputs_embeds...")
            embeds = model.decoder_module.embed_tokens(input_ids)
            pt_out = decoder(
                inputs_embeds=embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask.long(),
                pm_decoder_position_ids=pos_ids,
                pm_encoder_position_ids=enc_pos_ids
            )
            pt_hidden = pt_out.last_hidden_state

    # --- 5. Run TRT Inference ---
    print("[Verification] Running TRT Forward...")
    trt_inputs = {
        "input_ids": input_ids,
        "encoder_hidden_states": encoder_hidden_states.to(torch.bfloat16),
        "position_ids": pos_ids,
        "encoder_position_ids": enc_pos_ids,
        "encoder_attention_mask": encoder_attention_mask
    }
    trt_outputs = session.run(trt_inputs)
    
    # build.py marks output as 'output'
    trt_hidden = trt_outputs.get('output')
    if trt_hidden is None:
        for k, v in trt_outputs.items():
            if v.shape == pt_hidden.shape:
                trt_hidden = v
                print(f"Found matching TRT output in key '{k}'")
                break
            
    if trt_hidden is None:
        # Fallback for shape mismatch (maybe 1 vs 0 dim?)
        trt_hidden = list(trt_outputs.values())[0]
        
    # --- 6. Compare ---
    print("\n[Comparison]")
    pt_np = pt_hidden.cpu().float().numpy()
    trt_np = trt_hidden.cpu().float().numpy()
    
    diff = np.abs(pt_np - trt_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"PyTorch Shape: {pt_np.shape}")
    print(f"TRT Shape:     {trt_np.shape}")
    print(f"Max Diff:      {max_diff:.6f}")
    print(f"Mean Diff:     {mean_diff:.6f}")
    
    # Threshold for fp16
    if max_diff < 1e-2:
        print("\n✅ PASSED: Outputs match within tolerance.")
    else:
        print("\n❌ FAILED: Outputs diverge significantly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--engine_dir", type=str, default="engine_output")
    args = parser.parse_args()
    
    verify(args.weights_dir, args.engine_dir)