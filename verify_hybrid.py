
import sys
import os
import torch
import numpy as np

# Ensure current dir is in path
sys.path.append(os.getcwd())

try:
    from inference_hybrid import HybridT5Gemma
    
    print("Initializing HybridT5Gemma...")
    tts = HybridT5Gemma(
        model_name="Aratako/T5Gemma-TTS-2b-2b",
        onnx_dir="./onnx_models_fp16", # Test with FP16 while INT8 is building
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_int8=False
    )
    
    print("Model initialized successfully.")
    print(f"Using ONNX Encoder: {tts.use_onnx_encoder}")
    
    # Create dummy inputs to test inference_tts flow
    print("Testing inference_tts flow...")
    
    # Dummy tokens 'Hello world'
    text = "Hello world"
    inputs = tts.model.model.encoder_module.embed_tokens(torch.tensor([[101, 102, 103]], device=tts.device)) # Simulate embedding or just use raw forward if accessible?
    # Actually inference_tts takes input_ids if text_input_type is text
    
    input_ids = torch.tensor([[101, 102, 103]], device=tts.device, dtype=torch.long)
    input_lens = torch.tensor([3], device=tts.device, dtype=torch.long)
    
    # Dummy prompt (audio codes) - 1 codebook, sequence length 10
    prompt = torch.randint(0, 100, (1, 1, 10), device=tts.device, dtype=torch.long)
    tgt_lens = torch.tensor([20], device=tts.device, dtype=torch.long) # Gen 10 more tokens
    
    try:
        concat_frames, gen_frames = tts.inference_tts(
            x=input_ids,
            x_lens=input_lens,
            y=prompt,
            tgt_y_lens=tgt_lens,
            num_samples=1
        )
        print("Inference successful!")
        print(f"Generated frames shape: {gen_frames.shape}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Verification failed: {e}")
    import traceback
    traceback.print_exc()
