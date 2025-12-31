import torch
import time
from inference_hybrid import HybridT5Gemma

def main():
    print("Initializing Hybrid Inference (FP16 ONNX Encoder + PyTorch Decoder)...")
    
    # Initialize implementation with FP16 ONNX path
    tts = HybridT5Gemma(
        model_name="Aratako/T5Gemma-TTS-2b-2b", 
        onnx_dir="./onnx_models_fp16",
        use_int8=False,  # Use FP16
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\nModel initialized. Ready for inference requests.")
    
    # Verify encoder type
    if tts.use_onnx_encoder:
        print("SUCCESS: Using ONNX Encoder (FP16)")
    else:
        print("WARNING: Fallback to PyTorch Encoder (Check onnx_models_fp16 path)")

if __name__ == "__main__":
    main()
