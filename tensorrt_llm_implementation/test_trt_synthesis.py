
import os
import torch
import numpy as np
import logging
from inference_trt import TRTTTSModelHolder, SynthesizeRequest

logging.basicConfig(level=logging.INFO)

def test():
    holder = TRTTTSModelHolder()
    try:
        # Load model using internal container paths
        holder.load_model(
            onnx_dir="/app/onnx_models_fp16_fixed",
            engine_path="/app/tensorrt_llm_implementation/engine_output/t5gemma_decoder_new.engine",
            weights_path="/app/weights/decoder_pmrope.bin"
        )
        
        print("\n[Test] Model loaded successfully. Starting synthesis...")
        
        request = SynthesizeRequest(
            text="Hello, this is a test.",
            top_k=30,
            temperature=0.7
        )
        
        # Test 1 token step or small generation
        sr, audio, inf_time = holder.synthesize(request)
        
        print(f"\n[Success] Synthesized audio: {len(audio)} samples, SR: {sr}")
        print(f"[Success] Inference time: {inf_time:.2f}s")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[Failed] Test failed: {e}")

if __name__ == "__main__":
    test()
