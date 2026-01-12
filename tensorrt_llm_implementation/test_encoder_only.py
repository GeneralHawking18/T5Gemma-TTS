import tensorrt as trt
import torch
import numpy as np
import os
import argparse

def test_encoder(engine_path):
    print(f"Testing Encoder Engine: {engine_path}")
    if not os.path.exists(engine_path):
        print("Engine not found!")
        return
        
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    
    # Input
    seq_len = 64
    # TRT often expects Int64 for T5 inputs if exported that way
    input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long).cuda()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long).cuda()
    
    context.set_input_shape("input_ids", (1, seq_len))
    context.set_input_shape("attention_mask", (1, seq_len))
    
    # Output
    hidden_size = 2304
    output = torch.empty((1, seq_len, hidden_size), dtype=torch.float16, device="cuda")
    
    # Bindings
    context.set_tensor_address("input_ids", input_ids.data_ptr())
    context.set_tensor_address("attention_mask", attention_mask.data_ptr())
    context.set_tensor_address("encoder_hidden_states", output.data_ptr())
    
    # Run
    print("Running inference...")
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    
    # Stats
    out_cpu = output.cpu().numpy()
    print(f"Output Shape: {out_cpu.shape}")
    print(f"Mean: {out_cpu.mean():.4f}")
    print(f"Std: {out_cpu.std():.4f}")
    print(f"Max: {out_cpu.max():.4f}")
    print(f"Min: {out_cpu.min():.4f}")
    
    if np.isnan(out_cpu).any():
        print("FAILED: Output contains NaNs!")
    elif out_cpu.max() == 0 and out_cpu.min() == 0:
        print("WARNING: Output is all zeros!")
    else:
        print("PASSED: Encoder output looks valid.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="/app/trt_weights/encoder.engine")
    args = parser.parse_args()
    test_encoder(args.engine)
