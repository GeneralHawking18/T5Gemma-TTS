import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
import torch
import numpy as np
import argparse
import os

def run(engine_dir, prompt_text="Test"):
    engine_path = os.path.join(engine_dir, "t5gemma_decoder.engine")
    print(f"Loading engine from {engine_path}...")

    with open(engine_path, "rb") as f:
        engine_buffer = f.read()

    # Create TensorRT runtime and deserialize engine
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_buffer)

    # Print binding info for debugging
    print(f"\nEngine has {engine.num_io_tensors} I/O tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"  [{i}] {name}: shape={shape}, dtype={dtype}, mode={mode}")

    # Create execution context
    context = engine.create_execution_context()
    
    # Inputs must match the fixed shapes used during engine build
    batch_size = 1
    dec_seq_len = 32   # Must match DEC_SEQ in build.py
    enc_seq_len = 64   # Must match ENC_SEQ in build.py
    hidden_size = 2304 # Must match config.hidden_size

    # 1. Encoder Hidden States (From Encoder) - must be contiguous
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, hidden_size, dtype=torch.float16).cuda().contiguous()

    # 2. Position IDs (PM-RoPE progress values 0-1) - must be contiguous
    position_ids = torch.linspace(0, 1, dec_seq_len, dtype=torch.float32).unsqueeze(0).cuda().contiguous()

    # 3. Encoder Position IDs - must be contiguous
    encoder_position_ids = torch.linspace(0, 1, enc_seq_len, dtype=torch.float32).unsqueeze(0).cuda().contiguous()

    # 4. Input IDs [batch, dec_seq_len] - must be contiguous
    input_ids = torch.randint(0, 256000, (batch_size, dec_seq_len), dtype=torch.int32).cuda().contiguous()

    # 5. Encoder Attention Mask - must be contiguous
    encoder_attention_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.int32).cuda().contiguous()

    # Debug: Print input stats
    print(f"\nInput stats:")
    print(f"  input_ids: {input_ids.shape}, range=[{input_ids.min()}, {input_ids.max()}]")
    print(f"  encoder_hidden: {encoder_hidden_states.shape}, range=[{encoder_hidden_states.min():.4f}, {encoder_hidden_states.max():.4f}]")
    print(f"  position_ids: {position_ids.shape}, range=[{position_ids.min():.4f}, {position_ids.max():.4f}]")
    
    print("\nRunning Inference Step...")

    # Prepare output tensor - shape matches [batch, dec_seq, hidden_size]
    # Note: Model outputs hidden states, not vocab logits
    output_tensor = torch.empty(batch_size, dec_seq_len, hidden_size, dtype=torch.float16).cuda()

    # Set tensor addresses for TensorRT context
    context.set_tensor_address("input_ids", input_ids.data_ptr())
    context.set_tensor_address("encoder_hidden_states", encoder_hidden_states.data_ptr())
    context.set_tensor_address("position_ids", position_ids.data_ptr())
    context.set_tensor_address("encoder_position_ids", encoder_position_ids.data_ptr())
    context.set_tensor_address("encoder_attention_mask", encoder_attention_mask.data_ptr())
    context.set_tensor_address("output", output_tensor.data_ptr())

    # Create CUDA stream and execute
    stream = torch.cuda.Stream()
    success = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    if success:
        print("Inference Successful!")
        print(f"Output shape: {output_tensor.shape}")
        print(f"Output dtype: {output_tensor.dtype}")
        print(f"Output sample values (first 10): {output_tensor[0, 0, :10]}")
        print(f"Output max: {output_tensor.max()}, min: {output_tensor.min()}")
    else:
        print("Inference FAILED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_dir", type=str, default="engine_output")
    args = parser.parse_args()
    run(args.engine_dir)
