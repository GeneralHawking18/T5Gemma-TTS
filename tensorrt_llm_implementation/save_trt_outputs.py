"""
Run TensorRT model and save outputs for comparison with PyTorch.
This script saves intermediate outputs at each step for debugging.
"""

import tensorrt as trt
import tensorrt_llm
import torch
import numpy as np
import os

def run_and_save():
    # Register TensorRT-LLM plugins
    import tensorrt_llm
    # In newer versions of TRT-LLM, plugins are registered via this call
    # if it doesn't exist, we fallback to logger initialization which sometimes triggers it
    if hasattr(tensorrt_llm, 'register_plugins'):
        tensorrt_llm.register_plugins()
    elif hasattr(tensorrt_llm.runtime, 'register_plugins'):
        tensorrt_llm.runtime.register_plugins()
    
    # Ensure TRT-LLM logger is initialized which helps with plugin registry
    trt_logger = trt.Logger(trt.Logger.INFO)
    
    engine_path = "engine_output/t5gemma_decoder_new.engine"
    output_dir = "../validation_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading engine from {engine_path}...")
    with open(engine_path, "rb") as f:
        engine_buffer = f.read()

    # Create TensorRT runtime
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_buffer)
    context = engine.create_execution_context()

    # Print engine info
    print(f"\nEngine has {engine.num_io_tensors} I/O tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"  [{i}] {name}: shape={shape}, dtype={dtype}, mode={mode}")

    # Fixed shapes from build
    batch_size = 1
    dec_seq_len = 32
    enc_seq_len = 64
    hidden_size = 2304

    # Create deterministic inputs (use seed for reproducibility)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load inputs from validation_outputs if available
    input_dir = "../validation_outputs"
    print(f"Loading inputs from {input_dir}...")
    
    try:
        input_ids_np = np.load(f"{input_dir}/input_ids.npy")
        enc_hidden_np = np.load(f"{input_dir}/encoder_hidden_states.npy")
        pos_ids_np = np.load(f"{input_dir}/position_ids.npy")
        enc_pos_ids_np = np.load(f"{input_dir}/encoder_position_ids.npy")
        # Try to load encoder_attention_mask if it exists
        try:
            enc_mask_np = np.load(f"{input_dir}/encoder_attention_mask.npy")
        except:
            enc_mask_np = np.ones((batch_size, enc_seq_len), dtype=np.int32)
        
        input_ids = torch.from_numpy(input_ids_np).int().cuda().contiguous()
        encoder_hidden_states = torch.from_numpy(enc_hidden_np).to(torch.bfloat16).cuda().contiguous()
        position_ids = torch.from_numpy(pos_ids_np).float().cuda().contiguous()
        encoder_position_ids = torch.from_numpy(enc_pos_ids_np).float().cuda().contiguous()
        encoder_attention_mask = torch.from_numpy(enc_mask_np).int().cuda().contiguous()
        print("Inputs loaded successfully!")
    except Exception as e:
        print(f"Failed to load inputs: {e}")
        print("Falling back to random inputs...")
        # Input tensors
        input_ids = torch.randint(0, 1000, (batch_size, dec_seq_len), dtype=torch.int32).cuda().contiguous()
        encoder_hidden_states = torch.randn(batch_size, enc_seq_len, hidden_size, dtype=torch.bfloat16).cuda().contiguous()
        position_ids = torch.linspace(0, 1, dec_seq_len, dtype=torch.float32).unsqueeze(0).cuda().contiguous()
        encoder_position_ids = torch.linspace(0, 1, enc_seq_len, dtype=torch.float32).unsqueeze(0).cuda().contiguous()
        encoder_attention_mask = torch.ones(batch_size, enc_seq_len, dtype=torch.int32).cuda().contiguous()

    # Output tensors map
    outputs = {}

    # Allocate buffers for all outputs
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT:
            dtype = engine.get_tensor_dtype(name)
            # We assume all outputs have shape [batch, dec_seq, hidden_size] for now,
            # or we can query shape. But for dynamic shapes, we might need to know max?
            # Build used fixed shapes, so get_tensor_shape should work
            shape = engine.get_tensor_shape(name)
            # Convert TRT shape (Dims) to list
            shape_list = [shape[j] for j in range(len(shape))]
            
            # Map TRT dtype to Torch dtype
            if dtype == trt.DataType.HALF:
                torch_dtype = torch.float16
            elif dtype == trt.DataType.BF16:
                torch_dtype = torch.bfloat16
            elif dtype == trt.DataType.FLOAT:
                torch_dtype = torch.float32
            else:
                print(f"Warning: Unknown dtype {dtype} for {name}, using float32")
                torch_dtype = torch.float32
            
            tensor = torch.empty(tuple(shape_list), dtype=torch_dtype).cuda().contiguous()
            outputs[name] = tensor
            context.set_tensor_address(name, tensor.data_ptr())
            print(f"Allocated output buffer for {name}: {shape_list} {torch_dtype}")

    # Set input tensor addresses
    context.set_tensor_address("input_ids", input_ids.data_ptr())
    context.set_tensor_address("encoder_hidden_states", encoder_hidden_states.data_ptr())
    context.set_tensor_address("position_ids", position_ids.data_ptr())
    context.set_tensor_address("encoder_position_ids", encoder_position_ids.data_ptr())
    context.set_tensor_address("encoder_attention_mask", encoder_attention_mask.data_ptr())
    
    # Save inputs used for verification
    np.save(f"{output_dir}/input_ids.npy", input_ids.cpu().numpy())
    np.save(f"{output_dir}/encoder_hidden_states.npy", encoder_hidden_states.float().cpu().numpy())
    np.save(f"{output_dir}/position_ids.npy", position_ids.float().cpu().numpy())
    np.save(f"{output_dir}/encoder_position_ids.npy", encoder_position_ids.float().cpu().numpy())
    np.save(f"{output_dir}/encoder_attention_mask.npy", encoder_attention_mask.cpu().numpy())

    # Run inference
    print("\nRunning TensorRT inference...")
    stream = torch.cuda.Stream()
    success = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    if success:
        print("TensorRT inference completed!")
        
        for name, tensor in outputs.items():
            print(f"\nOutput {name}:")
            print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
            print(f"  Has NaN: {torch.isnan(tensor).any()}")
            
            # Save output (cast to float32 for numpy)
            fname = "trt_output.npy" if name == "output" else f"trt_{name}.npy"
            np.save(f"{output_dir}/{fname}", tensor.float().cpu().numpy())
            print(f"  Saved to {output_dir}/{fname}")
    else:
        print("TensorRT inference FAILED!")

    # Also save the TRT weights for reference
    print("\nSaving TRT weights info...")
    trt_weights = np.load("../trt_weights/weights.npz")
    weight_info = {}
    for key in trt_weights.files:
        weight_info[key] = {
            'shape': trt_weights[key].shape,
            'dtype': str(trt_weights[key].dtype),
            'min': float(trt_weights[key].min()),
            'max': float(trt_weights[key].max()),
        }

    import json
    with open(f"{output_dir}/trt_weights_info.json", 'w') as f:
        json.dump(weight_info, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    run_and_save()
