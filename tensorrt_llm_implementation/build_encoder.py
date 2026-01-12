import argparse
import os
import tensorrt as trt
import sys

def build_engine(onnx_path, engine_path):
    print(f"[Info] Starting TensorRT Engine Build")
    print(f"  - ONNX: {onnx_path}")
    print(f"  - Output: {engine_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # 1. Network Definition
    # 1 << 0 = Explicit Batch
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 2. Config
    config = builder.create_builder_config()
    # Memory pool limit (e.g., 8GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 * 1024 * 1024)
    
    # if builder.platform_has_fast_fp16:
    #     config.set_flag(trt.BuilderFlag.FP16)
    #     print("[Info] FP16 enabled.")
    # else:
    print("[Info] FP16 disabled (forcing FP32 for stability).")

    # 3. Parse ONNX
    print(f"[Info] Parsing ONNX...")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        
    # Use parse_from_file to correctly handle external weights
    if not parser.parse_from_file(onnx_path):
        print("[Error] Failed to parse ONNX file:")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    # 4. Optimization Profile (Dynamic Shapes)
    # T5 Encoder inputs: input_ids [Batch, Seq], attention_mask [Batch, Seq]
    profile = builder.create_optimization_profile()
    
    # Define Min, Opt, Max shapes
    # We set max sequence length to 512 to speed up build
    min_shape = (1, 1)
    opt_shape = (1, 128) 
    max_shape = (4, 512)
    
    print(f"[Info] Setting Optimization Profile:")
    print(f"  - Min: {min_shape}")
    print(f"  - Opt: {opt_shape}")
    print(f"  - Max: {max_shape}")

    profile.set_shape("input_ids", min_shape, opt_shape, max_shape)
    profile.set_shape("attention_mask", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 5. Build Engine
    print("[Info] Building serialized network (this may take a few minutes)...")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"[Error] Exception during build: {e}")
        return None
    
    if serialized_engine is None:
        print("[Error] Engine build failed (serialized_engine is None).")
        return None
        
    # 6. Save
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    print(f"[Info] Saving engine to {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("[Success] Encoder Engine built successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to encoder.onnx")
    parser.add_argument("--output", type=str, required=True, help="Path to save encoder.engine")
    args = parser.parse_args()
    
    build_engine(args.onnx, args.output)
