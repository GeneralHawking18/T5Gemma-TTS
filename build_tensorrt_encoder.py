"""
Build TensorRT Engine for T5Gemma Encoder.
Requires: pip install tensorrt
"""
import os
import tensorrt as trt
import sys

def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # 1. Network Definition
    # 1 << 0 = Explicit Batch
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 2. Config
    config = builder.create_builder_config()
    # Memory pool limit (e.g., 4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[Info] FP16 enabled.")
        else:
            print("[Warn] FP16 not supported on this platform, falling back to FP32.")

    # 3. Parse ONNX
    print(f"[Info] Parsing ONNX: {onnx_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("[Error] Failed to parse ONNX file:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 4. Optimization Profile (Dynamic Shapes)
    # T5 Encoder inputs: input_ids [Batch, Seq], attention_mask [Batch, Seq]
    profile = builder.create_optimization_profile()
    
    # Define Min, Opt, Max shapes
    # Adjust 'Max' based on your longest expected text (e.g., 512 chars)
    min_shape = (1, 1)
    opt_shape = (1, 128) 
    max_shape = (1, 512)
    
    profile.set_shape("input_ids", min_shape, opt_shape, max_shape)
    profile.set_shape("attention_mask", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 5. Build Engine
    print("[Info] Building TensorRT Engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("[Error] Engine build failed.")
        return None
        
    # 6. Save
    print(f"[Info] Saving engine to {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("[Success] Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to encoder.onnx")
    parser.add_argument("--output", type=str, default="encoder.engine", help="Output path")
    args = parser.parse_args()
    
    build_engine(args.onnx, args.output)
