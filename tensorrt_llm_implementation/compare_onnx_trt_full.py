import os
import sys
import numpy as np
import torch
import tensorrt as trt
import onnxruntime as ort
from transformers import AutoTokenizer
import argparse

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def compare_tensors(name, onnx_val, trt_val):
    onnx_val = onnx_val.astype(np.float32)
    trt_val = trt_val.astype(np.float32)
    
    diff = np.abs(onnx_val - trt_val)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    max_val = np.max(np.abs(onnx_val))
    rel_diff = max_diff / (max_val + 1e-9) if max_val > 0 else 0.0
    
    print(f"\n{name}:")
    print(f"  ONNX shape: {onnx_val.shape}, range: [{onnx_val.min():.4f}, {onnx_val.max():.4f}]")
    print(f"  TRT  shape: {trt_val.shape}, range: [{trt_val.min():.4f}, {trt_val.max():.4f}]")
    print(f"  Max Diff: {max_diff:.6e}")
    print(f"  Mean Diff: {mean_diff:.6e}")
    print(f"  Rel Diff: {rel_diff:.6e}")
    
    status = "OK"
    if rel_diff > 0.05: status = "FAIL"
    elif rel_diff > 0.01: status = "WARN"
    
    print(f"  Status: {status}")
    return max_diff

def run_comparison(onnx_path, trt_path, tokenizer_path, text):
    print(f"Input text: '{text}'")
    
    # 1. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"Input IDs shape: {input_ids.shape}")

    # 2. Run ONNX
    print(f"\n--- Running ONNX ({onnx_path}) ---")
    sess_options = ort.SessionOptions()
    onnx_sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
    
    onnx_inputs = {}
    for inp in onnx_sess.get_inputs():
        if inp.name == "input_ids":
            onnx_inputs[inp.name] = input_ids.astype(np.int64)
        elif inp.name == "attention_mask":
            onnx_inputs[inp.name] = attention_mask.astype(np.int64)
        elif inp.name == "position_ids":
            # Some models might need position_ids
            seq_len = input_ids.shape[1]
            pos_ids = np.arange(seq_len, dtype=np.float32)[None, :]
            onnx_inputs[inp.name] = pos_ids
            
    onnx_outputs = onnx_sess.run(None, onnx_inputs)
    onnx_hidden_states = onnx_outputs[0]
    
    # 3. Run TensorRT
    print(f"\n--- Running TensorRT ({trt_path}) ---")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    
    # Set input shapes
    context.set_input_shape('input_ids', input_ids.shape)
    context.set_input_shape('attention_mask', attention_mask.shape)
    
    # Allocate GPU buffers
    input_ids_gpu = torch.from_numpy(input_ids.astype(np.int64)).cuda()
    attention_mask_gpu = torch.from_numpy(attention_mask.astype(np.int64)).cuda()
    
    # Assume output name from engine
    output_name = 'encoder_hidden_states'
    # Try to find output name if not default
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    
    if output_name not in output_names and output_names:
        output_name = output_names[0]
        
    out_shape = tuple(context.get_tensor_shape(output_name))
    output_gpu = torch.empty(out_shape, dtype=torch.float32, device='cuda')
    
    # Set addresses
    context.set_tensor_address('input_ids', input_ids_gpu.data_ptr())
    context.set_tensor_address('attention_mask', attention_mask_gpu.data_ptr())
    context.set_tensor_address(output_name, output_gpu.data_ptr())
    
    # Execute
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    
    trt_hidden_states = output_gpu.cpu().numpy()
    
    # 4. Save results
    np.savez("onnx.npz", hidden_states=onnx_hidden_states)
    np.savez("trt.npz", hidden_states=trt_hidden_states)
    print("\nSaved onnx.npz and trt.npz")
    
    # 5. Compare
    compare_tensors("Encoder Hidden States", onnx_hidden_states, trt_hidden_states)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="../onnx_models_fp16/encoder.onnx")
    parser.add_argument("--trt", type=str, default="../trt_weights/encoder_trt.engine")
    parser.add_argument("--tokenizer", type=str, default="../tokenizer_local")
    parser.add_argument("--text", type=str, default="Hello, this is a test for TensorRT and ONNX consistency.")
    
    args = parser.parse_args()
    
    # Adjust paths if they don't exist and are relative to project root
    def fix_path(p):
        if not os.path.exists(p):
            alt = os.path.join(PROJECT_ROOT, p.lstrip("./").lstrip("../"))
            if os.path.exists(alt): return alt
        return p

    onnx_path = fix_path(args.onnx)
    trt_path = fix_path(args.trt)
    tok_path = fix_path(args.tokenizer)
    
    try:
        run_comparison(onnx_path, trt_path, tok_path, args.text)
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
