import os
import sys
import torch
import numpy as np
import tensorrt_llm
from tensorrt_llm.runtime import Session

def compare(name, trt_val, pt_path):
    if not os.path.exists(pt_path):
        print(f"[{name}] PT Golden not found at {pt_path}")
        return
    pt_val = np.load(pt_path)
    
    # Ensure same dtype and shape for comparison
    # TRT outputs might have extra dims or different layout
    if trt_val.shape != pt_val.shape:
        # Try to squeeze to match
        trt_val_sq = np.squeeze(trt_val)
        pt_val_sq = np.squeeze(pt_val)
        if trt_val_sq.shape == pt_val_sq.shape:
            trt_val = trt_val_sq
            pt_val = pt_val_sq
        else:
            print(f"[{name}] Shape mismatch: TRT {trt_val.shape} vs PT {pt_val.shape}")
            return

    trt_val = trt_val.astype(np.float32)
    pt_val = pt_val.astype(np.float32)
    
    diff = np.abs(trt_val - pt_val)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    # Check ranges to see if scaling is the issue
    pt_range = [pt_val.min(), pt_val.max()]
    trt_range = [trt_val.min(), trt_val.max()]
    
    status = "OK" if max_diff < 1e-2 else "FAIL"
    if max_diff > 1.0: status = "CRITICAL"
    
    print(f"{name:<20}: MaxDiff={max_diff:.6f}, MeanDiff={mean_diff:.6f} [{status}]")
    print(f"{ ' ':20}  PT Range: [{pt_range[0]:.4f}, {pt_range[1]:.4f}]")
    print(f"{ ' ':20}  TRT Range: [{trt_range[0]:.4f}, {trt_range[1]:.4f}]")

def verify():
    engine_path = "./engine/l0_debug.engine"
    golden_dir = "../golden_outputs"
    
    if not os.path.exists(engine_path):
        print(f"Engine not found: {engine_path}")
        return

    # 1. Load Engine
    print(f"Loading engine: {engine_path}")
    with open(engine_path, 'rb') as f:
        engine_buffer = f.read()
    
    # TRT-LLM Session handles execution
    session = Session.from_serialized_engine(engine_buffer)
    
    # 2. Prepare Inputs from Golden Data
    print(f"Loading inputs from {golden_dir}...")
    input_ids = np.load(f"{golden_dir}/input_ids.npy").astype(np.int32)
    enc_hidden = np.load(f"{golden_dir}/encoder_hidden_states.npy")
    pos_ids = np.load(f"{golden_dir}/position_ids.npy").astype(np.float32)
    enc_pos_ids = np.load(f"{golden_dir}/encoder_position_ids.npy").astype(np.float32)
    
    # Map to torch tensors on GPU
    inputs = {
        'input_ids': torch.from_numpy(input_ids).cuda(),
        'encoder_hidden_states': torch.from_numpy(enc_hidden).cuda().to(torch.bfloat16),
        'position_ids': torch.from_numpy(pos_ids).cuda(),
        'encoder_position_ids': torch.from_numpy(enc_pos_ids).cuda(),
        'encoder_attention_mask': torch.ones(input_ids.shape[0], enc_hidden.shape[1], dtype=torch.int32).cuda()
    }
    
    # 3. Run Inference
    print("Running inference...")
    # Get output names from engine to pre-allocate
    import tensorrt as trt
    output_names = [session.engine.get_tensor_name(i) 
                   for i in range(session.engine.num_io_tensors) 
                   if session.engine.get_tensor_mode(session.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    
    stream = torch.cuda.current_stream()
    outputs = {}
    for name in output_names:
        shape = session.engine.get_tensor_shape(name)
        trt_dtype = session.engine.get_tensor_dtype(name)
        
        # Map TRT dtype to torch dtype
        torch_dtype = torch.float32
        if trt_dtype == trt.DataType.BF16:
            torch_dtype = torch.bfloat16
        elif trt_dtype == trt.DataType.HALF:
            torch_dtype = torch.float16
        elif trt_dtype == trt.DataType.INT32:
            torch_dtype = torch.int32
        
        outputs[name] = torch.empty(tuple(shape), dtype=torch_dtype, device='cuda')
    
    session.run(inputs, outputs, stream.cuda_stream)
    torch.cuda.synchronize()
    
    # 4. Surgical Comparison
    print("\n" + "="*60)
    print("SURGICAL LAYER 0 COMPARISON")
    print("="*60)
    
    # Map TRT output names (from modeling.py) to PyTorch golden file names
    mapping = [
        ("embedding_out", "embedding_out.npy"),
        ("layer_0_norm_1", "layer_0_norm_1.npy"),
        ("layer_0_attn_out", "layer_0_attn_out.npy"),
        ("layer_0_post_sa", "layer_0_post_sa.npy"),
        ("layer_0_cross_out", "layer_0_cross_out.npy"),
        ("layer_0_norm_ff", "layer_0_norm_ff.npy"),
        ("layer_0_mlp_out", "layer_0_mlp_out.npy")
    ]
    
    for trt_name, pt_file in mapping:
        if trt_name in outputs:
            # Move to float32 before numpy since numpy doesn't support bfloat16 natively
            val = outputs[trt_name].to(torch.float32).cpu().numpy()
            compare(trt_name, val, f"{golden_dir}/{pt_file}")
        else:
            print(f"Missing TRT output: {trt_name} (Check if model.mark_output was called)")

if __name__ == "__main__":
    verify()
