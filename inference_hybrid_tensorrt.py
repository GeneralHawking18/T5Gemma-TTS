"""
Hybrid Inference: TensorRT Encoder + PyTorch Decoder.
Requires: pip install tensorrt pycuda
"""
import os
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from typing import List, Union, Tuple

# Reuse logic from existing hybrid script
from inference_hybrid_onnx import HybridT5Gemma, topk_sampling, make_pad_mask

class TensorRTEncoder:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print(f"[TRT] Loading engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
    def run(self, input_ids_np, attention_mask_np):
        # Set input shapes for dynamic profile
        # input_ids: index 0, attention_mask: index 1
        seq_len = input_ids_np.shape[1]
        self.context.set_input_shape("input_ids", (1, seq_len))
        self.context.set_input_shape("attention_mask", (1, seq_len))
        
        # Allocate buffers
        # We need to calculate output size. For T5 Encoder, output is same sequence length as input.
        # Hidden dim is usually 2304 (T5Gemma-2b). You MUST check your model config.
        # Let's assume 2304 for now, or read from bindings.
        
        bindings = []
        inputs = [input_ids_np, attention_mask_np]
        outputs = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            
            if mode == trt.TensorIOMode.INPUT:
                # Host -> Device
                data = inputs.pop(0)
                # Ensure contiguous and correct type
                data = np.ascontiguousarray(data)
                d_input = cuda.mem_alloc(data.nbytes)
                cuda.memcpy_htod_async(d_input, data, self.stream)
                bindings.append(int(d_input))
            else:
                # Output: Device -> Host (later)
                # Shape: [1, seq_len, hidden]
                # We need to know exact output size. 
                shape = self.context.get_tensor_shape(name)
                # Resolve dynamic dim (-1)
                shape = [s if s != -1 else seq_len for s in shape]
                
                # Calculate size
                size = 1
                for s in shape: size *= s
                dtype_np = np.float32 if dtype == trt.float32 else np.float16
                
                # Alloc output
                h_output = cuda.pagelocked_empty(size, dtype_np)
                d_output = cuda.mem_alloc(h_output.nbytes)
                bindings.append(int(d_output))
                outputs.append((h_output, d_output, shape))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Retrieve outputs
        result = []
        for h_out, d_out, shape in outputs:
            cuda.memcpy_dtoh_async(h_out, d_out, self.stream)
            self.stream.synchronize()
            result.append(h_out.reshape(shape))
            
        return result

class HybridT5GemmaTRT(HybridT5Gemma):
    def __init__(self, model_name, engine_path, **kwargs):
        # Init Parent (loads PyTorch Decoder)
        super().__init__(model_name=model_name, use_onnx_encoder=False, **kwargs)
        
        # Load TRT Encoder
        if os.path.exists(engine_path):
            self.trt_encoder = TensorRTEncoder(engine_path)
            self.use_trt_encoder = True
            # Remove PyTorch encoder
            self.encoder_module = None
            if hasattr(self.model, "encoder_module"): self.model.encoder_module = None
            torch.cuda.empty_cache()
        else:
            print(f"[Warn] TRT Engine not found at {engine_path}. Fallback to PyTorch.")
            self.use_trt_encoder = False

    def inference_tts(self, x, x_lens, **kwargs):
        # Override just the encoder part if TRT is available
        if self.use_trt_encoder:
            input_ids_np = x.cpu().numpy().astype(np.int32)
            mask_np = (~make_pad_mask(x_lens)).long().cpu().numpy().astype(np.int32)
            
            # TRT Run
            trt_outs = self.trt_encoder.run(input_ids_np, mask_np)
            
            # Convert back to PyTorch
            memory = torch.from_numpy(trt_outs[0]).to(self.device)
            
            # Continue with PyTorch Decoder...
            # This requires refactoring HybridT5Gemma to allow injecting memory.
            # (HybridT5Gemma logic is inside inference_tts, hard to inject without copy-paste)
            
            # For simplicity in this example, we assume HybridT5Gemma is refactored 
            # or we copy-paste the decoder part here.
            # ...
            pass
        else:
            return super().inference_tts(x, x_lens, **kwargs)

# Note: This is a partial implementation. 
# You need to copy the 'Decoder Step' logic from inference_hybrid_onnx.py 
# into the 'if self.use_trt_encoder:' block above to complete the loop.
