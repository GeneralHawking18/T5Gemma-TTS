import tensorrt as trt
import torch
import numpy as np
import os
import time
from transformers import AutoTokenizer


ENGINE_PATH="/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/trt_weights/encoder_trt.engine"
TOKENIZER_PATH="/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/tokenizer_local"

class T5GemmaTRTEncoder:  # Đổi tên class cho đúng bản chất
    """TensorRT inference wrapper for T5/Gemma Text Encoder."""

    def __init__(self, engine_path: str, tokenizer_path: str, device: int = 0):
        self.device = device
        torch.cuda.set_device(device)
        self.stream = torch.cuda.Stream()

        # 1. Load Tokenizer
        print(f"🔄 Đang tải tokenizer từ: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 2. Load TensorRT Engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        print(f"🚀 Loading TensorRT engine from {engine_path}...")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        print("✅ Engine loaded successfully!")

    def tokenize(self, text: str):
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        # TensorRT thường yêu cầu int32 hoặc int64 tùy lúc build. 
        # T5 ONNX gốc thường là int64.
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        return input_ids, attention_mask

    def encode(self, text: str) -> np.ndarray:
        """
        Text -> Tokens -> TRT Engine -> Hidden States
        """
        # 1. Tokenize
        input_ids_cpu, attention_mask_cpu = self.tokenize(text)
        
        batch_size, seq_len = input_ids_cpu.shape

        # 2. Set Dynamic Input Shapes (QUAN TRỌNG)
        # Tên 'input_ids' và 'attention_mask' phải khớp với script build engine của bạn
        self.context.set_input_shape('input_ids', input_ids_cpu.shape)
        self.context.set_input_shape('attention_mask', attention_mask_cpu.shape)

        # 3. Allocate GPU Memory cho Input
        input_ids_gpu = torch.from_numpy(input_ids_cpu).cuda(self.device)
        attention_mask_gpu = torch.from_numpy(attention_mask_cpu).cuda(self.device)

        # 4. Tính toán Output Shape và Allocate GPU Output
        # Sau khi set input shape, ta có thể hỏi TRT xem output shape là bao nhiêu
        # Output name thường là 'encoder_hidden_states' hoặc 'last_hidden_state' (check lại script build)
        output_name = 'encoder_hidden_states' 
        
        # Lấy shape output từ context (vì nó phụ thuộc vào seq_len)
        out_shape = tuple(self.context.get_tensor_shape(output_name))
        
        # Create output tensor
        output_gpu = torch.empty(out_shape, dtype=torch.float32, device=f'cuda:{self.device}')

        # 5. Binding (Gán địa chỉ bộ nhớ)
        # set_tensor_address yêu cầu TRT 8.5+
        self.context.set_tensor_address('input_ids', input_ids_gpu.data_ptr())
        self.context.set_tensor_address('attention_mask', attention_mask_gpu.data_ptr())
        self.context.set_tensor_address(output_name, output_gpu.data_ptr())

        # 6. Execute Inference
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # 7. Copy về CPU
        hidden_states = output_gpu.cpu().numpy()
        
        return hidden_states

# === Usage Example ===
if __name__ == "__main__":
    import time

    try:
        encoder = T5GemmaTRTEncoder(ENGINE_PATH, TOKENIZER_PATH)
        start = time.time()

        hidden_states = encoder.encode("Xin chào, hôm nay trời đẹp quá!")
        print("Inference_time: ", time.time() - start)
        output_filename="trt.npz"
        np.savez(
            output_filename, 
            hidden_states=hidden_states
        )
        


        
    except Exception as e:
        print(f"Lỗi: {e}")