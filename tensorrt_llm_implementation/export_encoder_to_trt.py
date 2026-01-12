
import os
import time
import json
import numpy as np
import onnxruntime as ort
import tensorrt as trt



from transformers import AutoTokenizer

# --- Cấu hình đường dẫn (Giả lập lại các biến bạn đã khai báo) ---
ENCODER_ONNX_PATH = "/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/onnx_models_fp16_fixed/encoder.onnx"
TOKENIZER_PATH = "/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/tokenizer_local" 
ENCODER_TRT_PATH = "/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/trt_weights/encoder_trt.engine"

min_shape = (1, 1)
opt_shape = (1, 64) 
max_shape = (4, 256)


# --- Giả lập class wrapper để code của bạn chạy được (vì bạn dùng self.) ---
class T5GemmaEncoderTokenizer:
    def __init__(self):
        # 1. Load Tokenizer
        print(f"🔄 Đang tải tokenizer từ: {TOKENIZER_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self._init_onnx_model()
        


    def convert_trt_weight(self, onnx_file_path, engine_file_path):
        # 1. Cấu hình Logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # 2. Tạo builder
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # 3. Cho phép dùng FP16 (tăng tốc) - DISABLED do NaN issues
        # if builder.platform_has_fast_fp16:
        #     config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP32 precision (FP16 disabled due to potential NaN issues)")

        # 4. Parse file ONNX
        if not parser.parse_from_file(onnx_file_path):
            print("❌ Lỗi khi parse ONNX:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

        # 5. Cấu hình Dynamic Shapes (Cực quan trọng)
        profile = builder.create_optimization_profile()

        
        profile.set_shape("input_ids", min_shape, opt_shape, max_shape)
        profile.set_shape("attention_mask", min_shape, opt_shape, max_shape)


        config.add_optimization_profile(profile)

        # 6. Xây nhà (Build Engine)
        print("Đang build TensorRT Engine (có thể mất vài phút)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine:
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)
            print(f"✅ Xong! Engine đã lưu tại: {engine_file_path}")
        else:
            print("❌ Build thất bại.")


    def _init_onnx_model(self):
        start = time.time()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        print(f"🔄 Đang khởi tạo ONNX Session từ: {ENCODER_ONNX_PATH}")
        self.encoder_session = ort.InferenceSession(ENCODER_ONNX_PATH, sess_options, providers=["CPUExecutionProvider"])
        print("✅ Encoder Session Loaded!")
        print(f"Init time: {time.time() - start}")


    def tokenize(self, text: str) -> tuple[np.array, np.array]:
        # A. Sơ chế (Tokenize)
        # return_tensors="np" để trả về numpy array luôn, đỡ phải convert
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        input_ids = inputs["input_ids"].astype(np.int64) # ONNX thường yêu cầu int64
        attention_mask = inputs["attention_mask"].astype(np.int64)

        return input_ids, attention_mask


    def inference(self, text: str) -> np.array:
        """
        Hàm chạy encoder: Text -> Tokenize -> ONNX -> Hidden States
        """

        start = time.time()
        input_ids, attention_mask = self.tokenize(text)
    
        # B. Chuẩn bị đầu vào cho ONNX (Input Feed)
        # Tên input phải khớp với lúc export (thường là 'input_ids', 'attention_mask')
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # C. Nấu (Run Inference)
        # None ở tham số đầu tiên nghĩa là lấy tất cả output
        print("🚀 Đang chạy inference encoder...")
        encoder_outputs = self.encoder_session.run(None, ort_inputs)
        
        # Output thường là list, phần tử đầu tiên là last_hidden_state
        hidden_states = encoder_outputs[0]
        print(f"Inference time: {time.time() - start}")
        return hidden_states

# ==========================================
# PHẦN MAIN ĐỂ CHẠY TEST
# ==========================================
if __name__ == "__main__":
    # 1. Khởi tạo
    module = T5GemmaEncoderTokenizer()

    # module.convert_trt_weight(ENCODER_ONNX_PATH, ENCODER_TRT_PATH)


    # # 2. Test thử một câu
    sample_text = "Xin chào, hôm nay trời đẹp quá!"
    
    # 3. Chạy Encoder
    hidden_states = module.inference(sample_text)
    output_filename = "onnx.npz"

    # Lưu 3 thứ: output của onnx, output của trt, và ma trận sai số
    np.savez(
        output_filename, 
        hidden_states=hidden_states
    )


    
        
    

    


