import tensorrt as trt
import os

# 1. Cấu hình Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
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
    print(f"Đang đọc file {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 5. Cấu hình Dynamic Shapes (Cực quan trọng)
    profile = builder.create_optimization_profile()
    # Input name là 'vq_code' với shape [batch, 1, seq_length]
    # set_shape(tên_input, min, opt, max)
    profile.set_shape("vq_code", (1, 1, 50), (1, 1, 200), (4, 1, 1000))
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

# Chạy thôi
build_engine("vocoder.onnx", "vocoder.engine")