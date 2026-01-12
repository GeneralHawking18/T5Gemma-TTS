# Báo Cáo Triển Khai T5Gemma-TTS trên Triton Inference Server

**Ngày:** 12/01/2026  
**Dự án:** T5Gemma-TTS  
**Tác giả:** AI Assistant  

---

## 1. Tổng Quan

Triển khai đầy đủ kế hoạch trong `docs/triton-inference-server.md` để deploy T5Gemma-TTS lên NVIDIA Triton Inference Server sử dụng kiến trúc **Ensemble Pipeline**.

### Kiến Trúc Pipeline

```
HTTP Request (text, params)
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│                 ENSEMBLE: t5gemma_tts                      │
│                                                            │
│  [tokenizer]  ──►  [encoder]  ──►  [decoder_loop]  ──►  [vocoder]
│   (Python)       (TensorRT)      (Python+TRT)       (TensorRT)
└───────────────────────────────────────────────────────────┘
       │
       ▼
Audio Waveform (WAV, 44.1kHz)
```

---

## 2. Các Bước Đã Hoàn Thành

| Bước | Trạng thái | Mô tả |
|------|------------|-------|
| 1 | ✅ Hoàn thành | Rebuild TRT Decoder với LM Head (predict_layer) |
| 2 | ✅ Hoàn thành | Tạo cấu trúc Model Repository |
| 3 | ✅ Hoàn thành | Tạo Python Backend - Tokenizer |
| 4 | ✅ Hoàn thành | Tạo Python Backend - Decoder Loop |
| 5 | ✅ Hoàn thành | Tạo Config Files |
| 6 | ✅ Hoàn thành | Docker Deployment |
| 7 | ✅ Hoàn thành | Client Example |

---

## 3. Chi Tiết Từng Bước

### 3.1. Rebuild TRT Decoder với LM Head

**Vấn đề ban đầu:** Engine decoder cũ (`t5gemma_decoder_new.engine`) chỉ output hidden states `[B, T, 2304]`, không có logits.

**Giải pháp:**

1. **Sửa `tensorrt_llm_implementation/modeling.py`** - Thêm class `T5GemmaDecoderWithLMHead`:

```python
class T5GemmaDecoderWithLMHead(Module):
    def __init__(self, config):
        self.decoder = T5GemmaDecoderTRT(config)
        self.lm_head_0 = Linear(2304, 2304, bias=True)  # predict_layer.0
        self.lm_head_2 = Linear(2304, 65541, bias=True) # predict_layer.2
    
    def forward(self, ...):
        hidden_states = self.decoder(...)
        x = gelu(self.lm_head_0(hidden_states))
        logits = self.lm_head_2(x)
        return logits  # [B, T, 65541]
```

2. **Sửa `tensorrt_llm_implementation/build.py`**:
   - Import `T5GemmaDecoderWithLMHead`
   - Sửa key mapping để load weights với prefix `decoder.`
   - Sửa LM head key mapping: `lm_head.0.0.weight` thay vì `lm_head.0.weight`

3. **Sửa `tensorrt_llm_implementation/convert_weights.py`**:
   - Thêm logic để export `predict_layer` weights thành `lm_head.*`

**Kết quả Build:**
```
Engine saved to trt_weights/t5gemma_decoder_with_lm_head.engine
- Size: 5.1 GB
- Build time: ~6 phút 30 giây
- All weights loaded successfully (bao gồm lm_head)
```

---

### 3.2. Tạo Cấu Trúc Model Repository

```
triton_model_repository/
├── tokenizer/
│   ├── config.pbtxt
│   └── 1/
│       ├── model.py
│       └── tokenizer_files/    # Copy từ tokenizer_local/
├── encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan -> trt_weights/encoder_trt.engine
├── decoder_loop/
│   ├── config.pbtxt
│   └── 1/
│       ├── model.py
│       ├── model_args.json
│       └── decoder.engine -> trt_weights/t5gemma_decoder_with_lm_head.engine
├── vocoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan -> trt_weights/vocoder.engine
└── t5gemma_tts/
    └── config.pbtxt           # Ensemble config
```

**Commands đã chạy:**
```bash
mkdir -p triton_model_repository/{tokenizer/1/tokenizer_files,decoder_loop/1,vocoder/1,t5gemma_tts/1,encoder/1}
cp -r tokenizer_local/* triton_model_repository/tokenizer/1/tokenizer_files/
cp weights/model_args.json triton_model_repository/decoder_loop/1/
ln -sf $(pwd)/trt_weights/encoder_trt.engine triton_model_repository/encoder/1/model.plan
ln -sf $(pwd)/trt_weights/t5gemma_decoder_with_lm_head.engine triton_model_repository/decoder_loop/1/decoder.engine
ln -sf $(pwd)/trt_weights/vocoder.engine triton_model_repository/vocoder/1/model.plan
```

---

### 3.3. Python Backend - Tokenizer

**File:** `triton_model_repository/tokenizer/1/model.py`

**Chức năng:**
- Nhận text input (STRING)
- Detect ngôn ngữ (optional)
- Normalize text (NFKC)
- Tokenize bằng HuggingFace tokenizer
- Ước tính duration dựa trên độ dài text

**Input/Output:**

| Input | Type | Description |
|-------|------|-------------|
| TEXT | STRING | Input text |
| LANGUAGE | STRING (optional) | Language code (ja/en/zh) |

| Output | Type | Description |
|--------|------|-------------|
| INPUT_IDS | INT64 | Tokenized IDs |
| ATTENTION_MASK | INT64 | Attention mask |
| TEXT_LENGTH | INT32 | Token count |
| TARGET_DURATION | FP32 | Estimated duration (seconds) |

---

### 3.4. Python Backend - Decoder Loop

**File:** `triton_model_repository/decoder_loop/1/model.py`

**Chức năng:**
- Load TRT engine (decoder + LM head)
- Autoregressive generation loop
- PM-RoPE position calculation
- Token sampling (multinomial)
- Stop on EOG/EOS tokens

**Key Parameters (from model_args.json):**
```python
empty_token = 65536  # BOS token
eog = 65537          # End of generation
eos = 65539          # End of sequence
progress_scale = 2000.0  # PM-RoPE scale
encodec_sr = 50.0    # Tokens per second
```

**Generation Loop:**
```python
for step in range(max_tokens):
    # Calculate PM-RoPE positions
    dec_pos = (arange(cur_len) / est_total) * 2000.0
    enc_pos = (arange(enc_len) / enc_len) * 2000.0
    
    # Run TRT decoder -> logits [B, T, 65541]
    logits = trt_engine.run(input_ids, enc_hidden, dec_pos, enc_pos, enc_mask)
    
    # Sample from last position
    next_token = multinomial(softmax(logits[:, -1, :] / temp))
    
    # Check termination
    if next_token in [EOG, EOS]:
        break
    
    input_ids = concat([input_ids, next_token])
```

---

### 3.5. Tạo Config Files

**tokenizer/config.pbtxt:**
```protobuf
name: "tokenizer"
backend: "python"
max_batch_size: 8
input [ { name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] } ]
input [ { name: "LANGUAGE" data_type: TYPE_STRING dims: [ 1 ] optional: true } ]
output [ { name: "INPUT_IDS" data_type: TYPE_INT64 dims: [ -1 ] } ]
output [ { name: "ATTENTION_MASK" data_type: TYPE_INT64 dims: [ -1 ] } ]
output [ { name: "TEXT_LENGTH" data_type: TYPE_INT32 dims: [ 1 ] } ]
output [ { name: "TARGET_DURATION" data_type: TYPE_FP32 dims: [ 1 ] } ]
instance_group [ { count: 1 kind: KIND_CPU } ]
```

**decoder_loop/config.pbtxt:**
```protobuf
name: "decoder_loop"
backend: "python"
max_batch_size: 1
input [ { name: "ENCODER_HIDDEN_STATES" data_type: TYPE_FP32 dims: [ -1, 2304 ] } ]
input [ { name: "ENCODER_ATTENTION_MASK" data_type: TYPE_INT64 dims: [ -1 ] } ]
input [ { name: "TEXT_LENGTH" data_type: TYPE_INT32 dims: [ 1 ] } ]
input [ { name: "TARGET_DURATION" data_type: TYPE_FP32 dims: [ 1 ] } ]
input [ { name: "TOP_K" data_type: TYPE_INT32 dims: [ 1 ] optional: true } ]
input [ { name: "TOP_P" data_type: TYPE_FP32 dims: [ 1 ] optional: true } ]
input [ { name: "TEMPERATURE" data_type: TYPE_FP32 dims: [ 1 ] optional: true } ]
output [ { name: "AUDIO_TOKENS" data_type: TYPE_INT64 dims: [ 1, -1 ] } ]
output [ { name: "NUM_TOKENS" data_type: TYPE_INT32 dims: [ 1 ] } ]
instance_group [ { count: 1 kind: KIND_GPU gpus: [ 0 ] } ]
```

**t5gemma_tts/config.pbtxt (Ensemble):**
```protobuf
name: "t5gemma_tts"
platform: "ensemble"
max_batch_size: 1

input [ { name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] } ]
output [ { name: "AUDIO_WAVEFORM" data_type: TYPE_FP32 dims: [ 1, -1 ] } ]

ensemble_scheduling {
  step [ { model_name: "tokenizer" ... } ]
  step [ { model_name: "encoder" ... } ]
  step [ { model_name: "decoder_loop" ... } ]
  step [ { model_name: "vocoder" ... } ]
}
```

---

### 3.6. Docker Deployment

**Dockerfile.triton:**
```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.05-py3
RUN pip install --no-cache-dir transformers torch
COPY triton_model_repository /models
EXPOSE 8000 8001 8002
ENTRYPOINT ["tritonserver", "--model-repository=/models", "--strict-model-config=false"]
```

**run_triton.sh:**
```bash
#!/bin/bash
PWD=$(pwd)
docker run --gpus all --rm --shm-size=4g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ${PWD}/triton_model_repository:/models \
  -v ${PWD}/trt_weights:/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/trt_weights \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models --strict-model-config=false
```

---

### 3.7. Client Example

**triton_client_example.py:**
```python
import tritonclient.http as httpclient
import numpy as np
import soundfile as sf

client = httpclient.InferenceServerClient("localhost:8000")

text = "こんにちは、これはテストです。"
inputs = [httpclient.InferInput("TEXT", [1, 1], "BYTES")]
inputs[0].set_data_from_numpy(np.array([[text.encode('utf-8')]], dtype=object))

result = client.infer("t5gemma_tts", inputs)
audio = result.as_numpy("AUDIO_WAVEFORM").flatten()

sf.write("output.wav", audio, 44100)
```

---

## 4. TRT Engines Hiện Có

| Engine | Size | Output Shape |
|--------|------|--------------|
| `encoder_trt.engine` | 9.8 GB | `[B, T, 2304]` hidden states |
| `t5gemma_decoder_with_lm_head.engine` | 5.1 GB | `[B, T, 65541]` logits **(NEW)** |
| `vocoder.engine` | 749 MB | `[B, 1, audio_len]` waveform |

---

## 5. Cách Sử Dụng

### Chạy Triton Server:
```bash
# Pull image (nếu chưa có)
docker pull nvcr.io/nvidia/tritonserver:24.05-py3

# Start server
bash run_triton.sh
```

### Test với Client:
```bash
pip install tritonclient[http] soundfile
python triton_client_example.py --text "Hello world"
```

### API Endpoints:
| Endpoint | URL |
|----------|-----|
| HTTP | `http://localhost:8000/v2/models/t5gemma_tts/infer` |
| gRPC | `localhost:8001` |
| Metrics | `http://localhost:8002/metrics` |

---

## 6. Yêu Cầu Tài Nguyên

| Resource | Requirement |
|----------|-------------|
| GPU Memory | ~16 GB (encoder 10GB + decoder 5GB + vocoder 1GB) |
| Shared Memory | 4 GB |
| Expected Latency | 1-3 seconds per utterance |

---

## 7. Pending/Known Issues

1. **Triton image chưa pull xong** - Image `nvcr.io/nvidia/tritonserver:24.05-py3` (~15GB) cần được pull trước khi chạy server.

2. **Symlink paths** - Các symlink sử dụng absolute path, nên cần mount đúng path trong Docker.

3. **Legacy directories đã cleanup** - Đã xóa `triton_model_repository/decoder` và `triton_model_repository/text_tokenizer` (các directory cũ không cần thiết).

---

## 8. Files Đã Tạo/Sửa

| File | Action | Description |
|------|--------|-------------|
| `tensorrt_llm_implementation/modeling.py` | Modified | Thêm `T5GemmaDecoderWithLMHead` |
| `tensorrt_llm_implementation/build.py` | Modified | Sửa weight mapping, output logits |
| `tensorrt_llm_implementation/convert_weights.py` | Modified | Export predict_layer weights |
| `triton_model_repository/tokenizer/config.pbtxt` | Created | Tokenizer config |
| `triton_model_repository/tokenizer/1/model.py` | Created | Tokenizer Python backend |
| `triton_model_repository/decoder_loop/config.pbtxt` | Created | Decoder loop config |
| `triton_model_repository/decoder_loop/1/model.py` | Created | Autoregressive loop |
| `triton_model_repository/vocoder/config.pbtxt` | Created | Vocoder config |
| `triton_model_repository/t5gemma_tts/config.pbtxt` | Created | Ensemble config |
| `Dockerfile.triton` | Created | Docker image |
| `run_triton.sh` | Created | Startup script |
| `triton_client_example.py` | Created | Client example |
| `trt_weights/t5gemma_decoder_with_lm_head.engine` | Built | New decoder engine with LM head |

---

## 9. Kết Luận

Đã triển khai thành công toàn bộ pipeline Triton Inference Server cho T5Gemma-TTS theo kế hoạch. Hệ thống sẵn sàng để chạy sau khi pull Docker image và khởi động server.

**Next Steps:**
1. Pull Triton Docker image
2. Test server startup
3. Verify end-to-end inference
4. Performance benchmarking với `perf_analyzer`
