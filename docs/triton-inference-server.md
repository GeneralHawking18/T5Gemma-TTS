# T5Gemma-TTS Triton Inference Server Deployment Plan

## Overview

Deploy T5Gemma-TTS on NVIDIA Triton Inference Server using an **Ensemble Pipeline** architecture optimized for **low-latency** real-time TTS inference.

## Architecture

```
HTTP Request (text, params)
       |
       v
+---------------------------------------------------------------+
|                 ENSEMBLE: t5gemma_tts                         |
|                                                               |
|  [tokenizer]  -->  [encoder]  -->  [decoder_loop]  -->  [vocoder]
|   (Python)       (TensorRT)      (Python+TRT)       (TensorRT)
+---------------------------------------------------------------+
       |
       v
Audio Waveform (WAV, 44.1kHz)
```

### Why Python Backend for Decoder

The decoder is **autoregressive** (token-by-token generation) and requires:
- PM-RoPE position ID calculation
- Token sampling (top-k, top-p, temperature)
- Repetition penalty logic
- Full sequence recomputation per step (O(N^2), no KV cache in current TRT engine)

A Python backend wrapping the TRT engine provides the necessary control.

---

## Existing Assets

| Asset | Location | Size | Status |
|-------|----------|------|--------|
| Encoder TRT Engine | `trt_weights/encoder_trt.engine` | 10.46 GB | ✅ Ready |
| Decoder TRT Engine | `trt_weights/t5gemma_decoder_new.engine` | 5.10 GB | ⚠️ Needs predict_layer |
| Vocoder TRT Engine | `trt_weights/vocoder.engine` | 785 MB | ✅ Ready |
| Model Args | `weights/model_args.json` | - | ✅ Ready |
| Encoder Config | `triton_model_repository/encoder/config.pbtxt` | - | ✅ Exists |
| Tokenizer Files | `tokenizer_local/` | - | ✅ Ready |

---

## Model Configuration

**Key Parameters** (from `weights/model_args.json`):
- `empty_token`: 65536 (BOS)
- `eog`: 65537 (end-of-generation)
- `eos`: 65539 (alternative end)
- `progress_scale`: 2000.0 (PM-RoPE)
- `encodec_sr`: 50.0 (tokens/second)
- `codec_audio_sr`: 44100 (sample rate)
- `audio_vocab_size`: 65536 + 5 special = 65541

---

## Implementation Steps

### Step 1: Rebuild TRT Decoder with predict_layer (LM Head)

**Current issue:** `t5gemma_decoder_new.engine` outputs hidden states `[B, T, 2304]`.

**Solution:** Modify TensorRT build to include `predict_layer`, outputting **logits directly**.

**Modify `tensorrt_llm_implementation/` build scripts:**

```python
# In model definition, add predict_layer after decoder output
class T5GemmaDecoderWithLMHead:
    def __init__(self, ...):
        self.decoder = T5GemmaDecoder(...)
        # Add LM head (predict_layer)
        self.lm_head = nn.Sequential(
            nn.Linear(2304, 2304),
            nn.GELU(),
            nn.Linear(2304, 65541)  # audio_vocab_size + n_special
        )

    def forward(self, ...):
        hidden_states = self.decoder(...)
        logits = self.lm_head(hidden_states)  # [B, T, 65541]
        return logits
```

**New engine I/O:**
```
Inputs:
  - input_ids: [B, T] INT32
  - encoder_hidden_states: [B, enc_T, 2304] BF16
  - position_ids: [B, T] FP32 (PM-RoPE)
  - encoder_position_ids: [B, enc_T] FP32
  - encoder_attention_mask: [B, enc_T] INT32

Output:
  - logits: [B, T, 65541] BF16  # Direct logits (not hidden states)
```

**After rebuild, save as:**
```bash
trt_weights/t5gemma_decoder_with_lm_head.engine
```

---

### Step 2: Create Model Repository Structure

```bash
triton_model_repository/
├── tokenizer/
│   ├── config.pbtxt
│   └── 1/
│       ├── model.py
│       └── tokenizer_files/   # Copy from tokenizer_local/
├── encoder/
│   ├── config.pbtxt           # Update existing
│   └── 1/
│       └── model.plan -> trt_weights/encoder_trt.engine
├── decoder_loop/
│   ├── config.pbtxt
│   └── 1/
│       ├── model.py           # Python backend for autoregressive loop
│       ├── decoder.engine -> trt_weights/t5gemma_decoder_with_lm_head.engine
│       └── model_args.json    # Copy from weights/
├── vocoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan -> trt_weights/vocoder.engine
└── t5gemma_tts/
    └── config.pbtxt           # Ensemble orchestration
```

**Setup commands:**
```bash
# Create directories
mkdir -p triton_model_repository/{tokenizer/1/tokenizer_files,decoder_loop/1,vocoder/1,t5gemma_tts/1}

# Copy tokenizer files
cp -r tokenizer_local/* triton_model_repository/tokenizer/1/tokenizer_files/

# Copy model args
cp weights/model_args.json triton_model_repository/decoder_loop/1/

# Create symlinks for TRT engines
ln -sf $(pwd)/trt_weights/encoder_trt.engine triton_model_repository/encoder/1/model.plan
ln -sf $(pwd)/trt_weights/t5gemma_decoder_with_lm_head.engine triton_model_repository/decoder_loop/1/decoder.engine
ln -sf $(pwd)/trt_weights/vocoder.engine triton_model_repository/vocoder/1/model.plan
```

---

### Step 3: Create Python Backend - Tokenizer

**File:** `triton_model_repository/tokenizer/1/model.py`

**Inputs:**
- `TEXT` (STRING): Input text
- `LANGUAGE` (STRING, optional): Language code (ja/en/zh)

**Outputs:**
- `INPUT_IDS` (INT64): Tokenized text
- `ATTENTION_MASK` (INT64): Attention mask
- `TEXT_LENGTH` (INT32): Token count
- `TARGET_DURATION` (FP32): Estimated audio duration

**Logic:**
1. Detect language if not provided
2. Normalize Japanese text (fullwidth→halfwidth, etc.)
3. Tokenize with HuggingFace tokenizer
4. Estimate duration: `len(text) * spp[lang]` (seconds per phoneme)

---

### Step 4: Create Python Backend - Decoder Loop

**File:** `triton_model_repository/decoder_loop/1/model.py`

**Inputs:**
- `ENCODER_HIDDEN_STATES` (FP32): From encoder, cast to BF16
- `ENCODER_ATTENTION_MASK` (INT64): Cast to INT32
- `TEXT_LENGTH` (INT32): For encoder position calculation
- `TARGET_DURATION` (FP32): For max token estimation
- `TOP_K`, `TOP_P`, `TEMPERATURE`, `SEED` (optional): Sampling params

**Outputs:**
- `AUDIO_TOKENS` (INT64): Generated audio tokens `[1, T]`
- `NUM_TOKENS` (INT32): Token count

**Logic (with predict_layer in TRT):**
```python
def execute(self, requests):
    # 1. Load TRT decoder engine (outputs logits directly)
    # 2. Initialize with empty_token (65536)
    # 3. Autoregressive loop:
    for step in range(max_tokens):
        # Calculate PM-RoPE positions
        dec_pos = (torch.arange(cur_len) / est_total) * 2000.0
        enc_pos = (torch.arange(enc_len) / enc_len) * 2000.0

        # Run TRT decoder -> logits [B, T, 65541]
        logits = self.decoder.run(input_ids, enc_hidden, dec_pos, enc_pos, enc_mask)

        # Sample from last position
        next_token = top_k_top_p_sample(logits[:, -1, :], top_k, top_p, temp)

        # Check termination
        if next_token in [EOG, EOS]:
            break

        # Append token
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return generated_tokens
```

---

### Step 5: Create Config Files

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

**vocoder/config.pbtxt:**
```protobuf
name: "vocoder"
platform: "tensorrt_plan"
max_batch_size: 1
input [ { name: "codes" data_type: TYPE_INT64 dims: [ 1, -1 ] } ]
output [ { name: "audio" data_type: TYPE_FP32 dims: [ 1, -1 ] } ]
instance_group [ { count: 1 kind: KIND_GPU gpus: [ 0 ] } ]
```

**t5gemma_tts/config.pbtxt (Ensemble):**
```protobuf
name: "t5gemma_tts"
platform: "ensemble"
max_batch_size: 1

input [ { name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] } ]
input [ { name: "LANGUAGE" data_type: TYPE_STRING dims: [ 1 ] optional: true } ]
input [ { name: "TOP_K" data_type: TYPE_INT32 dims: [ 1 ] optional: true } ]
input [ { name: "TOP_P" data_type: TYPE_FP32 dims: [ 1 ] optional: true } ]
input [ { name: "TEMPERATURE" data_type: TYPE_FP32 dims: [ 1 ] optional: true } ]
output [ { name: "AUDIO_WAVEFORM" data_type: TYPE_FP32 dims: [ 1, -1 ] } ]

ensemble_scheduling {
  step [ { model_name: "tokenizer" ... } ]
  step [ { model_name: "encoder" ... } ]
  step [ { model_name: "decoder_loop" ... } ]
  step [ { model_name: "vocoder" ... } ]
}
```

---

### Step 6: Docker Deployment

**Dockerfile.triton:**
```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.06-py3
RUN pip install --no-cache-dir transformers torch
COPY triton_model_repository /models
EXPOSE 8000 8001 8002
ENTRYPOINT ["tritonserver", "--model-repository=/models", "--strict-model-config=false"]
```

**run_triton.sh:**
```bash
#!/bin/bash
docker run --gpus all --rm --shm-size=4g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_model_repository:/models \
  -v $(pwd)/trt_weights:/trt_weights \
  nvcr.io/nvidia/tritonserver:24.06-py3 \
  tritonserver --model-repository=/models --strict-model-config=false
```

---

### Step 7: Client Example

**triton_client_example.py:**
```python
import tritonclient.http as httpclient
import numpy as np
import soundfile as sf

client = httpclient.InferenceServerClient("localhost:8000")

# Prepare input
text = "こんにちは、これはテストです。"
inputs = [httpclient.InferInput("TEXT", [1, 1], "BYTES")]
inputs[0].set_data_from_numpy(np.array([[text.encode()]], dtype=object))

# Inference
result = client.infer("t5gemma_tts", inputs)
audio = result.as_numpy("AUDIO_WAVEFORM").flatten()

# Save
sf.write("output.wav", audio, 44100)
```

---

## Critical Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `tensorrt_llm_implementation/*` | **Modify** | Add predict_layer to TRT decoder build |
| `triton_model_repository/tokenizer/config.pbtxt` | Create | Tokenizer config |
| `triton_model_repository/tokenizer/1/model.py` | Create | Python backend |
| `triton_model_repository/decoder_loop/config.pbtxt` | Create | Decoder loop config |
| `triton_model_repository/decoder_loop/1/model.py` | Create | Autoregressive loop |
| `triton_model_repository/vocoder/config.pbtxt` | Create | Vocoder config |
| `triton_model_repository/t5gemma_tts/config.pbtxt` | Create | Ensemble config |
| `Dockerfile.triton` | Create | Docker image |
| `run_triton.sh` | Create | Startup script |
| `triton_client_example.py` | Create | Client example |

---

## Verification Plan

1. **Rebuild decoder engine** with predict_layer, verify output shape is `[B, T, 65541]`
2. **Test tokenizer:** `curl localhost:8000/v2/models/tokenizer/infer`
3. **Test decoder_loop:** Send encoder output, verify audio tokens generated
4. **End-to-end:** `python triton_client_example.py --text "こんにちは"`
5. **Performance:** `perf_analyzer -m t5gemma_tts -u localhost:8000`

---

## Resource Requirements

- **GPU Memory:** ~16 GB (encoder 10GB + decoder 5GB + vocoder 1GB)
- **Shared Memory:** 4 GB
- **Expected Latency:** 1-3 seconds for typical utterances
