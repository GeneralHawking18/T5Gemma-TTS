# Triton Inference Server Setup for T5Gemma-TTS

This directory contains the configuration and scripts to host the T5Gemma-TTS models (Encoder and Decoder) on NVIDIA Triton Inference Server.

## 1. Directory Structure

The model repository is located at `triton_model_repository/` and follows the standard Triton layout:

```
triton_model_repository/
├── encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan -> .../trt_weights/encoder_trt.engine
└── decoder/
    ├── config.pbtxt
    └── 1/
        └── model.plan -> .../trt_weights/t5gemma_decoder_new.engine
```

## 2. Prerequisites

- Docker with NVIDIA GPU support.
- `trt_weights/` must contain the built engines (`encoder_trt.engine`, `t5gemma_decoder_new.engine`).

## 3. Running the Server

Run the following command to start the Triton server:

```bash
docker run --gpus all --rm --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

## 4. Client Usage

To query the server, you can use the `tritonclient` library. The logic is similar to `tensorrt_llm_implementation/inference_trt_end2end.py`, but instead of using `TRTEncoder`/`TRTDecoder` classes, you send requests to localhost:8000.

### Example (Conceptual)

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# Encoder
input_ids_data = ... # numpy array int64
inputs = [httpclient.InferInput("input_ids", input_ids_data.shape, "INT64")]
inputs[0].set_data_from_numpy(input_ids_data)
# ... add attention_mask ...

result = client.infer("encoder", inputs)
encoder_hidden = result.as_numpy("encoder_hidden_states")

# Decoder (Loop)
# Call "decoder" model repeatedly with updated input_ids and position_ids
```
