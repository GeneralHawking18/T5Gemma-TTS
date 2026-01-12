# Serving T5Gemma-TTS with TensorRT-LLM

This guide outlines the architecture and steps required to serve the T5Gemma-TTS model using the **TensorRT-LLM** backend for high-performance inference.

## 1. System Architecture

Serving this specific TTS model requires a **Hybrid Architecture** because TensorRT-LLM currently optimizes the **Decoder** (generation phase), while the **Encoder** (context processing) remains in PyTorch or ONNX.

### Data Flow
1.  **Client Request**: Sends text and reference audio (JSON).
2.  **API Server (FastAPI/Triton)**:
    *   **Step 1 (Preprocessing)**: Tokenize text & audio using `AudioTokenizer` (XCodec2) and `AutoTokenizer`.
    *   **Step 2 (Encoder)**: Run the T5 Encoder (PyTorch/ONNX) to get `encoder_hidden_states`.
    *   **Step 3 (Decoder - Optimized)**: Pass hidden states to **TensorRT-LLM Engine** to generate audio tokens autoregressively.
    *   **Step 4 (Vocoder)**: Decode generated tokens back to waveform using XCodec2.
3.  **Response**: Returns WAV audio or base64 string.

---

## 2. Prerequisites & Preparation

Before serving, you must build the TensorRT engine.

### Step A: Convert Weights
Convert the PyTorch model weights to TensorRT-LLM format (`.npz`).

```bash
cd tensorrt_llm_implementation
# Requires ~16GB RAM
python3 convert_weights.py \
    --model_name "Aratako/T5Gemma-TTS-2b-2b" \
    --output_dir ../trt_weights
```

### Step B: Build TensorRT Engine
Compile the weights into an optimized engine for your specific GPU.

```bash
# Using the provided Docker setup
docker compose run --rm builder python3 build.py \
    --weights_path ../trt_weights/weights.npz \
    --output_dir ../engine_output
```
**Output**: `engine_output/t5gemma_decoder.engine`

---

## 3. Serving Implementation Strategies

There are two recommended ways to serve this engine.

### Option A: Python-based Server (FastAPI)
*Best for flexibility and custom logic (e.g., handling the Hybrid Encoder/Decoder split).*

**Concept:**
Create a Python service that loads the TRT Engine into memory using the `tensorrt_llm` Python bindings (similar to `run_inference.py`) and wraps it with a web framework.

**Required Code Structure:**
1.  **Model Loader Class**:
    *   Initialize `tensorrt_llm.runtime.ModelRunner` with the `.engine` path.
    *   Load the PyTorch Encoder side-by-side.
2.  **Inference Handler**:
    *   Input: `text`, `ref_audio`
    *   Logic:
        ```python
        # 1. PyTorch Encoder
        inputs = tokenizer(text, ...)
        encoder_out = pytorch_encoder(inputs)
        
        # 2. TRT Decoder
        # Prepare inputs (contiguous GPU tensors)
        trt_inputs = {
            "encoder_hidden_states": encoder_out.contiguous(),
            "input_ids": start_tokens,
            ...
        }
        # Execute
        output_tokens = trt_runner.generate(trt_inputs)
        ```
3.  **API Endpoint**:
    *   Expose via `FastAPI`.

### Option B: NVIDIA Triton Inference Server
*Best for production, scaling, and standardized metrics.*

**Concept:**
Use Triton's **Ensemble** pattern.
1.  **Model A (Python Backend)**: Preprocessing & PyTorch Encoder.
2.  **Model B (TensorRT-LLM Backend)**: The `t5gemma_decoder.engine`.
3.  **Model C (Python Backend)**: Vocoder (Token -> Audio).

**Configuration:**
You need to create a `model_repository` folder:
```
model_repository/
  ├── preprocessing/ (config.pbtxt, model.py)
  ├── t5_encoder/    (config.pbtxt, model.pt)
  ├── trt_decoder/   (config.pbtxt, t5gemma_decoder.engine)
  └── ensemble_pipeline/ (config.pbtxt)
```

---

## 4. Current Limitations & TODOs

To achieve a fully functional server based on the current repository state:

1.  **RoPE Implementation**: The current TRT implementation has `apply_rope` disabled/commented out in `modeling.py` for debugging. This must be re-enabled and verified against PyTorch outputs for correct audio generation.
2.  **Integration**: The `run_inference.py` script needs to be merged with the logic from `api.py` to replace the PyTorch Decoder calls with TRT Engine calls.
3.  **Dynamic Shapes**: Ensure the engine is built with support for dynamic batch sizes and sequence lengths to handle varying text inputs.

## 5. Quick Start (Testing the Engine)

To verify the serving capability of the engine (layer-0 test):

```bash
cd tensorrt_llm_implementation
docker compose run --rm builder python3 run_inference.py \
    --engine_dir ../engine_output
```

```