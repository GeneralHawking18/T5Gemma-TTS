# AGENTS.md

This file provides context and guidelines for agentic coding agents (AI) operating in the **T5Gemma-TTS** repository.

## 1. Project Overview

**T5Gemma-TTS** is a text-to-speech system based on the Encoder-Decoder LLM architecture (T5Gemma).
It supports multilingual TTS (English, Chinese, Japanese), voice cloning, and duration control.
The project includes training, inference (Python & Gradio), and model export (ONNX, TensorRT) capabilities.

## 2. Environment Setup

### 2.1. Dependencies
The project relies on Python 3.10+ and PyTorch (with CUDA support).
Core dependencies are listed in `requirements.txt` and `pyproject.toml`.

- **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  # OR using uv (faster)
  uv pip install -r requirements.txt
  ```

- **GPU Support:**
  Ensure PyTorch is installed with CUDA support (e.g., CUDA 12.1 or 12.4).
  ```bash
  pip install "torch<=2.8.0" torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```

### 2.2. Model Setup
Before running inference, models must be downloaded and cached.
- **Run Setup Script:**
  ```bash
  python setup.py
  ```
  This script downloads the T5Gemma model, tokenizer, and XCodec2 audio codec.

### 2.3. Docker
For consistent environments (especially on Windows), use Docker:
```bash
docker compose up --build
```

## 3. Build & Compile

This project is primarily Python-based, but includes specific build steps for TensorRT optimization.

### 3.1. Build TensorRT Engine
To build the TensorRT engine for optimized inference:
```bash
bash run_build.sh
```
*Under the hood:* This calls `tensorrt_llm_implementation/build.py`.

### 3.2. Linting & Formatting
There are no strict, enforced linter configurations (e.g., `ruff.toml` or `.flake8`) present in the root.
- **Recommendation:** Follow standard PEP 8 guidelines.
- **Tools:** Agents may use `ruff` or `flake8` for sanity checks if installed, but should not reformat the entire codebase without user request.

## 4. Testing & Verification

The project uses standalone Python scripts for testing rather than a centralized test runner like `pytest`.

### 4.1. Run a Single Test
To run a specific test, execute the corresponding Python script.
Common test scripts include:

- **Full Inference Test:**
  ```bash
  python test_full_inference.py
  ```
  Verifies the full TTS pipeline (Text -> Audio).

- **Encoder Wrapper Test:**
  ```bash
  python test_encoder_wrapper.py
  ```
  Verifies that the ONNX-exportable wrapper matches the original PyTorch encoder.

- **Import Verification:**
  ```bash
  python test_imports.py
  ```
  Checks if all necessary libraries can be imported successfully.

- **Logic Simulation:**
  ```bash
  python test_logic_simulation.py
  ```

### 4.2. Run Inference Verification
To verify the TensorRT implementation:
```bash
bash run_test.sh
```
*Under the hood:* Calls `tensorrt_llm_implementation/run_inference.py`.

### 4.3. Verify Outputs
To verify generated audio outputs against golden references (if available) or check consistency:
```bash
bash run_verify.sh
```

## 5. Code Style Guidelines

Agents must strictly adhere to the existing code style to maintain consistency.

### 5.1. Formatting
- **Indentation:** Use **4 spaces** for indentation. Do NOT use tabs.
- **Line Length:** Soft limit of roughly **100-120 characters**.
- **Whitespace:**
    -   Use blank lines to separate functions and logical blocks within functions.
    -   Space after commas in lists/arguments: `func(a, b, c)`.
    -   Space around operators: `x = y + z`.

### 5.2. Imports
Organize imports in the following order:
1.  **Standard Library** (`os`, `sys`, `json`, `typing`, etc.)
2.  **Third-Party Libraries** (`torch`, `numpy`, `transformers`, `gradio`, etc.)
3.  **Local Modules** (`from models import ...`, `from config import ...`)

Example:
```python
import os
import time
from typing import Optional, List

import torch
import numpy as np
from transformers import AutoTokenizer

from config import Config
from models.t5gemma import T5GemmaVoiceModel
```

### 5.3. Naming Conventions
- **Variables & Functions:** `snake_case` (e.g., `load_model`, `target_duration`).
- **Classes:** `PascalCase` (e.g., `T5GemmaVoiceModel`, `AudioTokenizer`).
- **Constants:** `UPPER_CASE` (e.g., `DEFAULT_MODEL_ID`, `SAMPLE_RATE`).
- **Private Members:** Prefix with `_` (e.g., `_download_model`).

### 5.4. Typing
- Use Python **type hints** for function arguments and return values where helpful for clarity.
- Example:
  ```python
  def synthesize(self, text: str, duration: Optional[float] = None) -> np.ndarray:
      ...
  ```

### 5.5. Error Handling
- Use `try-except` blocks for external operations (IO, network, model loading).
- Provide informative error messages.
- Do not suppress exceptions silently unless explicitly intended (and commented).

### 5.6. Documentation
- **Docstrings:** Use docstrings (triple quotes `"""`) for modules, classes, and complex functions.
- **Style:** Google-style or simple descriptive style is acceptable.
- **Comments:** Comment complex logic or "magic numbers". explain *why*, not just *what*.

## 6. Architecture & Key Files

### 6.1. Core Components
- **Encoder:** T5Gemma (Text -> Latent).
- **Decoder:** T5Gemma Decoder (Latent -> Audio Tokens).
- **Vocoder/Codec:** XCodec2 (Audio Tokens -> Waveform).

### 6.2. Directory Structure
- `models/`: Model definitions (PyTorch).
- `tensorrt_llm_implementation/`: TensorRT specific implementation and scripts.
- `examples/`: Training and preprocessing examples.
- `scripts/`: Utility scripts for export and conversion.
- `hf_export/`: Code specifically for HuggingFace model export.

### 6.3. Key Entry Points
- `inference_gradio.py`: The web UI application.
- `inference_commandline_hf.py`: CLI for inference using HF format models.
- `main.py`: Entry point for training.
- `export_onnx.py`: Script to export models to ONNX.

## 7. Rules & Best Practices

1.  **Read Before Write:** Always read the file content before editing to understand the context and existing style.
2.  **No "Magic" Fixes:** If fixing a bug, explain the root cause.
3.  **Verify Changes:** Run the relevant test script (e.g., `test_full_inference.py`) after making changes to core logic.
4.  **Hardware Awareness:** Be aware that this code runs on GPUs. Operations involving `torch` tensors should handle `device` (CPU/CUDA) correctly.
5.  **Paths:** Use absolute paths or robust relative path handling (e.g., `os.path.join`).
6.  **Environment Variables:** Use `.env` files and `python-dotenv` for sensitive or environment-specific config (API keys, paths).
