#!/bin/bash
# T5Gemma-TTS FastAPI Server using TensorRT-LLM base image
# This uses the same TRT version as the engine builder to avoid version mismatches

PWD=$(pwd)

echo "Starting T5Gemma-TTS FastAPI server..."
echo "NOTE: This uses the TensorRT-LLM base image for TRT version compatibility"
echo ""
echo "API will be available at:"
echo "  - POST http://localhost:8000/synthesize"
echo "  - GET  http://localhost:8000/health"
echo ""

docker run --gpus all --rm --shm-size=4g \
  -p 8000:8000 \
  -v ${PWD}:/app \
  -w /app \
  tensorrt_llm_implementation-builder \
  python3 scripts/serve_tts.py
