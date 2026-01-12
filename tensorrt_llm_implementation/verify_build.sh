#!/bin/bash
echo "Verifying build..."

# Image name from docker-compose
IMAGE_NAME="tensorrt_llm_implementation-builder"

# Check if engine exists
ENGINE_PATH="../engine_output/t5gemma_decoder_new.engine"
if [ ! -f "$ENGINE_PATH" ]; then
    echo "Engine file $ENGINE_PATH not found! Build might be still running or failed."
    exit 1
fi

echo "Engine found. Running verification..."

# Run save_trt_outputs.py
echo "Running save_trt_outputs.py..."
docker run --rm --gpus all \
    -v $(pwd)/..:/app \
    -w /app/tensorrt_llm_implementation \
    $IMAGE_NAME \
    python3 save_trt_outputs.py

# Run compare_with_pytorch.py
echo "Running compare_with_pytorch.py..."
docker run --rm --gpus all \
    -v $(pwd)/..:/app \
    -w /app/tensorrt_llm_implementation \
    $IMAGE_NAME \
    python3 compare_with_pytorch.py