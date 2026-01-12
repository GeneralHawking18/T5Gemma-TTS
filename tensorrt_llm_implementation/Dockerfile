# Use NVIDIA TensorRT-LLM release image as base
FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6

WORKDIR /app

# Install runtime dependencies for the API
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    onnxruntime-gpu \
    transformers \
    scipy \
    python-dotenv

# We don't copy files here because we will mount the volume in docker-compose
# This allows for easier development and access to weights in the parent directory.
