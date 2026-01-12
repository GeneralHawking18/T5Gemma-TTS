#!/bin/bash
# Use current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Container name
CONTAINER_NAME="t5gemma_triton"

# Stop existing container
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Stopping existing container..."
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

# Start Triton with model repository and trt_weights mounted
# Using 25.11-trtllm-python-py3 which has:
#   - TensorRT 10.13.3.9 (matching our engines)
#   - PyTorch, Transformers pre-installed
echo "Starting Triton Server..."
docker run --gpus all -d --rm --shm-size=4g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ${SCRIPT_DIR}/triton_model_repository:/models \
  -v ${SCRIPT_DIR}/trt_weights:/workspace/trt_weights \
  -v ${SCRIPT_DIR}:/workspace/T5Gemma-TTS \
  -e HF_HOME=/workspace/T5Gemma-TTS/.cache \
  --name ${CONTAINER_NAME} \
  nvcr.io/nvidia/tritonserver:25.11-trtllm-python-py3 \
  tritonserver --model-repository=/models --strict-model-config=false --log-verbose=1

echo "Container ${CONTAINER_NAME} started."
echo "Logs: docker logs -f ${CONTAINER_NAME}"
