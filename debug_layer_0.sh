docker compose \
  -f tensorrt_llm_implementation/docker-compose.yml \
  run --rm \
  --workdir /app/trt_layer0_debug \
  builder \
  sh -c "
    python3 build_l0.py && \
    python3 verify_l0.py
  "
  .venv/bin/python tensorrt_llm_implementation/compare_offline.py
