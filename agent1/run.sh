#!/bin/bash
# Run script for Agent 1 (HK-47)

# Navigate to llama.cpp directory
cd ../llama.cpp

# Run the server
llama-server \
  -m ../models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --port 5000 \
  --host 0.0.0.0 \
  --n-gpu-layers 0 \
  --ctx-size 2048 \
  --threads 4 \
  --log-file ../agent1/logs/server.log \
  --chat-template-kwargs "{\"enable_thinking\": false}"