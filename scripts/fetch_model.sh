#!/usr/bin/env bash
set -e

MODEL_DIR="shared/models"
MODEL_FILE="$MODEL_DIR/Qwen3.5-0.8B.Q4_K_M.gguf"
REPO="unsloth/Qwen3.5-0.8B-GGUF"
MODEL_NAME="Qwen3.5-0.8B.Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "✅ Model already exists at $MODEL_FILE, skipping download."
else
    echo "⬇️ Downloading $MODEL_NAME from Hugging Face..."
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$REPO" "$MODEL_NAME" \
        --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
    echo "✅ Download complete."
fi
