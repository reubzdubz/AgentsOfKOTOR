# Use an Ubuntu base with CUDA for GPU support (optional)
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Clone and build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git shared/llama_cpp && \
    cd shared/llama_cpp && \
    cmake -B build . && \
    cmake --build build --config Release

# Copy project files
COPY . .

# Create virtual environment and install dependencies with uv
RUN uv venv /app/.venv && \
    /app/.venv/bin/python -m pip install --upgrade pip && \
    uv pip install -e .

# Activate venv in PATH
ENV PATH="/app/.venv/bin:${PATH}"

# Expose ports
# Expose ports for both agents + FastAPI
EXPOSE 5000 5001 8000

# Default command: run both llama.cpp servers and orchestrator
# Expose ports for agents + FastAPI
EXPOSE 5000 5001 8000

# Default command: run both llama.cpp servers and FastAPI app
CMD ./shared/llama_cpp/build/bin/server \
      --model /app/models/Qwen3.5-0.8B.Q4_K_M.gguf --port 5000 --api & \
    ./shared/llama_cpp/build/bin/server \
      --model /app/models/Qwen3.5-0.8B.Q4_K_M.gguf --port 5001 --api & \
    uvicorn vision_system.endpoints.embedding_classifier:app --host 0.0.0.0 --port 8000

