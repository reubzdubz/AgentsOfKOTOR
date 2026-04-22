# KOTOR Agents: Multi-Agent Llama.cpp with CrewAI

This project sets up two llama.cpp instances running Qwen3.5 0.8B models, personified as HK-47 and Kreia from Star Wars KOTOR, orchestrated by CrewAI for collaborative tasks.

## Setup

1. **Install Dependencies:**
   - Clone and build llama.cpp in `shared/llama_cpp/`
   - Install Python packages: `pip install crewai langchain-openai`
   - Download Qwen3.5 0.8B QK_M model to `shared/models/`

2. **Download Model:**
   - Find the appropriate GGUF model from Hugging Face (e.g., bartowski/Qwen3.5-0.8B-Instruct-GGUF)
   - Place as `qwen3.5-0.8b-qk_m.gguf` in `shared/models/`

3. **Run Agents:**
   - Start Agent 1: `cd agent1 && chmod +x run.sh && ./run.sh`
   - Start Agent 2: `cd agent2 && chmod +x run.sh && ./run.sh`

4. **Orchestrate:**
   - `cd crew_ai && python orchestrator.py`

## Structure

- `agent1/`: HK-47 instance
- `agent2/`: Kreia instance
- `crew_ai/`: Orchestration logic
- `shared/`: Common resources
- `vision_classifier/`: Scripts for vision data collation and benchmarking

## Notes

- Ensure ports 5000 and 5001 are free.
- Adjust GPU layers and threads in run.sh as needed.
- Logs are stored in each agent's logs/ directory.

## To-do

- Optimize vision data pipeline to collate gameplay footage and log game state into three categories: dialogue, combat and leveling up, which will have their own subagents
- Compare and determine which approach is best to classify game state based on player view in terms of latency and accuracy
- Finetune on scrapped dialogue data to instill character behavior (prompt engineering is rather out of character)