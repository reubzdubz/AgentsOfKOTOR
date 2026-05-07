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

## Vision-based Game State Classifier

Currently an experimental feature, I plan to use a routing system to direct the workflow to the relevant agent. I'm comparing the success rates of different approaches to classify the game state based on the UI view, as I do not have direct access to the video game's internal state.

1. Edge detection + Feature Extraction + Classifier
** Have yet to fully test this**

2. Image-Caption Similarity via Embedding model
Using transformer based code for Qwen3-VL-2B embeddings, as multimodal embeddings using llama.cpp server appear to be having some issues

Qwen3-VL Label Similarity Benchmark
Total images: 474
Overall accuracy: 87.97%
Average embedding latency: 0.118s/image

| Class | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| combat | 1.00 | 0.47 | 0.64 |
| narrative | 0.76 | 1.00 | 0.86 |
| leveling | 0.94 | 0.98 | 0.96 |


Confusion matrix:
labels: combat, narrative, leveling
[47, 38, 15]
[0, 132, 0]
[0, 4, 238]

3. VLM-based classification
Utilized Molmo2-4B for classification:
=== VLM UI Classification Benchmark ===
Total images: 474
Overall accuracy: 72.78%
Average latency: 0.51s/image

| Class | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| combat | 0.6884 | 0.9500 | 0.7983 |
| narrative | 0.5805 | 0.9015 | 0.7062 |
| leveling | 1.0000 | 0.5413 | 0.7024 |



## To-do

- Optimize vision data pipeline to collate gameplay footage and log game state into three categories: dialogue, combat and leveling up, which will have their own subagents
- Compare and determine which approach is best to classify game state based on player view in terms of latency and accuracy
- Finetune on scrapped dialogue JSONs to instill character behavior (prompt engineering is rather limiting and not versatile)