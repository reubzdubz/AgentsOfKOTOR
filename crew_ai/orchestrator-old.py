#!/usr/bin/env python3
"""
CrewAI Orchestrator for KOTOR Agents
Connects two llama.cpp instances as HK-47 and Kreia agents.
"""

import os
import yaml
from crewai import Agent, Task, Crew
from llamacpp_wrapper import LlamaCppFlatLLM 

# Load config
with open('crew_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load system prompts
with open('../shared/prompts/hk47_system.txt', 'r') as f:
    hk47_system = f.read().strip()

with open('../shared/prompts/kreia_system.txt', 'r') as f:
    kreia_system = f.read().strip()

# Get endpoints from environment variables or use defaults
hk47_endpoint = os.getenv("HK47_ENDPOINT", "http://localhost:5000")
kreia_endpoint = os.getenv("KREIA_ENDPOINT", "http://localhost:5001")

# Define LLMs for each agent
llm_hk47 = LlamaCppFlatLLM(
    model="openai/Qwen3.5-0.8B-GGUF",
    endpoint=hk47_endpoint,
    api_key="dummy"
)


llm_kreia = LlamaCppFlatLLM(
    model="openai/Qwen3.5-0.8B-GGUF",
    endpoint=kreia_endpoint,
    api_key="dummy"
)

# Define Agents
hk47 = Agent(
    role=config['agents'][0]['role'],
    goal=config['agents'][0]['goal'],
    backstory=hk47_system,
    llm=llm_hk47,
    verbose=True
)

kreia = Agent(
    role=config['agents'][1]['role'],
    goal=config['agents'][1]['goal'],
    backstory=kreia_system,
    llm=llm_kreia,
    verbose=True
)

# Define Tasks
plan_task = Task(
    description=config['tasks'][0]['description'],
    expected_output="A concise plan or response in Kreia's voice that addresses the task.",
    agent=kreia
)

execute_task = Task(
    description=config['tasks'][1]['description'],
    expected_output="A complete response in HK-47's voice that follows the prior context and finishes the task.",
    agent=hk47
)

# Create Crew
crew = Crew(
    agents=[hk47, kreia],
    tasks=[plan_task, execute_task],
    verbose=True
)

if __name__ == "__main__":
    # Run the crew
    result = crew.kickoff()
    print(result)