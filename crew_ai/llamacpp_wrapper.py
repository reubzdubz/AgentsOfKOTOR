"""
Custom LLM class for LLaMA.cpp to be used in crew.ai framework
"""
# llamacpp_wrapper.py
from typing import Any, Dict, List, Optional, Union
import requests
from crewai import BaseLLM


class LlamaCppFlatLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str,
        endpoint: str,
        temperature: Optional[float] = 0.7,
        max_tokens: int = 512,
        timeout: int = 120,
    ):
        super().__init__(model=model, temperature=temperature)
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        prompt = self._format_messages_as_prompt(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Construct full endpoint URL for completions
        endpoint_url = self.endpoint
        if not endpoint_url.endswith('/completions'):
            endpoint_url = endpoint_url.rstrip('/') + '/v1/completions'
        
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["text"].strip()

    def _format_messages_as_prompt(self, messages: List[Dict[str, str]]) -> str:
        system_content = ""
        user_content = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
        # Combine system and user into one prompt
        combined_prompt = f"{system_content}\n\n{user_content}"
        return f"<|im_start|>user\n{combined_prompt}<|im_end|>\n<|im_start|>assistant\n"

    def supports_function_calling(self) -> bool:
        return False
