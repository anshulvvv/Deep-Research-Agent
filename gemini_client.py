import json
import requests
from config import settings, logger

class GeminiClient:
    """Wrapper around Google Gemini Chat Completions API with function-calling support."""
    endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {settings.GEMINI_API_KEY}",
            "Content-Type": "application/json"
        }

    def chat(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        payload = {
            "model": settings.MODEL,
            "messages": [
                {"role": "system",  "content": "You are a helpful assistant."},
                {"role": "user",    "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(self.endpoint, json=payload, headers=self.headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"Gemini request failed: {e}")
            raise

    def call_function(
        self,
        prompt: str,
        functions: list[dict],
        function_name: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict:
        payload = {
            "model": settings.MODEL,
            "messages": [
                {"role": "system",  "content": "You are a helpful assistant."},
                {"role": "user",    "content": prompt}
            ],
            "functions": functions,
            "function_call": {"name": function_name},
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(self.endpoint, json=payload, headers=self.headers, timeout=30)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            args_str = msg["function_call"]["arguments"]
            return json.loads(args_str)
        except requests.RequestException as e:
            logger.error(f"Gemini function-call failed: {e}")
            raise
