from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from .errors import DependencyUnavailableError, PipelineError


class TextGenerationBackend(Protocol):
    """Small provider-neutral boundary for future LLM backends."""

    name: str

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


class OpenAICompatibleBackend:
    """Uses the broadly supported /chat/completions contract via stdlib HTTP."""

    name = "openai_compatible_api"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.getenv("LLM_MODEL")
        if not self.api_key or not self.base_url or not self.model:
            raise DependencyUnavailableError(
                "API narration/planning requires LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL environment variables.",
                stage="llm_configuration",
            )

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"]).strip()
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise PipelineError(f"LLM API request failed: {exc}", stage="llm_generation") from exc


class FilePromptBackend:
    """Useful for reproducible experiments: reads a prepared answer from a file."""

    name = "file_prompt"

    def __init__(self, response_path: str | Path):
        self.response_path = Path(response_path)

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.response_path.read_text(encoding="utf-8")
