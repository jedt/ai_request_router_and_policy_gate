from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from llm_router.defaults import DEFAULT_CLASSIFIER_MODEL
from llm_router.models import QueryProfile


class MockOpenAIService:
    """Offline HTTP backend for the classifier and provider OpenAI clients."""

    def __init__(self, classification_cases: dict[str, QueryProfile]) -> None:
        self._classification_cases = classification_cases
        self.calls: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        model = str(payload.get("model", ""))
        prompt = self._extract_prompt(payload)
        self.calls.append({"model": model, "prompt": prompt})

        if model == DEFAULT_CLASSIFIER_MODEL:
            query = self._extract_query(prompt)
            profile = self._classification_cases.get(query)
            if profile is None:
                raise AssertionError(f"No mock classification fixture for: {query!r}")
            content = profile.model_dump_json()
        else:
            content = json.dumps(
                {"answer": "Mock provider response.", "model": model}
            )

        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            json={
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25,
                    "total_tokens": 75,
                },
            },
        )

    @staticmethod
    def _extract_prompt(payload: dict[str, Any]) -> str:
        messages = payload.get("messages", [])
        if isinstance(messages, list):
            return "\n".join(
                content
                for message in messages
                if isinstance(message, dict)
                and isinstance((content := message.get("content", "")), str)
            )

        prompt = payload.get("prompt", "")
        return prompt if isinstance(prompt, str) else ""

    @staticmethod
    def _extract_query(prompt: str) -> str:
        marker = "User request:\n"
        if marker not in prompt:
            raise AssertionError("Could not find classifier request marker.")
        return prompt.split(marker, 1)[1].strip()
