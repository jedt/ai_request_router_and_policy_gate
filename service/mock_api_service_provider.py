from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

import httpx

from llm_router.defaults import DEFAULT_CLASSIFIER_MODEL
from llm_router.models import QueryProfile


ProviderFailure = Literal["offline", "budget_exhausted"]

class MockOpenAIService:
    """Offline HTTP backend for the classifier and provider OpenAI clients."""

    def __init__(
        self,
        classification_profile: QueryProfile,
        *,
        failing_models: frozenset[str] = frozenset(),
        failure: ProviderFailure = "offline",
    ) -> None:
        self._classification_profile = classification_profile
        self._failing_models = failing_models
        self._failure = failure
        self.calls: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        model = str(payload.get("model", ""))
        prompt = self._extract_prompt(payload)
        self.calls.append({"model": model, "prompt": prompt})

        if model == DEFAULT_CLASSIFIER_MODEL:
            content = self._classification_profile.model_dump_json()
        elif model in self._failing_models:
            return self._failure_response()
        else:
            content = json.dumps(
                {"answer": "Mock provider response.", "model": model}
            )

        return _openai_compatible_response(model, content)

    def _failure_response(self) -> httpx.Response:
        if self._failure == "budget_exhausted":
            return httpx.Response(
                status_code=429,
                headers={"content-type": "application/json"},
                json={
                    "error": {
                        "message": "The OpenAI API budget has been exhausted.",
                        "type": "insufficient_quota",
                        "code": "insufficient_quota",
                    }
                },
            )

        return httpx.Response(
            status_code=503,
            headers={"content-type": "application/json"},
            json={
                "error": {
                    "message": "The OpenAI service is unavailable.",
                    "type": "service_unavailable",
                    "code": "service_unavailable",
                }
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


class MockGeminiService:
    """Offline Gemini backend exposed through the demo's HTTP transport."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        model = str(payload.get("model", ""))
        prompt = MockOpenAIService._extract_prompt(payload)
        self.calls.append({"model": model, "prompt": prompt})
        content = json.dumps(
            {
                "answer": "Philip K. Dick.",
                "model": model,
                "provider": "google",
            }
        )
        return _openai_compatible_response(model, content)


class MockGeminiLLM:
    """Minimal LLM adapter that sends completions to MockGeminiService."""

    def __init__(self, http_client: httpx.Client, model: str) -> None:
        self._http_client = http_client
        self._model = model

    def complete(self, prompt: str, **_: Any) -> str:
        response = self._http_client.post(
            "https://mock-gemini.local/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])


class MockAPIServiceProvider:
    """Dispatch mock HTTP calls to the service matching the requested model."""

    def __init__(
        self,
        openai: MockOpenAIService,
        gemini: MockGeminiService,
    ) -> None:
        self.openai = openai
        self.gemini = gemini

    def handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        model = str(payload.get("model", ""))
        if model.startswith("gemini-"):
            return self.gemini.handle(request)
        return self.openai.handle(request)


def _openai_compatible_response(model: str, content: str) -> httpx.Response:
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
