from __future__ import annotations

import json
from typing import Protocol

from llama_index.llms.openai import OpenAI
from pydantic import ValidationError

from llm_router.models import QueryProfile, RequestClassification


CLASSIFIER_SYSTEM_PROMPT = """You are a semantic requirement analyzer.
Analyze the user's request and estimate:
- reasoning_depth: How much reasoning is required?
- latency_sensitivity: How important is a fast response?
- cost_sensitivity: How important is minimizing model cost?

Return ONLY JSON matching this schema:
{
  "reasoning_depth": 0.0,
  "latency_sensitivity": 0.0,
  "cost_sensitivity": 0.0
}

Rules:
- Values must be between 0 and 1.
- Do not answer the request.
- Do not choose a provider or model.
- Do not classify by topic or use topic labels.
""".strip()


class QueryClassifier(Protocol):
    """Classifies a natural-language query into routing requirements."""

    def classify(self, query: str) -> QueryProfile: ...


class CompletionClient(Protocol):
    """Small completion boundary implemented by deterministic local mocks."""

    def complete(self, prompt: str, **kwargs: object) -> object: ...


class RequestTypeClassifier:
    """Validate mock-generated request type and normalized cost JSON."""

    def __init__(
        self,
        llm: CompletionClient,
        request_types: tuple[str, ...],
    ) -> None:
        if not request_types:
            raise ValueError("request_types must not be empty.")
        self._llm = llm
        self._request_types = request_types

    def classify(self, query: str) -> RequestClassification:
        allowed = ", ".join((*self._request_types, "unknown"))
        prompt = (
            "Classify the request without answering it. Return ONLY JSON with "
            'keys "request_type" and "estimated_cost". The request type must '
            f"be one of: {allowed}. estimated_cost must be between 0 and 1."
            f"\n\nUser request:\n{query}"
        )
        raw = str(self._llm.complete(prompt, temperature=0)).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Request type classifier returned invalid JSON: {raw!r}"
            ) from exc

        try:
            classification = RequestClassification.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                "Request type classifier returned invalid classification: "
                f"{payload!r}"
            ) from exc

        if classification.request_type not in (*self._request_types, "unknown"):
            raise ValueError(
                "Request type classifier returned unsupported request type: "
                f"{classification.request_type!r}"
            )
        return classification


class LLMQueryClassifier:
    """Extract query requirements with an LLM and validate its JSON output."""

    def __init__(self, llm: OpenAI) -> None:
        self._llm = llm

    def classify(self, query: str) -> QueryProfile:
        prompt = f"{CLASSIFIER_SYSTEM_PROMPT}\n\nUser request:\n{query}"
        raw = str(self._llm.complete(prompt, temperature=0)).strip()

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Classifier returned invalid JSON: {raw!r}") from exc

        try:
            return QueryProfile.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"Classifier returned invalid QueryProfile: {payload!r}"
            ) from exc
