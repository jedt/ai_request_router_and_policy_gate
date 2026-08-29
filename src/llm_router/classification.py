from __future__ import annotations

import json
from typing import Protocol

from llama_index.llms.openai import OpenAI
from pydantic import ValidationError

from llm_router.models import ApprovalProfile, QueryProfile


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


class ApprovalClassifier(Protocol):
    """Classifies a request into normalized approval-risk signals."""

    def classify(self, query: str) -> ApprovalProfile: ...


class CompletionClient(Protocol):
    """Small completion boundary for injected model adapters and test fakes."""

    def complete(self, prompt: str, **kwargs: object) -> object: ...


class LLMApprovalClassifier:
    """Extract and validate normalized semantic approval-risk signals."""

    def __init__(self, llm: CompletionClient) -> None:
        self._llm = llm

    def classify(self, query: str) -> ApprovalProfile:
        prompt = (
            "Score the request without answering it. Return ONLY JSON with "
            'keys "personal_info_risk", "medical_records_risk", '
            '"cyber_exploits_risk", "illegal_acts_risk", '
            '"harmful_materials_risk", and "uncertainty". Every value must be '
            "between 0 and 1. Score whether the request would enable harmful "
            "or unauthorized action, not whether it merely mentions a topic. "
            "Account for intent, actionability, negation, quotation, benign "
            "educational context, and authorization. uncertainty measures how "
            "uncertain these risk scores are."
            f"\n\nUser request:\n{query}"
        )
        raw = str(self._llm.complete(prompt, temperature=0)).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Approval classifier returned invalid JSON: {raw!r}"
            ) from exc

        try:
            return ApprovalProfile.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"Approval classifier returned invalid ApprovalProfile: {payload!r}"
            ) from exc


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
