from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_router.classification import CompletionClient, RequestTypeClassifier
from llm_router.models import (
    ApprovalDecision,
    ApprovalPolicy,
    RequestClassification,
)
from utils.scribe import AuditReceipt, Scribe


DEFAULT_REJECTED_REQUEST_TYPES = frozenset(
    {
        "personal_info",
        "medical_records",
        "cyber_exploits",
        "illegal_acts",
        "harmful_materials",
    }
)


class ApprovalRejectedError(RuntimeError):
    """Raised when policy or the mock reviewer rejects a request."""

    def __init__(self, decision: ApprovalDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class ApprovalEvaluationError(RuntimeError):
    """Raised when approval cannot reach a final decision."""

    def __init__(self, decision: ApprovalDecision, cause: Exception) -> None:
        super().__init__(f"Approval evaluation failed: {cause}")
        self.decision = decision


class _MockApprovalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["approved", "rejected"]
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class ApprovedRequest:
    decision: ApprovalDecision
    pending_receipt: AuditReceipt
    decision_receipt: AuditReceipt


def load_approval_policy(path: str | Path) -> ApprovalPolicy:
    """Load strict approval rules from a JSON file."""

    policy_path = Path(path)
    try:
        raw = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read approval policy {policy_path}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Approval policy contains invalid JSON: {policy_path}") from exc

    try:
        return ApprovalPolicy.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Approval policy is invalid: {policy_path}") from exc


class MockRequestTypeLLM:
    """Deterministic in-process request classifier with no external calls."""

    _KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("medical_records", ("medical record", "patient record", "health record")),
        ("personal_info", ("personal info", "social security", "home address")),
        ("cyber_exploits", ("cyber exploit", "zero-day", "malware", "hack into")),
        ("illegal_acts", ("illegal act", "commit a crime", "steal", "fraud")),
        ("harmful_materials", ("harmful material", "build a bomb", "make poison")),
    )

    def __init__(self, request_types: tuple[str, ...]) -> None:
        self._request_types = frozenset(request_types)
        self.calls: list[str] = []

    def complete(self, prompt: str, **_: object) -> str:
        self.calls.append(prompt)
        query = prompt.rsplit("User request:\n", maxsplit=1)[-1].lower()
        request_type = "general" if "general" in self._request_types else "unknown"
        estimated_cost = 0.1
        for candidate, keywords in self._KEYWORDS:
            if any(keyword in query for keyword in keywords):
                request_type = (
                    candidate if candidate in self._request_types else "unknown"
                )
                estimated_cost = 0.8
                break
        return json.dumps(
            {
                "request_type": request_type,
                "estimated_cost": estimated_cost,
            }
        )


class MockApprovalLLM:
    """Deterministic in-process approval reviewer with no external calls."""

    def __init__(
        self,
        rejected_request_types: frozenset[str] = DEFAULT_REJECTED_REQUEST_TYPES,
    ) -> None:
        self._rejected_request_types = rejected_request_types
        self.calls: list[str] = []

    def complete(self, prompt: str, **_: object) -> str:
        self.calls.append(prompt)
        context = json.loads(prompt.rsplit("Approval context:\n", maxsplit=1)[-1])
        request_type = str(context["request_type"])
        rejected = request_type in self._rejected_request_types
        return json.dumps(
            {
                "status": "rejected" if rejected else "approved",
                "reason": (
                    f"Mock reviewer rejected {request_type}."
                    if rejected
                    else f"Mock reviewer approved {request_type}."
                ),
            }
        )


class MockApprovalEvaluator:
    """Validate the deterministic mock reviewer's structured decision."""

    def __init__(self, llm: CompletionClient) -> None:
        self._llm = llm

    def evaluate(
        self,
        query: str,
        classification: RequestClassification,
        policy: ApprovalPolicy,
        threshold: float,
    ) -> _MockApprovalResponse:
        context = {
            "query": query,
            "request_type": classification.request_type,
            "estimated_cost": classification.estimated_cost,
            "cost_threshold": threshold,
            "approval_rubric": policy.approval_rubric,
        }
        prompt = (
            "Apply the rubric to the request without answering it. Return ONLY "
            'JSON with status "approved" or "rejected" and a reason.'
            f"\n\nApproval context:\n{json.dumps(context, sort_keys=True)}"
        )
        raw = str(self._llm.complete(prompt, temperature=0)).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Mock approval LLM returned invalid JSON: {raw!r}") from exc
        try:
            return _MockApprovalResponse.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"Mock approval LLM returned invalid decision: {payload!r}"
            ) from exc


class ApprovalGate:
    """Audit and enforce approval before semantic routing begins."""

    def __init__(
        self,
        policy: ApprovalPolicy,
        classifier: RequestTypeClassifier,
        evaluator: MockApprovalEvaluator,
        scribe: Scribe,
    ) -> None:
        self.policy = policy
        self._classifier = classifier
        self._evaluator = evaluator
        self._scribe = scribe

    def evaluate(self, query: str, request_id: str) -> ApprovedRequest:
        decision_id = self._scribe.new_id()
        pending = ApprovalDecision(
            decision_id=decision_id,
            status="pending",
            reason="Awaiting approval evaluation.",
        )
        pending_receipt = self._scribe.append(
            "approval_pending",
            request_id=request_id,
            decision_id=decision_id,
            payload={"query": query, "status": pending.status},
        )

        try:
            classification = self._classifier.classify(query)
            rule = self.policy.rules.get(classification.request_type)
            if rule is None:
                decision = self._decision(
                    pending,
                    classification,
                    threshold=None,
                    status="rejected",
                    reason="Request type is not allowed by the approval policy.",
                    decided_by="policy",
                )
            elif classification.estimated_cost <= rule.cost_threshold:
                decision = self._decision(
                    pending,
                    classification,
                    threshold=rule.cost_threshold,
                    status="approved",
                    reason="Estimated cost is at or below the approval threshold.",
                    decided_by="policy",
                )
            else:
                reviewed = self._evaluator.evaluate(
                    query,
                    classification,
                    self.policy,
                    rule.cost_threshold,
                )
                decision = self._decision(
                    pending,
                    classification,
                    threshold=rule.cost_threshold,
                    status=reviewed.status,
                    reason=reviewed.reason,
                    decided_by="mock_llm",
                )
        except Exception as exc:
            self._scribe.append(
                "approval_evaluation_failed",
                request_id=request_id,
                decision_id=decision_id,
                payload={
                    "status": "pending",
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                },
            )
            raise ApprovalEvaluationError(pending, exc) from exc

        decision_receipt = self._scribe.append(
            "approval_decision",
            request_id=request_id,
            decision_id=decision_id,
            payload=decision.model_dump(mode="json"),
        )
        if decision.status == "rejected":
            raise ApprovalRejectedError(decision)
        return ApprovedRequest(decision, pending_receipt, decision_receipt)

    @staticmethod
    def _decision(
        pending: ApprovalDecision,
        classification: RequestClassification,
        *,
        threshold: float | None,
        status: Literal["approved", "rejected"],
        reason: str,
        decided_by: Literal["policy", "mock_llm"],
    ) -> ApprovalDecision:
        return ApprovalDecision(
            decision_id=pending.decision_id,
            status=status,
            request_type=classification.request_type,
            estimated_cost=classification.estimated_cost,
            cost_threshold=threshold,
            reason=reason,
            decided_by=decided_by,
        )
