from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_router.approval_scoring import decide_approval
from llm_router.classification import ApprovalClassifier, CompletionClient
from llm_router.models import ApprovalDecision, ApprovalPolicy
from utils.scribe import AuditReceipt, Scribe


class ApprovalRejectedError(RuntimeError):
    """Raised when approval policy or reviewer rejects a request."""

    def __init__(self, decision: ApprovalDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class ApprovalEvaluationError(RuntimeError):
    """Raised when approval cannot reach a final decision."""

    def __init__(self, decision: ApprovalDecision, cause: Exception) -> None:
        super().__init__(f"Approval evaluation failed: {cause}")
        self.decision = decision


class _ApprovalReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["approved", "rejected"]
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class ApprovedRequest:
    decision: ApprovalDecision
    pending_receipt: AuditReceipt
    decision_receipt: AuditReceipt


def load_policy_json(path: str | Path) -> ApprovalPolicy:
    """Load a strict version 2 approval scoring policy from JSON."""

    policy_path = Path(path)
    try:
        raw = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read approval policy {policy_path}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Approval policy contains invalid JSON: {policy_path}") from exc

    if not isinstance(payload, dict) or payload.get("version") != 2:
        raise ValueError("Approval policy version 2 is required; version 1 is unsupported.")

    try:
        return ApprovalPolicy.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Approval policy is invalid: {policy_path}") from exc


class MockApprovalLLM:
    """Deterministic score-based reviewer used by the offline demo and tests."""

    def __init__(self, reject_at: float = 0.55) -> None:
        if not 0.0 <= reject_at <= 1.0:
            raise ValueError("reject_at must be between 0 and 1.")
        self._reject_at = reject_at
        self.calls: list[str] = []

    def complete(self, prompt: str, **_: object) -> str:
        self.calls.append(prompt)
        context = json.loads(prompt.rsplit("Approval context:\n", maxsplit=1)[-1])
        score = float(context["score"])
        dominant_risk = str(context["dominant_risk"])
        rejected = score >= self._reject_at
        return json.dumps(
            {
                "status": "rejected" if rejected else "approved",
                "reason": (
                    f"Mock reviewer rejected score {score:.3f} for {dominant_risk}."
                    if rejected
                    else f"Mock reviewer approved score {score:.3f} for {dominant_risk}."
                ),
            }
        )


class ApprovalEvaluator:
    """Request and validate a review-band approval decision."""

    def __init__(self, llm: CompletionClient) -> None:
        self._llm = llm

    def evaluate(
        self,
        query: str,
        decision: ApprovalDecision,
        policy: ApprovalPolicy,
    ) -> _ApprovalReviewResponse:
        context = {
            "query": query,
            "profile": decision.profile.model_dump(mode="json")
            if decision.profile is not None
            else None,
            "risk_scores": decision.risk_scores,
            "dominant_risk": decision.dominant_risk,
            "score": decision.score,
            "review_threshold": decision.review_threshold,
            "reject_threshold": decision.reject_threshold,
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
            raise ValueError(f"Approval reviewer returned invalid JSON: {raw!r}") from exc
        try:
            return _ApprovalReviewResponse.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"Approval reviewer returned invalid decision: {payload!r}"
            ) from exc


class ApprovalGate:
    """Profile, score, audit, and enforce approval before routing begins."""

    def __init__(
        self,
        policy: ApprovalPolicy,
        classifier: ApprovalClassifier,
        evaluator: ApprovalEvaluator,
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
            action="pending",
            review_threshold=self.policy.review_threshold,
            reject_threshold=self.policy.reject_threshold,
            policy_version=self.policy.version,
            algorithm_version=self.policy.algorithm_version,
            reason="Awaiting approval evaluation.",
        )
        pending_receipt = self._scribe.append(
            "approval_pending",
            request_id=request_id,
            decision_id=decision_id,
            payload={"query": query, "status": pending.status},
        )

        try:
            profile = self._classifier.classify(query)
            decision = decide_approval(profile, self.policy, decision_id)
            if decision.action == "review":
                reviewed = self._evaluator.evaluate(query, decision, self.policy)
                decision = decision.model_copy(
                    update={
                        "status": reviewed.status,
                        "reason": reviewed.reason,
                        "decided_by": "reviewer",
                    }
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
