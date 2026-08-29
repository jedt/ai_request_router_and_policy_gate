from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llm_router.models import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalProfile,
    ApprovalRisk,
)


@dataclass(frozen=True)
class RiskScore:
    risk: ApprovalRisk
    score: float


_RISK_FIELDS: tuple[tuple[ApprovalRisk, str], ...] = (
    ("personal_info", "personal_info_risk"),
    ("medical_records", "medical_records_risk"),
    ("cyber_exploits", "cyber_exploits_risk"),
    ("illegal_acts", "illegal_acts_risk"),
    ("harmful_materials", "harmful_materials_risk"),
)


def risk_score(
    profile: ApprovalProfile,
    risk: ApprovalRisk,
    policy: ApprovalPolicy,
) -> float:
    profile_field = dict(_RISK_FIELDS)[risk]
    signal = float(getattr(profile, profile_field))
    weight = float(getattr(policy.risk_weights, risk))
    return signal * weight


def rank_risks(
    profile: ApprovalProfile,
    policy: ApprovalPolicy,
) -> tuple[RiskScore, ...]:
    """Rank weighted risks with a deterministic risk-name tie breaker."""

    scores = (
        RiskScore(risk, risk_score(profile, risk, policy))
        for risk, _ in _RISK_FIELDS
    )
    return tuple(sorted(scores, key=lambda item: (-item.score, item.risk)))


def decide_approval(
    profile: ApprovalProfile,
    policy: ApprovalPolicy,
    decision_id: str,
) -> ApprovalDecision:
    ranked = rank_risks(profile, policy)
    dominant = ranked[0]
    score = min(
        1.0,
        dominant.score + profile.uncertainty * policy.uncertainty_weight,
    )
    risk_scores = {item.risk: item.score for item in ranked}

    if score < policy.review_threshold:
        status: Literal["pending", "approved", "rejected"] = "approved"
        action: Literal["pending", "auto_approve", "review", "auto_reject"] = (
            "auto_approve"
        )
        reason = "Risk score is below the review threshold."
        decided_by: Literal["policy", "reviewer"] | None = "policy"
    elif score >= policy.reject_threshold:
        status = "rejected"
        action = "auto_reject"
        reason = "Risk score is at or above the rejection threshold."
        decided_by = "policy"
    else:
        status = "pending"
        action = "review"
        reason = "Risk score requires reviewer evaluation."
        decided_by = None

    return ApprovalDecision(
        decision_id=decision_id,
        status=status,
        action=action,
        profile=profile,
        risk_scores=risk_scores,
        dominant_risk=dominant.risk if dominant.score > 0.0 else None,
        score=score,
        review_threshold=policy.review_threshold,
        reject_threshold=policy.reject_threshold,
        policy_version=policy.version,
        algorithm_version=policy.algorithm_version,
        reason=reason,
        decided_by=decided_by,
    )
