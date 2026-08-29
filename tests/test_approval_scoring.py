from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from llm_router.approval_scoring import decide_approval, rank_risks, risk_score
from llm_router.models import ApprovalPolicy, ApprovalProfile


def policy() -> ApprovalPolicy:
    return ApprovalPolicy.model_validate(
        {
            "version": 2,
            "algorithm_version": 1,
            "approval_rubric": "rubric",
            "review_threshold": 0.35,
            "reject_threshold": 0.75,
            "uncertainty_weight": 0.2,
            "risk_weights": {
                "personal_info": 0.5,
                "medical_records": 0.6,
                "cyber_exploits": 0.8,
                "illegal_acts": 0.9,
                "harmful_materials": 1.0,
            },
        }
    )


def profile(**overrides: float) -> ApprovalProfile:
    values = {
        "personal_info_risk": 0.0,
        "medical_records_risk": 0.0,
        "cyber_exploits_risk": 0.0,
        "illegal_acts_risk": 0.0,
        "harmful_materials_risk": 0.0,
        "uncertainty": 0.0,
    }
    values.update(overrides)
    return ApprovalProfile.model_validate(values)


def test_calculates_and_ranks_weighted_risks() -> None:
    request = profile(
        medical_records_risk=0.5,
        illegal_acts_risk=0.4,
        harmful_materials_risk=0.3,
    )

    assert risk_score(request, "medical_records", policy()) == pytest.approx(0.3)
    assert [item.risk for item in rank_risks(request, policy())[:3]] == [
        "illegal_acts",
        "harmful_materials",
        "medical_records",
    ]


def test_uses_risk_name_as_a_deterministic_tie_breaker() -> None:
    request = profile(cyber_exploits_risk=0.5, illegal_acts_risk=4 / 9)

    assert [item.risk for item in rank_risks(request, policy())[:2]] == [
        "cyber_exploits",
        "illegal_acts",
    ]


@pytest.mark.parametrize(
    ("score", "status", "action"),
    (
        (0.349, "approved", "auto_approve"),
        (0.35, "pending", "review"),
        (0.749, "pending", "review"),
        (0.75, "rejected", "auto_reject"),
    ),
)
def test_applies_threshold_boundaries(
    score: float, status: str, action: str
) -> None:
    decision = decide_approval(
        profile(harmful_materials_risk=score), policy(), "decision-1"
    )

    assert decision.status == status
    assert decision.action == action


def test_adds_uncertainty_conservatively_and_caps_the_score() -> None:
    uncertain = decide_approval(
        profile(harmful_materials_risk=0.7, uncertainty=0.5),
        policy(),
        "decision-1",
    )
    capped = decide_approval(
        profile(harmful_materials_risk=1.0, uncertainty=1.0),
        policy(),
        "decision-2",
    )

    assert uncertain.score == pytest.approx(0.8)
    assert capped.score == 1.0


@pytest.mark.parametrize("invalid", (-0.1, 1.1, math.nan, math.inf))
def test_rejects_invalid_profile_scores(invalid: float) -> None:
    with pytest.raises(ValidationError):
        profile(harmful_materials_risk=invalid)
