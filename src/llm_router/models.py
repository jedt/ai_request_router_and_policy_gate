from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


ApprovalRisk = Literal[
    "personal_info",
    "medical_records",
    "cyber_exploits",
    "illegal_acts",
    "harmful_materials",
]


class ApprovalProfile(BaseModel):
    """Normalized semantic risk signals extracted from a request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    personal_info_risk: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    medical_records_risk: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cyber_exploits_risk: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    illegal_acts_risk: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    harmful_materials_risk: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    uncertainty: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ApprovalRiskWeights(BaseModel):
    """Severity weights applied to approval risk signals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    personal_info: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    medical_records: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cyber_exploits: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    illegal_acts: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    harmful_materials: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ApprovalPolicy(BaseModel):
    """Versioned scoring policy for pre-routing approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[2]
    algorithm_version: Literal[1]
    approval_rubric: str = Field(min_length=1)
    review_threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reject_threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    uncertainty_weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    risk_weights: ApprovalRiskWeights

    @field_validator("approval_rubric")
    @classmethod
    def validate_rubric(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("approval_rubric must not be blank.")
        return value

    @field_validator("reject_threshold")
    @classmethod
    def validate_threshold_order(cls, value: float, info: ValidationInfo) -> float:
        review_threshold = info.data.get("review_threshold")
        if review_threshold is not None and value <= review_threshold:
            raise ValueError("reject_threshold must be greater than review_threshold.")
        return value


class ApprovalDecision(BaseModel):
    """The approval outcome and complete scoring inputs that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str = Field(min_length=1)
    status: Literal["pending", "approved", "rejected"]
    action: Literal["pending", "auto_approve", "review", "auto_reject"]
    profile: ApprovalProfile | None = None
    risk_scores: dict[ApprovalRisk, float] = Field(default_factory=dict)
    dominant_risk: ApprovalRisk | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    review_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    reject_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    policy_version: int | None = None
    algorithm_version: int | None = None
    reason: str = Field(min_length=1)
    decided_by: Literal["policy", "reviewer"] | None = None


class QueryProfile(BaseModel):
    """Semantic requirements extracted from a user query."""

    model_config = ConfigDict(extra="forbid")

    reasoning_depth: float = Field(ge=0.0, le=1.0)
    latency_sensitivity: float = Field(ge=0.0, le=1.0)
    cost_sensitivity: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class ProviderProfile:
    """Provider identity and normalized capability scores."""

    id: str
    provider: str
    model: str
    reasoning_score: float
    latency_score: float
    cost_score: float


@dataclass(frozen=True)
class RoutingPolicy:
    """Weights applied when evaluating provider suitability."""

    reasoning_weight: float = 0.50
    latency_weight: float = 0.25
    cost_weight: float = 0.25


@dataclass(frozen=True)
class RoutingDecision:
    """The selected provider and the inputs that led to it."""

    provider: ProviderProfile
    profile: QueryProfile
    eligible_provider_ids: tuple[str, ...]
    score: float
