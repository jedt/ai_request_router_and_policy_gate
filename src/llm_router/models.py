from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApprovalRule(BaseModel):
    """Approval threshold for one configured request type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cost_threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ApprovalPolicy(BaseModel):
    """Validated approval rules loaded from JSON configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    approval_rubric: str = Field(min_length=1)
    unmatched_action: Literal["reject"]
    rules: dict[str, ApprovalRule] = Field(min_length=1)

    @field_validator("approval_rubric")
    @classmethod
    def validate_rubric(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("approval_rubric must not be blank.")
        return value

    @field_validator("rules")
    @classmethod
    def validate_request_types(
        cls, value: dict[str, ApprovalRule]
    ) -> dict[str, ApprovalRule]:
        for request_type in value:
            if not request_type or not request_type.replace("_", "").isalnum():
                raise ValueError(
                    "Request type keys must contain only letters, numbers, "
                    "and underscores."
                )
        return value


class RequestClassification(BaseModel):
    """Request type and normalized cost produced by the mock classifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_type: str = Field(min_length=1)
    estimated_cost: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ApprovalDecision(BaseModel):
    """The current or final state of pre-routing approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str = Field(min_length=1)
    status: Literal["pending", "approved", "rejected"]
    request_type: str | None = None
    estimated_cost: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    cost_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    reason: str = Field(min_length=1)
    decided_by: Literal["policy", "mock_llm"] | None = None


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
