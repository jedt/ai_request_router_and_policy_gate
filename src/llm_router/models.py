from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


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
