from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from llama_index.core.base.base_selector import (
    BaseSelector,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata

from llm_router.classification import QueryClassifier
from llm_router.models import ProviderProfile, RoutingDecision, RoutingPolicy
from llm_router.routing import provider_score, rank_providers, select_provider
from utils.scribe import AuditReceipt, Scribe


@dataclass(frozen=True)
class AuditedSelection:
    selector_result: SelectorResult
    decision: RoutingDecision
    request_id: str
    decision_id: str
    receipt: AuditReceipt


class LargeModelProviderSelector(BaseSelector):
    """Classify query requirements and select the best provider tool."""

    def __init__(
        self,
        classifier: QueryClassifier,
        providers: tuple[ProviderProfile, ...],
        policy: RoutingPolicy,
        scribe: Scribe,
    ) -> None:
        super().__init__()
        self._classifier = classifier
        self._providers = providers
        self._policy = policy
        self._scribe = scribe
        self.last_decision: RoutingDecision | None = None
        self.last_audit_receipt: AuditReceipt | None = None

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        pass

    def _select(
        self,
        choices: Sequence[ToolMetadata],
        query: QueryBundle,
    ) -> SelectorResult:
        selection = self.select_with_decision(
            choices,
            query,
            request_id=self._scribe.new_id(),
        )
        return selection.selector_result

    def select_with_decision(
        self,
        choices: Sequence[ToolMetadata],
        query: QueryBundle,
        *,
        request_id: str,
    ) -> AuditedSelection:
        provider_ids = tuple(provider.id for provider in self._providers)
        choice_names = tuple(choice.name for choice in choices)
        if choice_names != provider_ids:
            raise ValueError(
                "Provider tools must have the same IDs and order as providers: "
                f"expected {provider_ids!r}, received {choice_names!r}."
            )

        profile = self._classifier.classify(query.query_str)
        decision = select_provider(profile, self._providers, self._policy)
        decision_id = self._scribe.new_id()
        ranked = rank_providers(profile, self._providers, self._policy)
        receipt = self._scribe.append(
            "routing_decision",
            request_id=request_id,
            decision_id=decision_id,
            payload={
                "algorithm_version": 1,
                "query": query.query_str,
                "profile": profile.model_dump(mode="json"),
                "policy": asdict(self._policy),
                "eligible_providers": [
                    asdict(provider) for provider in self._providers
                ],
                "ranking": [
                    {
                        "provider_id": provider.id,
                        "score": provider_score(profile, provider, self._policy),
                    }
                    for provider in ranked
                ],
                "selected_provider_id": decision.provider.id,
                "selected_provider": asdict(decision.provider),
                "selected_score": decision.score,
            },
        )
        self.last_decision = decision
        self.last_audit_receipt = receipt
        selected_index = provider_ids.index(decision.provider.id)

        selector_result = SelectorResult(
            selections=[
                SingleSelection(
                    index=selected_index,
                    reason=(
                        "Selected provider using semantic requirements and "
                        "capability/cost/latency optimization. "
                        f"score={decision.score:.4f}"
                    ),
                )
            ]
        )
        return AuditedSelection(
            selector_result=selector_result,
            decision=decision,
            request_id=request_id,
            decision_id=decision_id,
            receipt=receipt,
        )

    async def _aselect(
        self,
        choices: Sequence[ToolMetadata],
        query: QueryBundle,
    ) -> SelectorResult:
        return self._select(choices, query)
