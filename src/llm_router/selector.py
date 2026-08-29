from __future__ import annotations

from collections.abc import Sequence

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
from llm_router.routing import select_provider


class SemanticProviderSelector(BaseSelector):
    """Classify query requirements and select the best provider tool."""

    def __init__(
        self,
        classifier: QueryClassifier,
        providers: tuple[ProviderProfile, ...],
        policy: RoutingPolicy,
    ) -> None:
        super().__init__()
        self._classifier = classifier
        self._providers = providers
        self._policy = policy
        self.last_decision: RoutingDecision | None = None

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        pass

    def _select(
        self,
        choices: Sequence[ToolMetadata],
        query: QueryBundle,
    ) -> SelectorResult:
        provider_ids = tuple(provider.id for provider in self._providers)
        choice_names = tuple(choice.name for choice in choices)
        if choice_names != provider_ids:
            raise ValueError(
                "Provider tools must have the same IDs and order as providers: "
                f"expected {provider_ids!r}, received {choice_names!r}."
            )

        profile = self._classifier.classify(query.query_str)
        decision = select_provider(profile, self._providers, self._policy)
        self.last_decision = decision
        selected_index = provider_ids.index(decision.provider.id)

        return SelectorResult(
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

    async def _aselect(
        self,
        choices: Sequence[ToolMetadata],
        query: QueryBundle,
    ) -> SelectorResult:
        return self._select(choices, query)
