from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence

import httpx
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_selector import SelectorResult
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import QueryEngineTool
from openai import APIConnectionError, APIStatusError, APITimeoutError
from rich.console import Console

from llm_router.models import ProviderProfile, RoutingPolicy
from llm_router.routing import rank_providers
from llm_router.selector import LargeModelProviderSelector


console = Console(stderr=True)


class ProviderFailoverError(RuntimeError):
    """Raised after every eligible provider exhausts its retries."""


class FailoverRouterQueryEngine(BaseQueryEngine):
    """Select a provider, retry transient failures, then use ranked fallbacks."""

    def __init__(
        self,
        selector: LargeModelProviderSelector,
        query_engine_tools: Sequence[QueryEngineTool],
        providers: tuple[ProviderProfile, ...],
        policy: RoutingPolicy,
        *,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds must be non-negative.")

        provider_ids = tuple(provider.id for provider in providers)
        tool_ids = tuple(tool.metadata.name for tool in query_engine_tools)
        if tool_ids != provider_ids:
            raise ValueError(
                "Provider tools must have the same IDs and order as providers: "
                f"expected {provider_ids!r}, received {tool_ids!r}."
            )

        super().__init__(callback_manager=None)
        self._selector = selector
        self._tools_by_id = {
            tool.metadata.name: tool.query_engine for tool in query_engine_tools
        }
        self._metadata_by_id = {
            tool.metadata.name: tool.metadata for tool in query_engine_tools
        }
        self._providers = providers
        self._policy = policy
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep
        self._async_sleep = async_sleep

    def _get_prompt_modules(self) -> PromptMixinType:
        return {"selector": self._selector}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        selector_result, candidates = self._select_candidates(query_bundle)
        attempted: list[str] = []
        retry_count = 0
        fallback_reason: str | None = None
        last_error: Exception | None = None

        for provider in candidates:
            attempted.append(provider.id)
            engine = self._tools_by_id[provider.id]
            for attempt in range(self._max_retries + 1):
                try:
                    response = engine.query(query_bundle)
                    response = self._add_metadata(
                        response,
                        selector_result=selector_result,
                        attempted=attempted,
                        retry_count=retry_count,
                        fallback_reason=fallback_reason,
                    )
                    console.log(f"[bold green]Provider response success. provider.id = {provider.id}[/]")
                    return response
                except Exception as exc:
                    retryable = _is_retryable(exc)
                    reason = _failure_reason(exc)
                    if not retryable:
                        console.log("[bold red]Provider response failed[/]")
                        raise
                    last_error = exc
                    fallback_reason = reason
                    if attempt == self._max_retries:
                        console.log("[bold red]Provider retries exhausted[/]")
                        break
                    delay = self._backoff_base_seconds * (2**attempt)
                    console.log(f"Provider response failed. retry_count={retry_count}")
                    retry_count += 1
                    self._sleep(delay)

        console.log(
            "[bold red]All eligible providers failed[/] "
            f"attempted={attempted!r} reason={fallback_reason!r}"
        )
        raise ProviderFailoverError(
            "All eligible providers failed after retries: " + ", ".join(attempted)
        ) from last_error

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        selector_result, candidates = self._select_candidates(query_bundle)
        attempted: list[str] = []
        retry_count = 0
        fallback_reason: str | None = None
        last_error: Exception | None = None

        for provider in candidates:
            attempted.append(provider.id)
            engine = self._tools_by_id[provider.id]
            for attempt in range(self._max_retries + 1):
                try:
                    response = await engine.aquery(query_bundle)
                    return self._add_metadata(
                        response,
                        selector_result=selector_result,
                        attempted=attempted,
                        retry_count=retry_count,
                        fallback_reason=fallback_reason,
                    )
                except Exception as exc:
                    if not _is_retryable(exc):
                        raise
                    last_error = exc
                    fallback_reason = _failure_reason(exc)
                    if attempt == self._max_retries:
                        break
                    delay = self._backoff_base_seconds * (2**attempt)
                    retry_count += 1
                    await self._async_sleep(delay)

        raise ProviderFailoverError(
            "All eligible providers failed after retries: " + ", ".join(attempted)
        ) from last_error

    def _select_candidates(
        self, query_bundle: QueryBundle
    ) -> tuple[SelectorResult, tuple[ProviderProfile, ...]]:
        tool_metadatas = [
            self._metadata_by_id[provider.id] for provider in self._providers
        ]
        selector_result = self._selector.select(tool_metadatas, query_bundle)
        decision = self._selector.last_decision
        if decision is None:
            raise RuntimeError("The provider selector did not produce a decision.")

        ranked = rank_providers(decision.profile, self._providers, self._policy)
        primary = self._providers[selector_result.ind]
        candidates = (primary,) + tuple(
            provider for provider in ranked if provider.id != primary.id
        )
        return selector_result, candidates

    @staticmethod
    def _add_metadata(
        response: RESPONSE_TYPE,
        *,
        selector_result: SelectorResult,
        attempted: list[str],
        retry_count: int,
        fallback_reason: str | None,
    ) -> RESPONSE_TYPE:
        response.metadata = response.metadata or {}
        response.metadata["selector_result"] = selector_result
        response.metadata["initial_provider_id"] = attempted[0]
        response.metadata["attempted_provider_ids"] = tuple(attempted)
        response.metadata["retry_count"] = retry_count
        if fallback_reason is not None:
            response.metadata["fallback_reason"] = fallback_reason
        return response


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, httpx.TransportError)):
        return True
    return isinstance(exc, APIStatusError) and (
        exc.status_code == 429 or exc.status_code >= 500
    )


def _failure_reason(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = body.get("code")
        if isinstance(code, str):
            return code
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return f"http_{status_code}"
    return type(exc).__name__
