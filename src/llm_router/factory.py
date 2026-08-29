from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import httpx
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

from llm_router.classification import LLMQueryClassifier
from llm_router.defaults import (
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_MAX_RETRIES,
)
from llm_router.failover import FailoverRouterQueryEngine
from llm_router.models import ProviderProfile, RoutingPolicy
from llm_router.provider_engine import CompletionProvider, ProviderQueryEngine
from llm_router.selector import LargeModelProviderSelector


def build_query_engine_tools(
    http_client: httpx.Client,
    providers: tuple[ProviderProfile, ...],
    api_key: str,
    provider_llms: Mapping[str, CompletionProvider] | None = None,
) -> list[QueryEngineTool]:
    tools: list[QueryEngineTool] = []
    provider_llms = provider_llms or {}

    for provider in providers:
        llm = provider_llms.get(provider.id)
        if llm is None:
            llm = OpenAI(
                model=provider.model,
                api_key=api_key,
                http_client=http_client,
                max_retries=0,
            )
        engine = ProviderQueryEngine(llm=llm, provider=provider)
        tools.append(
            QueryEngineTool.from_defaults(
                query_engine=engine,
                name=provider.id,
                description=(
                    "An LLM provider selected by the semantic routing layer."
                ),
            )
        )

    return tools


def build_router(
    http_client: httpx.Client,
    providers: tuple[ProviderProfile, ...],
    policy: RoutingPolicy,
    api_key: str,
    classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    provider_llms: Mapping[str, CompletionProvider] | None = None,
) -> tuple[BaseQueryEngine, LargeModelProviderSelector]:
    classifier_llm = OpenAI(
        model=classifier_model,
        api_key=api_key,
        http_client=http_client,
        max_retries=max_retries,
    )
    classifier = LLMQueryClassifier(classifier_llm)
    selector = LargeModelProviderSelector(
        classifier=classifier,
        providers=providers,
        policy=policy,
    )
    tools = build_query_engine_tools(
        http_client=http_client,
        providers=providers,
        api_key=api_key,
        provider_llms=provider_llms,
    )
    router = FailoverRouterQueryEngine(
        selector=selector,
        query_engine_tools=tools,
        providers=providers,
        policy=policy,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        sleep=sleep,
    )

    return router, selector
