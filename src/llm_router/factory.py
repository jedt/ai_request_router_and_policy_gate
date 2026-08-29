from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict

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
from utils.scribe import Scribe


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
    scribe: Scribe,
    classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    provider_llms: Mapping[str, CompletionProvider] | None = None,
) -> tuple[BaseQueryEngine, LargeModelProviderSelector]:
    configured_llms = provider_llms or {}
    scribe.append(
        "router_configured",
        request_id=scribe.new_id(),
        payload={
            "classifier_model": classifier_model,
            "providers": [
                {
                    **asdict(provider),
                    "adapter_type": _type_name(configured_llms.get(provider.id))
                    if provider.id in configured_llms
                    else "llama_index.llms.openai.OpenAI",
                }
                for provider in providers
            ],
            "policy": asdict(policy),
            "max_retries": max_retries,
            "backoff_base_seconds": backoff_base_seconds,
        },
    )
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
        scribe=scribe,
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
        scribe=scribe,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        sleep=sleep,
        async_sleep=async_sleep,
    )

    return router, selector


def _type_name(value: object | None) -> str:
    if value is None:
        return "None"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
