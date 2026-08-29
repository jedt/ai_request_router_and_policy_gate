from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
from pathlib import Path

import httpx
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

from llm_router.approval import (
    ApprovalGate,
    MockApprovalEvaluator,
    MockApprovalLLM,
    MockRequestTypeLLM,
    load_approval_policy,
)
from llm_router.classification import (
    CompletionClient,
    LLMQueryClassifier,
    RequestTypeClassifier,
)
from llm_router.defaults import (
    DEFAULT_APPROVAL_POLICY_PATH,
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
    sleep: Callable[[float], None] = time.sleep,
    async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    provider_llms: Mapping[str, CompletionProvider] | None = None,
    request_type_llm: CompletionClient | None = None,
    approval_llm: CompletionClient | None = None,
) -> tuple[BaseQueryEngine, LargeModelProviderSelector]:
    configured_llms = provider_llms or {}
    approval_policy = load_approval_policy(DEFAULT_APPROVAL_POLICY_PATH)
    request_types = tuple(approval_policy.rules)
    configured_request_type_llm = request_type_llm or MockRequestTypeLLM(request_types)
    configured_approval_llm = approval_llm or MockApprovalLLM()
    scribe.append(
        "router_configured",
        request_id=scribe.new_id(),
        payload={
            "classifier_model": DEFAULT_CLASSIFIER_MODEL,
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
            "max_retries": DEFAULT_MAX_RETRIES,
            "backoff_base_seconds": DEFAULT_BACKOFF_BASE_SECONDS,
            "approval_policy": approval_policy.model_dump(mode="json"),
            "request_type_adapter": _type_name(configured_request_type_llm),
            "approval_adapter": _type_name(configured_approval_llm),
        },
    )
    classifier_llm = OpenAI(
        model=DEFAULT_CLASSIFIER_MODEL,
        api_key=api_key,
        http_client=http_client,
        max_retries=DEFAULT_MAX_RETRIES,
    )
    classifier = LLMQueryClassifier(classifier_llm)
    selector = LargeModelProviderSelector(
        classifier=classifier,
        providers=providers,
        policy=policy,
        scribe=scribe,
    )
    approval_gate = ApprovalGate(
        policy=approval_policy,
        classifier=RequestTypeClassifier(
            configured_request_type_llm,
            request_types,
        ),
        evaluator=MockApprovalEvaluator(configured_approval_llm),
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
        approval_gate=approval_gate,
        max_retries=DEFAULT_MAX_RETRIES,
        backoff_base_seconds=DEFAULT_BACKOFF_BASE_SECONDS,
        sleep=sleep,
        async_sleep=async_sleep,
    )

    return router, selector


def _type_name(value: object | None) -> str:
    if value is None:
        return "None"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
