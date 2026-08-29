from __future__ import annotations

import httpx
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

from llm_router.classification import SemanticQueryClassifier
from llm_router.defaults import DEFAULT_CLASSIFIER_MODEL
from llm_router.models import ProviderProfile, RoutingPolicy
from llm_router.provider_engine import ProviderQueryEngine
from llm_router.selector import SemanticProviderSelector


def build_query_engine_tools(
    http_client: httpx.Client,
    providers: tuple[ProviderProfile, ...],
    api_key: str,
) -> list[QueryEngineTool]:
    """Build one LlamaIndex query-engine tool per provider."""
    tools: list[QueryEngineTool] = []

    for provider in providers:
        llm = OpenAI(
            model=provider.model,
            api_key=api_key,
            http_client=http_client,
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
    verbose: bool = False,
) -> tuple[RouterQueryEngine, SemanticProviderSelector]:
    """Build the classifier, selector, provider tools, and router."""
    classifier_llm = OpenAI(
        model=classifier_model,
        api_key=api_key,
        http_client=http_client,
    )
    classifier = SemanticQueryClassifier(classifier_llm)
    selector = SemanticProviderSelector(
        classifier=classifier,
        providers=providers,
        policy=policy,
    )
    tools = build_query_engine_tools(
        http_client=http_client,
        providers=providers,
        api_key=api_key,
    )
    router = RouterQueryEngine(
        selector=selector,
        query_engine_tools=tools,
        llm=classifier_llm,
        verbose=verbose,
    )

    return router, selector
