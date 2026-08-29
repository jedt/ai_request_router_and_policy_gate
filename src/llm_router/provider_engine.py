from __future__ import annotations

import time

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.llms.openai import OpenAI

from llm_router.models import ProviderProfile


class ProviderQueryEngine(CustomQueryEngine):
    """Execute a query with one provider and attach provider metadata."""

    def __init__(self, llm: OpenAI, provider: ProviderProfile) -> None:
        super().__init__()
        self._llm = llm
        self._provider = provider

    def custom_query(self, query_str: str) -> Response:
        started = time.perf_counter()
        response = self._llm.complete(query_str, temperature=0)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return Response(
            response=str(response),
            metadata={
                "provider_id": self._provider.id,
                "provider": self._provider.provider,
                "model": self._provider.model,
                "latency_ms": elapsed_ms,
            },
        )
