from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llama_index.core.tools.types import ToolMetadata

from llm_router.defaults import DEFAULT_POLICY, DEFAULT_PROVIDERS
from llm_router.models import QueryProfile
from llm_router.selector import LargeModelProviderSelector
from utils.scribe import Scribe


class FakeClassifier:
    def classify(self, query: str) -> QueryProfile:
        return QueryProfile(
            reasoning_depth=0.95,
            latency_sensitivity=0.1,
            cost_sensitivity=0.1,
        )


class LargeModelProviderSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.scribe = Scribe(Path(self.temp_dir.name) / "audit.db")
        self.selector = LargeModelProviderSelector(
            classifier=FakeClassifier(),
            providers=DEFAULT_PROVIDERS,
            policy=DEFAULT_POLICY,
            scribe=self.scribe,
        )

    def test_selects_the_matching_provider_tool(self) -> None:
        choices = [
            ToolMetadata(name=provider.id, description="provider")
            for provider in DEFAULT_PROVIDERS
        ]

        result = self.selector.select(choices, "query")

        self.assertEqual(result.ind, 2)
        self.assertIsNotNone(self.selector.last_decision)
        records = self.scribe.verify()
        self.assertEqual([record.event_type for record in records], ["routing_decision"])
        self.assertEqual(records[0].payload["query"], "query")
        self.assertEqual(records[0].payload["selected_provider_id"], "mock-reasoning")

    def test_rejects_misaligned_provider_tools(self) -> None:
        choices = [ToolMetadata(name="wrong", description="provider")]

        with self.assertRaisesRegex(ValueError, "same IDs and order"):
            self.selector.select(choices, "query")


if __name__ == "__main__":
    unittest.main()
