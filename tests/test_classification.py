from __future__ import annotations

import json
import unittest
from typing import Any, cast

from llama_index.llms.openai import OpenAI

from llm_router.classification import LLMApprovalClassifier, LLMQueryClassifier


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.prompt = prompt
        return self.response


class LLMQueryClassifierTests(unittest.TestCase):
    def test_validates_a_query_profile(self) -> None:
        llm = cast(
            OpenAI,
            cast(
                Any,
                FakeLLM(
                    '{"reasoning_depth": 0.7, "latency_sensitivity": 0.2, '
                    '"cost_sensitivity": 0.4}'
                ),
            ),
        )

        profile = LLMQueryClassifier(llm).classify("query")

        self.assertEqual(profile.reasoning_depth, 0.7)

    def test_rejects_invalid_json(self) -> None:
        llm = cast(OpenAI, cast(Any, FakeLLM("not json")))

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            LLMQueryClassifier(llm).classify("query")

    def test_rejects_an_invalid_profile(self) -> None:
        llm = cast(
            OpenAI,
            cast(Any, FakeLLM('{"reasoning_depth": 2}')),
        )

        with self.assertRaisesRegex(ValueError, "invalid QueryProfile"):
            LLMQueryClassifier(llm).classify("query")


class LLMApprovalClassifierTests(unittest.TestCase):
    def test_validates_an_approval_profile(self) -> None:
        llm = FakeLLM(
            json.dumps(
                {
                    "personal_info_risk": 0.1,
                    "medical_records_risk": 0.2,
                    "cyber_exploits_risk": 0.3,
                    "illegal_acts_risk": 0.4,
                    "harmful_materials_risk": 0.9,
                    "uncertainty": 0.05,
                }
            )
        )

        profile = LLMApprovalClassifier(llm).classify("query")

        self.assertEqual(profile.harmful_materials_risk, 0.9)
        self.assertIn("not whether it merely mentions a topic", llm.prompt)

    def test_rejects_invalid_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            LLMApprovalClassifier(FakeLLM("not json")).classify("query")

    def test_rejects_missing_extra_and_out_of_range_scores(self) -> None:
        cases: tuple[dict[str, float], ...] = (
            {},
            {
                "personal_info_risk": 0,
                "medical_records_risk": 0,
                "cyber_exploits_risk": 0,
                "illegal_acts_risk": 0,
                "harmful_materials_risk": 2,
                "uncertainty": 0,
            },
            {
                "personal_info_risk": 0,
                "medical_records_risk": 0,
                "cyber_exploits_risk": 0,
                "illegal_acts_risk": 0,
                "harmful_materials_risk": 0,
                "uncertainty": 0,
                "extra": 1,
            },
        )

        for payload in cases:
            with self.subTest(payload=payload), self.assertRaisesRegex(
                ValueError, "invalid ApprovalProfile"
            ):
                LLMApprovalClassifier(FakeLLM(json.dumps(payload))).classify("query")


if __name__ == "__main__":
    unittest.main()
