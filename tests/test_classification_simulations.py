from __future__ import annotations

import unittest

from main import select_test_case


class ClassificationSimulationTests(unittest.TestCase):
    def test_returns_expected_query_profiles(self) -> None:
        expected = {
            1: (0.05, 0.95, 0.95),
            2: (0.50, 0.60, 0.60),
            3: (0.95, 0.10, 0.10),
            4: (0.10, 0.90, 0.80),
        }

        for test_case, values in expected.items():
            with self.subTest(test_case=test_case):
                _, profile = select_test_case(test_case)
                self.assertEqual(
                    (
                        profile.reasoning_depth,
                        profile.latency_sensitivity,
                        profile.cost_sensitivity,
                    ),
                    values,
                )


if __name__ == "__main__":
    unittest.main()
