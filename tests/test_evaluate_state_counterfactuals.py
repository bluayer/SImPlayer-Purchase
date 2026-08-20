from __future__ import annotations

import unittest

from scripts.evaluate_state_counterfactuals import (
    select_pairs,
    summarize_pairs,
    summarize_saved_predictions,
    rollout_purchase_probability,
)


class EvaluateStateCounterfactualsTest(unittest.TestCase):
    def test_selects_requested_pairs_per_dimension(self) -> None:
        rows = [
            {"pair_id": "a1", "dimension": "balance"},
            {"pair_id": "a2", "dimension": "balance"},
            {"pair_id": "b1", "dimension": "urgency"},
        ]

        selected = select_pairs(rows, per_dimension=1)

        self.assertEqual(
            [row["pair_id"] for row in selected],
            ["a1", "b1"],
        )

    def test_uses_rollout_probability_when_available(self) -> None:
        self.assertEqual(
            rollout_purchase_probability(
                {
                    "probability": 0.2,
                    "components": {"rollout": 0.1},
                }
            ),
            0.1,
        )
        self.assertEqual(
            rollout_purchase_probability(
                {
                    "probability": 0.0,
                    "components": {"base": 0.0},
                }
            ),
            0.0,
        )

    def test_summarizes_directional_pass_rate(self) -> None:
        summary = summarize_pairs(
            [
                {"dimension": "balance", "passed": True},
                {"dimension": "balance", "passed": False},
                {"dimension": "urgency", "passed": True},
            ]
        )

        self.assertEqual(summary["pairs"], 3)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["dimensions"]["balance"]["pass_rate"], 0.5)

    def test_rescores_saved_pair_predictions(self) -> None:
        rows, summary = summarize_saved_predictions(
            [
                {
                    "dimension": "urgency",
                    "favorable": {
                        "result": {
                            "probability": 0.01,
                            "components": {
                                "raw_fusion": 0.04,
                                "rollout": 0.20,
                            },
                        }
                    },
                    "adverse": {
                        "result": {
                            "probability": 0.02,
                            "components": {
                                "raw_fusion": 0.05,
                                "rollout": 0.05,
                            },
                        }
                    },
                    "passed": False,
                }
            ]
        )

        self.assertTrue(rows[0]["passed"])
        self.assertEqual(summary["passed"], 1)


if __name__ == "__main__":
    unittest.main()
