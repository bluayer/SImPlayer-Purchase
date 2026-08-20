from __future__ import annotations

import unittest

from purchase_behavior_simulator.evaluation_diagnostics import enrich_predictions


class EvaluationDiagnosticsTest(unittest.TestCase):
    def test_detects_exact_history_purchase_without_using_trace_preview(
        self,
    ) -> None:
        predictions = [
            {
                "case_id": "u1:i1",
                "item_id": "i1",
                "label": 1,
                "analysis_features": {
                    "item_categories": ["currency"],
                    "category_preferences": {"currency": 0.8},
                    "price": 100,
                    "budget_reference": 200,
                },
                "result": {
                    "probability": 0.3,
                    "components": {"agent": 0.6},
                },
            }
        ]
        blind_cases = [{"case_id": "u1:i1", "actor_id": "actor-u1"}]
        bootstrap = [
            {
                "observation": {
                    "user_id": "actor-u1",
                    "events": [
                        {
                            "event_type": "purchase",
                            "item_id": "i1",
                            "categories": ["currency"],
                        }
                    ],
                }
            }
        ]

        enriched = enrich_predictions(predictions, blind_cases, bootstrap)

        diagnostics = enriched[0]["diagnostics"]
        self.assertTrue(diagnostics["exact_history_purchase"])
        self.assertEqual(diagnostics["same_category_purchase_count"], 1)
        self.assertEqual(diagnostics["price_to_budget"], 0.5)


if __name__ == "__main__":
    unittest.main()
