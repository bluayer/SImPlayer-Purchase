from __future__ import annotations

import unittest

from purchase_behavior_simulator.evaluation_summary import (
    continuous_target_metrics,
    is_neutral_fallback,
    prevalence_matched_top_k_metrics,
    summarize_simulation_run,
)


class EvaluationSummaryTest(unittest.TestCase):
    def test_continuous_target_metrics_reports_rank_and_top_overlap(self) -> None:
        metrics = continuous_target_metrics(
            predictions=[0.1, 0.9, 0.2, 0.8],
            targets=[0.2, 0.8, 0.1, 0.9],
            top_fraction=0.5,
        )

        self.assertGreater(metrics["spearman"], 0.0)
        self.assertEqual(metrics["top_count"], 2)
        self.assertEqual(metrics["top_overlap"], 2)
        self.assertEqual(metrics["random_expected_overlap"], 1.0)

    def test_prevalence_matched_top_k_uses_positive_quota(self) -> None:
        metrics = prevalence_matched_top_k_metrics(
            labels=[1, 0, 0, 1],
            scores=[0.9, 0.8, 0.1, 0.7],
        )
        self.assertEqual(metrics["predicted_positive_count"], 2)
        self.assertEqual(metrics["f1"], 0.5)

    def test_detects_neutral_structured_output_fallback(self) -> None:
        self.assertTrue(
            is_neutral_fallback(
                {
                    "components": {"agent_confidence": 0.0},
                    "reasons": ["neutral fallback used"],
                }
            )
        )

    def test_summarizes_success_failure_and_latency(self) -> None:
        rows = [
            {
                "label": 1,
                "oracle_probability": 0.75,
                "latency_seconds": 2.0,
                "result": {
                    "probability": 0.8,
                    "components": {
                        "agent": 0.7,
                        "agent_ranking_score": 0.9,
                        "agent_confidence": 0.5,
                    },
                    "reasons": ["positive"],
                },
            },
            {
                "label": 0,
                "oracle_probability": 0.25,
                "latency_seconds": 4.0,
                "result": {
                    "probability": 0.2,
                    "components": {
                        "agent": 0.5,
                        "agent_confidence": 0.0,
                    },
                    "reasons": ["neutral fallback used"],
                },
            },
            {
                "label": 0,
                "result": None,
                "trace": None,
            },
        ]
        summary = summarize_simulation_run(rows)
        self.assertEqual(summary["successful_cases"], 2)
        self.assertEqual(summary["failed_cases"], 1)
        self.assertEqual(summary["neutral_fallbacks"], 1)
        self.assertEqual(summary["latency_seconds"]["mean"], 3.0)
        self.assertEqual(summary["agent_ranking_score"]["roc_auc"], 1.0)
        self.assertIn(
            "signals",
            summary["inference_isolated_oracle_reference"],
        )

    def test_handles_all_failed_rows(self) -> None:
        summary = summarize_simulation_run(
            [{"label": 0, "result": None, "trace": None}]
        )
        self.assertEqual(summary["successful_cases"], 0)
        self.assertNotIn("sealed_oracle_reference", summary)

    def test_handles_eligibility_short_circuit_without_agent_components(self) -> None:
        summary = summarize_simulation_run(
            [
                {
                    "label": 0,
                    "oracle_probability": 0.01,
                    "latency_seconds": 0.01,
                    "result": {
                        "probability": 0.0,
                        "components": {"base": 0.0},
                        "reasons": ["already owned"],
                    },
                    "trace": {
                        "events": [
                            {
                                "stage": "eligibility_short_circuit",
                            }
                        ]
                    },
                }
            ]
        )

        self.assertEqual(summary["successful_cases"], 1)
        self.assertEqual(summary["agent_likelihood"]["mean_prediction"], 0.0)
        self.assertEqual(
            summary["observable_trace"]["eligibility_short_circuits"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
