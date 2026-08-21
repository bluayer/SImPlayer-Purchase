from __future__ import annotations

import unittest

from purchase_behavior_simulator.evaluation import observed_action_labels
from purchase_behavior_simulator.next_action_evaluation import (
    expected_binary_metrics,
    expected_multiclass_metrics,
    evaluate_next_actions,
    evaluate_next_actions_by_slice,
    monte_carlo_binary_f1,
    multiclass_metrics,
)


class NextActionEvaluationTest(unittest.TestCase):
    def test_report_separates_user_actions_from_environment_events(
        self,
    ) -> None:
        report = evaluate_next_actions(
            [
                {
                    "observed_initial_state": "ITEM_EXPOSURE",
                    "observed_action_path": [
                        "CLICK",
                        "START_PURCHASE",
                        "CONFIRM_PURCHASE",
                        "PAYMENT_SUCCESS",
                    ],
                    "action_distributions": {
                        "ITEM_EXPOSURE": {
                            "CLICK": 1.0,
                            "SKIP": 0.0,
                            "EXIT": 0.0,
                            "PURCHASE_NOW": 0.0,
                        },
                        "ITEM_DETAIL": {
                            "START_PURCHASE": 1.0,
                            "BACK": 0.0,
                            "EXIT": 0.0,
                        },
                        "PURCHASE_CONFIRMATION": {
                            "CONFIRM_PURCHASE": 1.0,
                            "CANCEL": 0.0,
                            "EXIT": 0.0,
                        },
                        "PAYMENT_PROCESSING": {
                            "PAYMENT_SUCCESS": 1.0,
                            "INSUFFICIENT_CURRENCY": 0.0,
                            "PAYMENT_FAILED": 0.0,
                        },
                    },
                }
            ]
        )

        stochastic = report["stochastic_expected"]
        self.assertIn(
            "CONFIRM_PURCHASE",
            stochastic["user_action_count_gaps"][
                "PURCHASE_CONFIRMATION"
            ],
        )
        self.assertNotIn(
            "PAYMENT_SUCCESS",
            stochastic["user_action_count_gaps"]["PAYMENT_PROCESSING"],
        )
        self.assertIn(
            "PAYMENT_SUCCESS",
            stochastic["environment_event_count_gaps"][
                "PAYMENT_PROCESSING"
            ],
        )

    def test_maps_sampled_behavior_to_game_store_actions(self) -> None:
        self.assertEqual(
            observed_action_labels(
                {"surface": "store_home", "clicked": 1, "purchased": 1}
            ),
            ("ITEM_EXPOSURE", "CLICK", "PURCHASE"),
        )
        self.assertEqual(
            observed_action_labels(
                {"surface": "store_home", "clicked": 0, "purchased": 1}
            ),
            ("ITEM_EXPOSURE", "PURCHASE_NOW", None),
        )
        self.assertEqual(
            observed_action_labels(
                {"surface": "checkout", "clicked": 0, "purchased": 0}
            ),
            ("ITEM_DETAIL", "BACK", None),
        )

    def test_multiclass_metrics_report_supported_and_all_action_f1(self) -> None:
        metrics = multiclass_metrics(
            ("CLICK", "SKIP"),
            (
                {"CLICK": 0.8, "SKIP": 0.2, "EXIT": 0.0, "PURCHASE_NOW": 0.0},
                {"CLICK": 0.1, "SKIP": 0.9, "EXIT": 0.0, "PURCHASE_NOW": 0.0},
            ),
            actions=("CLICK", "SKIP", "EXIT", "PURCHASE_NOW"),
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1_supported"], 1.0)
        self.assertEqual(metrics["macro_f1_all_actions"], 0.5)

    def test_trajectory_requires_both_click_and_detail_action(self) -> None:
        report = evaluate_next_actions(
            (
                {
                    "observed_initial_state": "ITEM_EXPOSURE",
                    "observed_next_action": "CLICK",
                    "observed_detail_action": "PURCHASE",
                    "observed_action_path": (
                        "CLICK",
                        "START_PURCHASE",
                        "CONFIRM_PURCHASE",
                        "PAYMENT_SUCCESS",
                    ),
                    "action_distributions": {
                        "ITEM_EXPOSURE": {
                            "CLICK": 0.8,
                            "SKIP": 0.1,
                            "EXIT": 0.05,
                            "PURCHASE_NOW": 0.05,
                        },
                        "ITEM_DETAIL": {
                            "START_PURCHASE": 0.2,
                            "BACK": 0.8,
                        },
                        "PURCHASE_CONFIRMATION": {
                            "CONFIRM_PURCHASE": 1.0,
                            "CANCEL": 0.0,
                        },
                        "PAYMENT_PROCESSING": {
                            "PAYMENT_SUCCESS": 1.0,
                            "INSUFFICIENT_CURRENCY": 0.0,
                            "PAYMENT_FAILED": 0.0,
                        },
                    },
                },
            )
        )

        self.assertEqual(report["exposure"]["accuracy"], 1.0)
        self.assertEqual(report["detail"]["accuracy"], 0.0)
        self.assertEqual(
            report["primary_metrics"]["trajectory_expected_exact_match"],
            0.16,
        )

    def test_expected_purchase_metrics_use_probability_mass(self) -> None:
        metrics = expected_binary_metrics(
            (True, False),
            (0.25, 0.10),
        )

        self.assertEqual(metrics["positives"], 1)
        self.assertAlmostEqual(metrics["recall"], 0.25)
        self.assertAlmostEqual(metrics["precision"], 0.71428571)
        self.assertAlmostEqual(metrics["f1"], 0.37037037)
        self.assertAlmostEqual(metrics["brier"], 0.28625)
        self.assertEqual(metrics["observed_count"], 1)
        self.assertEqual(metrics["expected_count"], 0.35)
        self.assertEqual(metrics["count_gap"], -0.65)
        self.assertEqual(metrics["observed_rate"], 0.5)
        self.assertEqual(metrics["expected_rate"], 0.175)
        self.assertEqual(metrics["rate_gap_pp"], -32.5)

    def test_expected_action_metrics_report_observed_expected_gap(self) -> None:
        metrics = expected_multiclass_metrics(
            ("CLICK", "SKIP"),
            (
                {"CLICK": 0.6, "SKIP": 0.4},
                {"CLICK": 0.2, "SKIP": 0.8},
            ),
            actions=("CLICK", "SKIP"),
        )

        click = metrics["per_action"]["CLICK"]
        self.assertEqual(click["observed_count"], 1)
        self.assertEqual(click["expected_count"], 0.8)
        self.assertEqual(click["count_gap"], -0.2)
        self.assertEqual(click["observed_rate"], 0.5)
        self.assertEqual(click["expected_rate"], 0.4)
        self.assertEqual(click["rate_gap_pp"], -10.0)

    def test_monte_carlo_purchase_f1_is_seeded(self) -> None:
        first = monte_carlo_binary_f1(
            (True, False, True),
            (0.8, 0.1, 0.6),
            simulations=25,
            seed=7,
        )
        second = monte_carlo_binary_f1(
            (True, False, True),
            (0.8, 0.1, 0.6),
            simulations=25,
            seed=7,
        )

        self.assertEqual(first, second)

    def test_reports_need_and_product_type_slices(self) -> None:
        rows = (
            {
                "motivation_segment": "rational",
                "product_type": "item",
                "observed_initial_state": "ITEM_EXPOSURE",
                "observed_next_action": "SKIP",
                "observed_detail_action": None,
                "action_distributions": {
                    "ITEM_EXPOSURE": {
                        "CLICK": 0.1,
                        "SKIP": 0.8,
                        "EXIT": 0.1,
                        "PURCHASE_NOW": 0.0,
                    },
                    "ITEM_DETAIL": {
                        "PURCHASE": 0.1,
                        "BACK": 0.8,
                        "EXIT": 0.1,
                    },
                },
            },
            {
                "motivation_segment": "mixed",
                "product_type": "bundle",
                "observed_initial_state": "ITEM_EXPOSURE",
                "observed_next_action": "CLICK",
                "observed_detail_action": "PURCHASE",
                "action_distributions": {
                    "ITEM_EXPOSURE": {
                        "CLICK": 0.6,
                        "SKIP": 0.2,
                        "EXIT": 0.1,
                        "PURCHASE_NOW": 0.1,
                    },
                    "ITEM_DETAIL": {
                        "PURCHASE": 0.6,
                        "BACK": 0.3,
                        "EXIT": 0.1,
                    },
                },
            },
        )

        report = evaluate_next_actions_by_slice(rows)

        self.assertEqual(report["motivation_segment"]["rational"]["cases"], 1)
        self.assertEqual(report["product_type"]["bundle"]["cases"], 1)


if __name__ == "__main__":
    unittest.main()
