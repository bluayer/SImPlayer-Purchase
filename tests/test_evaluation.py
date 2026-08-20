from __future__ import annotations

import unittest

from purchase_behavior_simulator.evaluation import (
    RATIO_SPECS,
    average_precision,
    binary_metrics,
    expected_calibration_error,
    ratio_membership,
    repeat_purchase_is_plausible,
    roc_auc,
    unique_holdout_candidates,
)


class EvaluationTest(unittest.TestCase):
    def test_ratio_membership_reuses_union_candidates(self) -> None:
        self.assertEqual(set(ratio_membership(0, None)), set(RATIO_SPECS))
        self.assertEqual(ratio_membership(7, None), ("1:1",))
        self.assertEqual(ratio_membership(None, 12), ("1:3", "1:9"))

    def test_unique_candidates_exclude_ever_purchased_items_from_negatives(self) -> None:
        rows = [
            {"item_id": "a", "purchased": 0},
            {"item_id": "a", "purchased": 1},
            {"item_id": "b", "purchased": 0},
            {"item_id": "b", "purchased": 0},
        ]
        positives, negatives = unique_holdout_candidates(rows)
        self.assertEqual([row["item_id"] for row in positives], ["a"])
        self.assertEqual([row["item_id"] for row in negatives], ["b"])

    def test_probability_metrics_reward_correct_ranking(self) -> None:
        labels = [0, 0, 1, 1]
        good = [0.1, 0.2, 0.8, 0.9]
        bad = [0.9, 0.8, 0.2, 0.1]
        good_metrics = binary_metrics(labels, good)
        bad_metrics = binary_metrics(labels, bad)
        self.assertLess(good_metrics["brier"], bad_metrics["brier"])
        self.assertLess(
            good_metrics["probability_mae"],
            bad_metrics["probability_mae"],
        )
        self.assertGreater(good_metrics["roc_auc"], bad_metrics["roc_auc"])
        self.assertEqual(roc_auc(good, labels), 1.0)
        self.assertEqual(average_precision(good, labels), 1.0)
        self.assertEqual(good_metrics["expected_positive_count"], 2.0)
        self.assertEqual(good_metrics["positive_count_error"], 0.0)
        self.assertEqual(good_metrics["positive_mean_prediction"], 0.85)
        self.assertEqual(good_metrics["negative_mean_prediction"], 0.15)
        self.assertEqual(good_metrics["mean_prediction_separation_pp"], 70.0)

    def test_ece_bins_include_every_row(self) -> None:
        labels = [0, 1, 0, 1, 1]
        probabilities = [0.01, 0.02, 0.3, 0.7, 0.95]
        _, table = expected_calibration_error(labels, probabilities, bins=3)
        self.assertEqual(sum(row["count"] for row in table), len(labels))

    def test_repeat_purchase_semantics_are_category_based(self) -> None:
        self.assertTrue(
            repeat_purchase_is_plausible({"categories": ["currency"]})
        )
        self.assertFalse(
            repeat_purchase_is_plausible({"categories": ["collectible"]})
        )


if __name__ == "__main__":
    unittest.main()
