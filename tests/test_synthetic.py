from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from purchase_behavior_simulator.synthetic import (
    SyntheticAssumptions,
    SyntheticConfig,
    SyntheticDatasetGenerator,
)
from purchase_behavior_simulator.synthetic_labeling import (
    LabelingConfig,
    SyntheticDatasetLabeler,
)


class SyntheticGeneratorTest(unittest.TestCase):
    def test_generates_and_labels_in_separate_reproducible_sessions(self) -> None:
        config = SyntheticConfig(
            users=12,
            items=20,
            impressions=600,
            days=30,
            seed=11,
        )
        with (
            tempfile.TemporaryDirectory() as generation_dir,
            tempfile.TemporaryDirectory() as first_label_dir,
            tempfile.TemporaryDirectory() as second_label_dir,
        ):
            generation = SyntheticDatasetGenerator(config).generate(
                Path(generation_dir), session_id="generator-session"
            )
            first = SyntheticDatasetLabeler(
                LabelingConfig(seed=19)
            ).label(
                Path(generation_dir),
                Path(first_label_dir),
                session_id="labeler-session",
            )
            second = SyntheticDatasetLabeler(
                LabelingConfig(seed=19)
            ).label(
                Path(generation_dir),
                Path(second_label_dir),
                session_id="labeler-session",
            )
            self.assertEqual(first["files"], second["files"])
            self.assertEqual(generation["counts"]["scenarios"], 600)
            self.assertEqual(
                generation["assumptions"],
                {
                    "schema_version": "state-rich-assumptions-v1",
                    "discount_probability": 0.3,
                    "discount_levels": (0.1, 0.15, 0.2, 0.3, 0.4),
                    "bundle_category_weight": 1.0,
                    "bundle_size_median": 4,
                    "returning_player_probability": 0.65,
                    "weekend_probability": 2.0 / 7.0,
                    "special_event_boost_probability": 0.1,
                    "session_product_views_median": 10.0,
                    "session_product_views_p90": 30.0,
                    "session_duration_seconds_median": 900.0,
                    "session_duration_seconds_p90": 3600.0,
                    "baseline_exit_pressure": 0.08,
                    "purchase_cooldown_decay_hours": 72.0,
                    "initial_owned_item_count_median": 4.0,
                    "initial_owned_item_count_p90": 12.0,
                },
            )
            self.assertEqual(first["counts"]["impressions"], 600)
            self.assertGreater(first["rates"]["click"], 0.0)
            self.assertGreater(first["rates"]["purchase"], 0.0)
            self.assertGreater(
                first["rates"]["high_affinity_expected_purchase"],
                first["rates"]["low_affinity_expected_purchase"],
            )
            self.assertTrue(first["quality_gates"]["all_passed"])
            self.assertGreaterEqual(first["action_paths"]["unique"], 8)
            self.assertGreaterEqual(first["action_paths"]["max_length"], 4)
            self.assertGreater(
                first["action_paths"]["length_at_least_4_rate"],
                0.0,
            )
            self.assertEqual(
                set(first["counterfactual_pass_rates"].values()),
                {1.0},
            )
            self.assertTrue(
                all(
                    value == 1.0
                    for value in first["state_coverage"].values()
                )
            )

            scenario = json.loads(
                (Path(generation_dir) / "scenarios.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertNotIn("purchased", scenario)
            self.assertNotIn("ground_truth_purchase_probability", scenario)
            self.assertIn("recent_failure_intensity", scenario)
            self.assertIn("event_urgency", scenario)
            self.assertIn("current_goals", scenario)

            oracle = json.loads(
                (Path(generation_dir) / "oracle" / "oracle.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertNotIn(
                "ground_truth_purchase_probability",
                oracle,
            )
            self.assertIn("latent_shock", oracle)

            impression_rows = [
                json.loads(line)
                for line in (
                    Path(first_label_dir) / "impressions.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            row = impression_rows[0]
            self.assertIn("ground_truth_purchase_probability", row)
            self.assertIn("ground_truth_organic_probability", row)
            self.assertIn("exposure_propensity", row)
            self.assertIn("causal_components", row)
            self.assertEqual(
                set(row["game_state"]),
                {
                    "currency_balance",
                    "progression_need",
                    "recent_failure_intensity",
                    "inventory_overlap",
                    "event_urgency",
                    "purchase_cooldown",
                    "current_goals",
                    "owned_item_ids",
                    "features",
                },
            )
            observable_actions = {
                action
                for impression in impression_rows
                for action in impression["observed_action_path"]
            }
            self.assertNotIn("COMPARISON", observable_actions)
            self.assertNotIn("HESITATE", observable_actions)
            self.assertNotIn("DEFER", observable_actions)
            self.assertTrue(
                {
                    "START_PURCHASE",
                    "CONFIRM_PURCHASE",
                    "PAYMENT_SUCCESS",
                }.issubset(observable_actions)
            )

            items = {
                value["item_id"]: value
                for value in (
                    json.loads(line)
                    for line in (
                        Path(first_label_dir) / "items.jsonl"
                    )
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }
            rows_by_user: dict[str, list[dict]] = {}
            for value in impression_rows:
                rows_by_user.setdefault(
                    value["user_id"],
                    [],
                ).append(value)
            verified_transition = False
            repeatable = {
                "currency",
                "subscription",
                "convenience",
            }
            for values in rows_by_user.values():
                for current, following in zip(values, values[1:]):
                    categories = set(
                        items[current["item_id"]]["categories"]
                    )
                    if (
                        current["purchased"]
                        and not categories.intersection(repeatable)
                    ):
                        self.assertIn(
                            current["item_id"],
                            following["game_state"][
                                "owned_item_ids"
                            ],
                        )
                        self.assertGreater(
                            following["game_state"][
                                "purchase_cooldown"
                            ],
                            0.0,
                        )
                        verified_transition = True
                        break
                if verified_transition:
                    break
            self.assertTrue(
                verified_transition,
                "fixture must contain a purchase followed by another exposure",
            )

    def test_rejects_invalid_assumptions(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "synthetic state assumptions",
        ):
            SyntheticAssumptions(
                purchase_cooldown_decay_hours=0.0,
            ).validate()

    def test_rejects_same_generation_and_labeling_session(self) -> None:
        config = SyntheticConfig(users=4, items=5, impressions=10, days=3, seed=2)
        with (
            tempfile.TemporaryDirectory() as generation_dir,
            tempfile.TemporaryDirectory() as label_dir,
        ):
            SyntheticDatasetGenerator(config).generate(
                Path(generation_dir), session_id="same-session"
            )
            with self.assertRaisesRegex(ValueError, "session IDs must differ"):
                SyntheticDatasetLabeler().label(
                    Path(generation_dir),
                    Path(label_dir),
                    session_id="same-session",
                )


if __name__ == "__main__":
    unittest.main()
