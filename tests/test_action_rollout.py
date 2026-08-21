from __future__ import annotations

import unittest
from pathlib import Path

from purchase_behavior_simulator.action_rollout import (
    ActionGraph,
    DEFAULT_ACTION_GRAPH,
    DeterministicStoreEnvironment,
    StoreState,
    UserAction,
    limit_action_distribution_revision,
    normalize_action_distributions,
    load_action_graph,
    rollout_purchase_probability,
)


class ActionRolloutTest(unittest.TestCase):
    def test_checked_in_default_graph_matches_runtime_default(self) -> None:
        graph = load_action_graph(
            Path(__file__).parents[1]
            / "src"
            / "purchase_behavior_simulator"
            / "action_graphs"
            / "game-store-purchase.json"
        )

        self.assertEqual(graph.to_dict(), DEFAULT_ACTION_GRAPH.to_dict())

    def test_default_graph_separates_user_actions_from_environment_events(
        self,
    ) -> None:
        self.assertEqual(
            DEFAULT_ACTION_GRAPH.transition(
                "PURCHASE_CONFIRMATION",
                "CONFIRM_PURCHASE",
            ).kind,
            "user_action",
        )
        self.assertEqual(
            DEFAULT_ACTION_GRAPH.transition(
                "PAYMENT_PROCESSING",
                "PAYMENT_FAILED",
            ).kind,
            "environment_event",
        )
        self.assertAlmostEqual(
            DEFAULT_ACTION_GRAPH.base_distribution_weight,
            0.25,
        )

    def test_environment_rejects_impossible_transition(self) -> None:
        environment = DeterministicStoreEnvironment()

        with self.assertRaises(ValueError):
            environment.transition(
                StoreState.ITEM_EXPOSURE,
                UserAction.CONFIRM_PURCHASE,
            )

    def test_rollout_sums_every_purchase_path(self) -> None:
        result = rollout_purchase_probability(
            {
                "ITEM_EXPOSURE": {
                    "CLICK": 0.4,
                    "SKIP": 0.3,
                    "EXIT": 0.1,
                    "PURCHASE_NOW": 0.2,
                },
                "ITEM_DETAIL": {
                    "START_PURCHASE": 0.3,
                    "BACK": 0.7,
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
            surface="store_home",
            max_depth=4,
        )

        self.assertAlmostEqual(
            result.purchase_probability,
            0.2 + 0.4 * 0.3,
        )

    def test_checkout_surface_starts_in_item_detail(self) -> None:
        result = rollout_purchase_probability(
            {
                "ITEM_DETAIL": {
                    "START_PURCHASE": 0.7,
                    "BACK": 0.3,
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
            surface="checkout",
            max_depth=3,
        )

        self.assertEqual(result.initial_state, StoreState.ITEM_DETAIL)
        self.assertAlmostEqual(result.purchase_probability, 0.7)

    def test_critic_revision_is_bounded_per_state(self) -> None:
        proposed = {
            "ITEM_EXPOSURE": {
                "CLICK": 0.1,
                "SKIP": 0.1,
                "EXIT": 0.0,
                "PURCHASE_NOW": 0.8,
            },
            "ITEM_DETAIL": {
                "START_PURCHASE": 0.8,
                "BACK": 0.2,
            },
        }
        revised = {
            "ITEM_EXPOSURE": {
                "CLICK": 0.3,
                "SKIP": 0.4,
                "EXIT": 0.2,
                "PURCHASE_NOW": 0.1,
            },
            "ITEM_DETAIL": {
                "START_PURCHASE": 0.1,
                "BACK": 0.9,
            },
        }

        limited = limit_action_distribution_revision(
            proposed,
            revised,
            max_total_variation=0.10,
        )

        proposed_normalized = normalize_action_distributions(proposed)
        for state in proposed:
            total_variation = 0.5 * sum(
                abs(
                    limited[state][action]
                    - proposed_normalized[state][action]
                )
                for action in proposed_normalized[state]
            )
            self.assertAlmostEqual(total_variation, 0.10)
        self.assertLess(
            limited["ITEM_EXPOSURE"]["PURCHASE_NOW"],
            proposed_normalized["ITEM_EXPOSURE"]["PURCHASE_NOW"],
        )

    def test_distributions_are_normalized_without_adding_actions(self) -> None:
        normalized = normalize_action_distributions(
            {
                "ITEM_EXPOSURE": {
                    "CLICK": 2.0,
                    "SKIP": 1.0,
                    "IMPOSSIBLE": 100.0,
                }
            }
        )

        self.assertAlmostEqual(
            sum(normalized["ITEM_EXPOSURE"].values()),
            1.0,
        )
        self.assertNotIn("IMPOSSIBLE", normalized["ITEM_EXPOSURE"])

    def test_custom_graph_adds_states_actions_and_timing_declaratively(
        self,
    ) -> None:
        graph = ActionGraph.from_dict(
            {
                "graph_id": "limited-offer",
                "version": "1",
                "default_initial_state": "OFFER",
                "terminal_outcomes": {
                    "PURCHASED": "purchase",
                    "IGNORED": "exit",
                },
                "max_depth": 3,
                "transitions": [
                    {
                        "state": "OFFER",
                        "action": "OPEN",
                        "next_state": "CONFIRM",
                        "timing": {"expected_seconds": 2},
                    },
                    {
                        "state": "OFFER",
                        "action": "IGNORE",
                        "next_state": "IGNORED",
                        "timing": {"expected_seconds": 1},
                    },
                    {
                        "state": "CONFIRM",
                        "action": "BUY",
                        "next_state": "PURCHASED",
                        "timing": {"expected_seconds": 4},
                    },
                    {
                        "state": "CONFIRM",
                        "action": "CANCEL",
                        "next_state": "IGNORED",
                        "timing": {"expected_seconds": 3},
                    },
                ],
            }
        )

        result = rollout_purchase_probability(
            {
                "OFFER": {"OPEN": 0.4, "IGNORE": 0.6},
                "CONFIRM": {"BUY": 0.25, "CANCEL": 0.75},
            },
            surface="event_popup",
            graph=graph,
        )

        self.assertAlmostEqual(result.purchase_probability, 0.1)
        purchase_path = next(path for path in result.paths if path.purchased)
        self.assertEqual(purchase_path.actions, ("OPEN", "BUY"))
        self.assertEqual(purchase_path.expected_duration_seconds, 6.0)
        self.assertEqual(result.graph_id, "limited-offer")


if __name__ == "__main__":
    unittest.main()
