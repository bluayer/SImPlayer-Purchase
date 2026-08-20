from __future__ import annotations

import unittest

from purchase_behavior_simulator.state_counterfactuals import (
    build_state_counterfactual_pairs,
)


class StateCounterfactualProtocolTest(unittest.TestCase):
    def test_builds_complete_paired_state_protocol(self) -> None:
        case = {
            "case_id": "u1:imp1",
            "payload": {
                "operation": "evaluate_snapshot",
                "request": {
                    "request_id": "base",
                    "item": {
                        "item_id": "upgrade-1",
                        "categories": ["upgrade"],
                        "price": 10000.0,
                        "discount_rate": 0.1,
                    },
                    "game_state": {
                        "currency_balance": 15000.0,
                        "progression_need": 0.7,
                        "recent_failure_intensity": 0.6,
                        "inventory_overlap": 0.0,
                        "event_urgency": 0.5,
                        "purchase_cooldown": 0.2,
                        "current_goals": ["progress:upgrade"],
                        "owned_item_ids": [],
                    },
                },
            },
        }
        pairs, report = build_state_counterfactual_pairs(
            [case],
            base_case_limit=1,
        )

        self.assertEqual(report["pairs"], 6)
        self.assertTrue(report["quality_gates"]["all_passed"])
        by_dimension = {
            pair["dimension"]: pair for pair in pairs
        }
        balance = by_dimension["currency_balance"]
        self.assertGreater(
            balance["favorable_payload"]["request"]["game_state"][
                "currency_balance"
            ],
            balance["adverse_payload"]["request"]["game_state"][
                "currency_balance"
            ],
        )
        ownership = by_dimension["ownership"]
        self.assertIn(
            "upgrade-1",
            ownership["adverse_payload"]["request"]["game_state"][
                "owned_item_ids"
            ],
        )

    def test_rejects_protocol_without_game_state(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "GameStateSnapshot",
        ):
            build_state_counterfactual_pairs(
                [
                    {
                        "case_id": "legacy",
                        "payload": {
                            "request": {
                                "item": {
                                    "item_id": "i1",
                                    "categories": ["upgrade"],
                                    "price": 1,
                                }
                            }
                        },
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
