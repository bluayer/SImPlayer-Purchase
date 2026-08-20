from __future__ import annotations

import unittest
from datetime import datetime, timezone

from purchase_behavior_simulator.decision_process import (
    DecisionState,
    apply_commitment_gate,
    build_decision_state,
    evaluate_counterfactual_consistency,
    intentions_from_state,
    merge_model_decision,
)
from purchase_behavior_simulator.episodic_memory import (
    select_contrastive_memory_documents,
    serialize_observed_transitions,
)
from purchase_behavior_simulator.models import (
    MemoryDocument,
    ObservedStateTransition,
    SimulationRequest,
)


ACTION_DISTRIBUTIONS = {
    "ITEM_EXPOSURE": {
        "CLICK": 0.35,
        "SKIP": 0.35,
        "EXIT": 0.10,
        "PURCHASE_NOW": 0.20,
    },
    "ITEM_DETAIL": {
        "PURCHASE": 0.30,
        "BACK": 0.50,
        "EXIT": 0.20,
    },
}


def request_with_state(game_state):
    return SimulationRequest.from_dict(
        {
            "user": {
                "user_id": "u1",
                "category_preferences": {"upgrade": 0.9},
                "pickiness": 0.3,
                "price_sensitivity": 0.5,
            },
            "item": {
                "item_id": "upgrade-1",
                "categories": ["upgrade"],
                "price": 100,
                "attributes": {"limited_time": True},
            },
            "context": {
                "surface": "failure_recovery",
                "budget_reference": 200,
                "timestamp": "2026-08-20T00:00:00+00:00",
            },
            "game_state": game_state,
            "interactions": [
                {
                    "event_type": "purchase",
                    "timestamp": "2026-08-18T00:00:00+00:00",
                    "item_id": "upgrade-old",
                    "categories": ["upgrade"],
                }
            ],
        }
    )


class DecisionProcessTest(unittest.TestCase):
    def test_event_membership_alone_does_not_imply_urgency(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {
                    "user_id": "u1",
                    "category_preferences": {"cosmetic": 0.5},
                },
                "item": {
                    "item_id": "cosmetic-1",
                    "categories": ["cosmetic"],
                    "price": 100,
                    "attributes": {"event_id": "campaign-1"},
                },
                "context": {"budget_reference": 200},
                "game_state": {
                    "currency_balance": 200,
                    "progression_need": 0.0,
                    "recent_failure_intensity": 0.0,
                    "event_urgency": 0.0,
                },
            }
        )

        self.assertEqual(build_decision_state(request).urgency, 0.0)

    def test_model_decision_is_bounded_by_deterministic_state(self) -> None:
        baseline = DecisionState(
            need_strength=0.2,
            selection_strength=0.2,
            feasibility=0.2,
            urgency=0.2,
            uncertainty=0.2,
            hesitation=0.2,
            evidence_confidence=0.6,
            state_confidence=1.0,
            repeat_purchase_plausible=False,
        )

        merged, intentions = merge_model_decision(
            baseline,
            {
                "need_strength": 1.0,
                "selection_strength": 1.0,
                "feasibility": 1.0,
                "urgency": 1.0,
                "uncertainty": 1.0,
                "hesitation": 1.0,
            },
            {
                "BUY_NOW": 1.0,
                "EXPLORE": 0.0,
                "DEFER": 0.0,
                "REJECT": 0.0,
            },
            model_confidence=1.0,
        )

        self.assertAlmostEqual(merged.selection_strength, 0.48)
        self.assertLess(merged.selection_strength, 0.5)
        self.assertLess(intentions["BUY_NOW"], 0.5)

    def test_current_game_state_changes_commitment_without_training(self) -> None:
        ready = build_decision_state(
            request_with_state(
                {
                    "currency_balance": 250,
                    "progression_need": 0.9,
                    "recent_failure_intensity": 0.8,
                    "event_urgency": 0.7,
                }
            )
        )
        constrained = build_decision_state(
            request_with_state(
                {
                    "currency_balance": 20,
                    "progression_need": 0.1,
                    "inventory_overlap": 0.9,
                    "purchase_cooldown": 0.8,
                }
            )
        )

        self.assertGreater(
            ready.commitment_strength,
            constrained.commitment_strength,
        )
        self.assertGreater(ready.feasibility, constrained.feasibility)
        self.assertLess(ready.hesitation, constrained.hesitation)

    def test_defer_is_internal_and_maps_to_existing_store_actions(self) -> None:
        request = request_with_state(
            {
                "currency_balance": 80,
                "progression_need": 0.7,
                "event_urgency": 0.1,
            }
        )
        state = build_decision_state(request)
        intentions = {
            "BUY_NOW": 0.05,
            "EXPLORE": 0.20,
            "DEFER": 0.65,
            "REJECT": 0.10,
        }

        result = apply_commitment_gate(
            ACTION_DISTRIBUTIONS,
            state,
            intentions,
        )

        self.assertIn("DEFER", result.intentions)
        self.assertNotIn(
            "DEFER",
            result.distributions["ITEM_EXPOSURE"],
        )
        self.assertNotIn("DEFER", result.distributions["ITEM_DETAIL"])
        self.assertLess(
            result.distributions["ITEM_DETAIL"]["PURCHASE"],
            ACTION_DISTRIBUTIONS["ITEM_DETAIL"]["PURCHASE"],
        )
        self.assertGreater(
            result.distributions["ITEM_DETAIL"]["BACK"],
            ACTION_DISTRIBUTIONS["ITEM_DETAIL"]["BACK"],
        )

    def test_commitment_gate_requires_current_state_evidence(self) -> None:
        without_state = SimulationRequest.from_dict(
            {
                "user": {
                    "user_id": "u1",
                    "category_preferences": {"upgrade": 0.9},
                },
                "item": {
                    "item_id": "upgrade-1",
                    "categories": ["upgrade"],
                    "price": 100,
                },
                "context": {"budget_reference": 200},
            }
        )
        with_state = request_with_state(
            {
                "currency_balance": 250,
                "progression_need": 0.9,
                "recent_failure_intensity": 0.8,
                "event_urgency": 0.7,
            }
        )
        weak = build_decision_state(without_state)
        strong = build_decision_state(with_state)

        weak_result = apply_commitment_gate(
            ACTION_DISTRIBUTIONS,
            weak,
            intentions_from_state(weak),
        )
        strong_result = apply_commitment_gate(
            ACTION_DISTRIBUTIONS,
            strong,
            intentions_from_state(strong),
        )

        self.assertEqual(weak.state_confidence, 0.0)
        self.assertEqual(strong.state_confidence, 1.0)
        self.assertLess(
            max(weak_result.total_variation.values()),
            max(strong_result.total_variation.values()),
        )

    def test_counterfactual_constraints_do_not_increase_purchase(self) -> None:
        request = request_with_state(
            {
                "currency_balance": 250,
                "progression_need": 0.9,
                "recent_failure_intensity": 0.8,
                "event_urgency": 0.7,
            }
        )
        state = build_decision_state(request)
        commitment = apply_commitment_gate(
            ACTION_DISTRIBUTIONS,
            state,
            intentions_from_state(state),
        )

        result = evaluate_counterfactual_consistency(
            request,
            ACTION_DISTRIBUTIONS,
            commitment.distributions,
            state,
        )

        self.assertTrue(all(result.checks.values()))
        self.assertLessEqual(
            result.purchase_probabilities["price_increase"],
            result.purchase_probabilities["current"],
        )
        self.assertLessEqual(
            result.purchase_probabilities["need_resolved"],
            result.purchase_probabilities["current"],
        )
        self.assertLessEqual(
            result.purchase_probabilities["ownership_constraint"],
            result.purchase_probabilities["current"],
        )

    def test_owned_item_in_game_state_blocks_nonrepeatable_item(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {
                    "item_id": "skin-1",
                    "categories": ["cosmetic"],
                },
                "game_state": {"owned_item_ids": ["skin-1"]},
            }
        )

        self.assertTrue(request.item.already_owned)

    def test_contrastive_memory_keeps_purchase_and_non_purchase(self) -> None:
        now = datetime(2026, 8, 20, tzinfo=timezone.utc)

        def document(action: str, next_state: str) -> MemoryDocument:
            return MemoryDocument(
                content=serialize_observed_transitions(
                    (
                        ObservedStateTransition(
                            state=(
                                "ITEM_DETAIL"
                                if action in {"PURCHASE", "BACK"}
                                else "ITEM_EXPOSURE"
                            ),
                            action=action,
                            next_state=next_state,
                            timestamp=now,
                            item_id="upgrade-1",
                            categories=("upgrade",),
                        ),
                    )
                )
            )

        selected = select_contrastive_memory_documents(
            (
                document("PURCHASE", "PURCHASED"),
                document("PURCHASE_NOW", "PURCHASED"),
                document("BACK", "EXITED"),
                document("SKIP", "EXITED"),
            ),
            limit=4,
            per_outcome=1,
        )
        text = " ".join(value.content for value in selected)

        self.assertIn('"PURCHASE"', text)
        self.assertTrue('"BACK"' in text or '"SKIP"' in text)


if __name__ == "__main__":
    unittest.main()
