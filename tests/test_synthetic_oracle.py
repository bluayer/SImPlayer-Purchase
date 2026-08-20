from __future__ import annotations

import unittest
from collections import Counter

from purchase_behavior_simulator.synthetic_oracle import (
    StatefulPurchaseOracle,
)


class StatefulPurchaseOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.user = {
            "favorite_character": "mage",
            "latent_vector": [1.0, 0.0],
            "category_preferences": {"upgrade": 0.8},
            "discount_sensitivity": 0.5,
            "novelty_affinity": 0.4,
            "social_conformity": 0.4,
            "random_effect": 0.0,
            "impulsivity": 0.3,
            "price_sensitivity": 0.6,
            "pickiness": 0.4,
        }
        self.item = {
            "item_id": "upgrade-1",
            "categories": ["upgrade"],
            "character": "mage",
            "price": 10000.0,
            "discount_rate": 0.0,
            "utility": 0.9,
            "emotionality": 0.2,
            "quality": 0.8,
            "release_day": 0,
            "random_effect": 0.0,
            "latent_vector": [1.0, 0.0],
        }
        self.scenario = {
            "active_character": "mage",
            "session_fatigue": 0.2,
            "surface": "failure_recovery",
            "event_active": True,
        }
        self.state = {
            "currency_balance": 20000.0,
            "progression_need": 0.8,
            "recent_failure_intensity": 0.7,
            "inventory_overlap": 0.0,
            "event_urgency": 0.7,
            "purchase_cooldown": 0.0,
            "current_goals": ["progress:upgrade"],
            "owned_item_ids": [],
        }

    def probability(self, state_updates=None, item_updates=None) -> float:
        probabilities, _ = StatefulPurchaseOracle.probabilities(
            user=self.user,
            item={**self.item, **(item_updates or {})},
            scenario=self.scenario,
            game_state={**self.state, **(state_updates or {})},
            item_stats=Counter(),
            repeat_stats=Counter(),
            day=1,
        )
        return probabilities["purchase"]

    def test_state_changes_purchase_probability_in_causal_direction(self) -> None:
        self.assertGreater(
            self.probability({"currency_balance": 30000.0}),
            self.probability({"currency_balance": 1000.0}),
        )
        self.assertGreater(
            self.probability({"progression_need": 1.0}),
            self.probability({"progression_need": 0.0}),
        )
        self.assertGreater(
            self.probability({"event_urgency": 1.0}),
            self.probability({"event_urgency": 0.0}),
        )
        self.assertGreater(
            self.probability({"purchase_cooldown": 0.0}),
            self.probability({"purchase_cooldown": 1.0}),
        )

    def test_owned_nonrepeatable_item_suppresses_purchase(self) -> None:
        owned = self.probability(
            {
                "inventory_overlap": 1.0,
                "owned_item_ids": ["upgrade-1"],
            }
        )
        self.assertLess(owned, self.probability())
        self.assertLess(
            self.probability(item_updates={"price": 15000.0}),
            self.probability(),
        )


if __name__ == "__main__":
    unittest.main()
