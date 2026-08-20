from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from .scoring import clamp, sigmoid


REPEATABLE_CATEGORIES = frozenset(
    {"currency", "subscription", "convenience"}
)


class StatefulPurchaseOracle:
    version = "stateful-causal-funnel-v3"

    @classmethod
    def probabilities(
        cls,
        *,
        user: Mapping[str, Any],
        item: Mapping[str, Any],
        scenario: Mapping[str, Any],
        game_state: Mapping[str, Any],
        item_stats: Counter[str],
        repeat_stats: Counter[str],
        day: int,
        latent_shock: float = 0.0,
    ) -> tuple[dict[str, float], dict[str, float]]:
        affinity = cls._affinity(
            user["latent_vector"],
            item["latent_vector"],
        )
        category_preference = max(
            float(user["category_preferences"].get(category, 0.0))
            for category in item["categories"]
        )
        active_character = str(
            scenario.get("active_character", "")
        )
        character_match = float(active_character == item["character"])
        favorite_character_match = float(
            user["favorite_character"] == item["character"]
        )
        effective_price = float(item["price"]) * (
            1.0 - float(item["discount_rate"])
        )
        balance = float(game_state.get("currency_balance", 0.0))
        balance_ratio = balance / max(1.0, effective_price)
        affordability = sigmoid(3.2 * (balance_ratio - 1.0))
        discount_effect = float(item["discount_rate"]) * (
            0.5 + float(user["discount_sensitivity"])
        )
        progression_need = float(
            game_state.get("progression_need", 0.5)
        )
        failure_intensity = float(
            game_state.get("recent_failure_intensity", 0.0)
        )
        urgency = float(game_state.get("event_urgency", 0.0))
        inventory_overlap = float(
            game_state.get("inventory_overlap", 0.0)
        )
        cooldown = float(game_state.get("purchase_cooldown", 0.0))
        current_goals = tuple(
            str(value) for value in game_state.get("current_goals", ())
        )
        owned_item_ids = {
            str(value) for value in game_state.get("owned_item_ids", ())
        }
        repeatable = bool(
            REPEATABLE_CATEGORIES.intersection(
                str(value) for value in item["categories"]
            )
        )
        exact_ownership = float(
            item["item_id"] in owned_item_ids and not repeatable
        )
        goal_fit = max(
            (
                float(
                    any(
                        category in goal
                        for goal in current_goals
                    )
                )
                for category in item["categories"]
            ),
            default=0.0,
        )
        utility = float(item["utility"])
        need_fit = progression_need * utility
        failure_relief = failure_intensity * utility
        emotional_fit = float(user["novelty_affinity"]) * float(
            item["emotionality"]
        )
        quality = float(item["quality"])
        social_proof = sigmoid(
            -1.0
            + math.log1p(item_stats.get("PURCHASED", 0))
            - 0.35 * math.log1p(item_stats.get("VIEWED", 0))
        )
        repeat_saturation = min(
            1.0,
            repeat_stats.get("VIEWED", 0) / 5.0,
        )
        prior_purchase = min(
            1.0,
            repeat_stats.get("PURCHASED", 0),
        )
        freshness = math.exp(
            -max(0, day - int(item["release_day"])) / 35.0
        )
        event_fit = float(scenario.get("event_active", False))
        fatigue = float(scenario["session_fatigue"])
        surface_purchase_intent = {
            "store_home": 0.0,
            "character_screen": 0.15,
            "match_preparation": 0.20,
            "failure_recovery": 0.40,
            "event_popup": 0.18,
            "checkout": 0.75,
        }[str(scenario["surface"])]
        user_effect = float(user["random_effect"])
        item_effect = float(item["random_effect"])
        impulsivity = float(user["impulsivity"])
        price_sensitivity = float(user["price_sensitivity"])
        pickiness = float(user["pickiness"])

        click_logit = (
            -3.15
            + 2.30 * affinity
            + 1.10 * category_preference
            + 0.55 * character_match
            + 0.45 * favorite_character_match
            + 0.70 * emotional_fit
            + 0.45 * freshness
            + 0.25 * event_fit
            + 0.30 * urgency
            + 0.30 * goal_fit
            + 0.30 * social_proof * float(user["social_conformity"])
            - 0.90 * fatigue
            - 0.65 * pickiness
            - 0.55 * exact_ownership
            + 0.35 * user_effect
            + 0.25 * item_effect
            + 0.15 * latent_shock
        )
        purchase_given_click_logit = (
            -4.55
            + 2.55 * affinity
            + 1.20 * category_preference
            + 1.55 * affordability
            - 1.25 * price_sensitivity * (1.0 - affordability)
            + 0.95 * discount_effect
            + 1.15 * need_fit
            + 0.70 * failure_relief
            + 0.65 * goal_fit
            + 0.60 * urgency
            + 0.70 * emotional_fit
            + 0.75 * quality
            + 0.45 * character_match
            + 0.45 * surface_purchase_intent
            + 0.25 * social_proof * float(user["social_conformity"])
            + 0.30 * impulsivity
            - 0.80 * fatigue
            - 0.95 * repeat_saturation
            - 1.70 * prior_purchase
            - 1.60 * inventory_overlap
            - 1.35 * cooldown
            - 6.00 * exact_ownership
            + user_effect
            + item_effect
            + 0.20 * latent_shock
        )
        direct_purchase_logit = (
            -7.25
            + 1.45 * affinity
            + 1.45 * affordability
            + 0.75 * discount_effect
            + 0.75 * need_fit
            + 0.45 * failure_relief
            + 0.40 * goal_fit
            + 0.55 * urgency
            + 0.90 * impulsivity
            + 0.55 * surface_purchase_intent
            - 0.75 * fatigue
            - 1.25 * inventory_overlap
            - 1.20 * cooldown
            - 6.00 * exact_ownership
            + 0.55 * user_effect
            + 0.35 * item_effect
            + 0.15 * latent_shock
        )
        organic_logit = (
            -6.75
            + 1.70 * affinity
            + 1.10 * category_preference
            + 1.35 * affordability
            + 0.80 * need_fit
            + 0.40 * failure_relief
            + 0.40 * goal_fit
            + 0.45 * quality
            - 0.75 * repeat_saturation
            - 1.70 * prior_purchase
            - 1.45 * inventory_overlap
            - 1.20 * cooldown
            - 6.00 * exact_ownership
            + 0.65 * user_effect
            + 0.50 * item_effect
            + 0.15 * latent_shock
        )

        click_probability = sigmoid(click_logit)
        purchase_given_click = sigmoid(purchase_given_click_logit)
        direct_purchase = sigmoid(direct_purchase_logit)
        purchase_probability = clamp(
            click_probability * purchase_given_click
            + (1.0 - click_probability) * direct_purchase,
            1e-6,
            1.0 - 1e-6,
        )
        probabilities = {
            "click": click_probability,
            "purchase_given_click": purchase_given_click,
            "direct_purchase": direct_purchase,
            "purchase": purchase_probability,
            "organic_purchase": sigmoid(organic_logit),
        }
        components = {
            "affinity": affinity,
            "category_preference": category_preference,
            "character_match": character_match,
            "affordability": affordability,
            "balance_to_price_ratio": balance_ratio,
            "discount_effect": discount_effect,
            "progression_need_fit": need_fit,
            "failure_relief": failure_relief,
            "goal_fit": goal_fit,
            "event_urgency": urgency,
            "emotional_fit": emotional_fit,
            "quality": quality,
            "social_proof": social_proof,
            "inventory_overlap": inventory_overlap,
            "purchase_cooldown": cooldown,
            "exact_ownership": exact_ownership,
            "repeat_saturation": repeat_saturation,
            "freshness": freshness,
            "fatigue": fatigue,
        }
        return probabilities, components

    @staticmethod
    def _affinity(
        user_vector: Sequence[float],
        item_vector: Sequence[float],
    ) -> float:
        cosine = sum(
            left * right
            for left, right in zip(user_vector, item_vector)
        )
        return clamp((cosine + 1.0) / 2.0)
