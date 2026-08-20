from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .scoring import clamp
from .synthetic_oracle import StatefulPurchaseOracle


CATEGORIES = (
    "upgrade",
    "cosmetic",
    "currency",
    "subscription",
    "convenience",
    "bundle",
    "social",
    "collectible",
)
CHARACTERS = ("warrior", "mage", "archer", "healer", "assassin", "tank")
SURFACES = (
    "store_home",
    "character_screen",
    "match_preparation",
    "failure_recovery",
    "event_popup",
    "checkout",
)
LIFECYCLE_STAGES = ("new", "growing", "mature", "returning")


@dataclass(frozen=True)
class SyntheticAssumptions:
    schema_version: str = "state-rich-assumptions-v1"
    discount_probability: float = 0.30
    discount_levels: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.40)
    bundle_category_weight: float = 1.0
    bundle_size_median: int = 4
    returning_player_probability: float = 0.65
    weekend_probability: float = 2.0 / 7.0
    special_event_boost_probability: float = 0.10
    session_product_views_median: float = 10.0
    session_product_views_p90: float = 30.0
    session_duration_seconds_median: float = 900.0
    session_duration_seconds_p90: float = 3600.0
    baseline_exit_pressure: float = 0.08
    purchase_cooldown_decay_hours: float = 72.0
    initial_owned_item_count_median: float = 4.0
    initial_owned_item_count_p90: float = 12.0

    def validate(self) -> None:
        if self.schema_version != "state-rich-assumptions-v1":
            raise ValueError("unsupported synthetic assumption schema")
        if not 0.0 <= self.discount_probability <= 1.0:
            raise ValueError("discount_probability must be between 0 and 1")
        if not self.discount_levels or any(
            value <= 0.0 or value >= 1.0 for value in self.discount_levels
        ):
            raise ValueError("discount_levels must contain values between 0 and 1")
        if self.bundle_category_weight <= 0.0 or self.bundle_size_median < 2:
            raise ValueError("bundle assumptions must be positive")
        for name in (
            "returning_player_probability",
            "weekend_probability",
            "special_event_boost_probability",
            "baseline_exit_pressure",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if (
            self.session_product_views_median <= 0.0
            or self.session_product_views_p90
            < self.session_product_views_median
            or self.session_duration_seconds_median <= 0.0
            or self.session_duration_seconds_p90
            < self.session_duration_seconds_median
            or self.purchase_cooldown_decay_hours <= 0.0
            or self.initial_owned_item_count_median <= 0.0
            or self.initial_owned_item_count_p90
            < self.initial_owned_item_count_median
        ):
            raise ValueError("synthetic state assumptions are invalid")


@dataclass(frozen=True)
class SyntheticConfig:
    users: int = 500
    items: int = 250
    impressions: int = 100_000
    days: int = 120
    latent_dimensions: int = 12
    candidate_pool_size: int = 48
    seed: int = 20260818
    start_time: str = "2026-01-01T00:00:00+00:00"

    def validate(self) -> None:
        if self.users < 2 or self.items < 2 or self.impressions < 1:
            raise ValueError("users/items must be >= 2 and impressions must be positive")
        if self.days < 3 or self.latent_dimensions < len(CATEGORIES):
            raise ValueError("days must be >= 3 and latent dimensions must cover categories")
        if self.candidate_pool_size < 2:
            raise ValueError("candidate_pool_size must be >= 2")


class SyntheticDatasetGenerator:
    def __init__(
        self,
        config: SyntheticConfig,
        assumptions: SyntheticAssumptions | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.assumptions = assumptions or SyntheticAssumptions()
        self.assumptions.validate()
        # Synthetic fixtures must be reproducible; this is not security randomness.
        self.rng = random.Random(config.seed)  # nosec B311
        self.start_time = datetime.fromisoformat(config.start_time.replace("Z", "+00:00"))
        if self.start_time.tzinfo is None:
            self.start_time = self.start_time.replace(tzinfo=timezone.utc)

    def generate(
        self,
        output_dir: Path,
        *,
        session_id: str | None = None,
    ) -> Mapping[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        neptune_dir = output_dir / "neptune"
        neptune_dir.mkdir(parents=True, exist_ok=True)
        oracle_dir = output_dir / "oracle"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        generation_session_id = session_id or f"generate-{uuid.uuid4()}"

        users = self._generate_users()
        items = self._generate_items()
        scenarios, oracle_rows = self._generate_interactions(users, items)
        graph_nodes, static_edges = self._graph_static_records(users, items)

        self._write_jsonl(output_dir / "users.jsonl", users)
        self._write_jsonl(output_dir / "items.jsonl", items)
        self._write_jsonl(output_dir / "scenarios.jsonl", scenarios)
        self._write_jsonl(oracle_dir / "oracle.jsonl", oracle_rows)
        self._write_neptune_nodes(neptune_dir / "nodes.csv", graph_nodes)
        self._write_neptune_edges(neptune_dir / "static_edges.csv", static_edges)

        report = self._generation_report(users, items, scenarios, oracle_rows)
        report["phase"] = "generation"
        report["generation_session_id"] = generation_session_id
        report["labeling_session_id"] = None
        report["assumptions"] = asdict(self.assumptions)
        report["files"] = {
            str(path.relative_to(output_dir)): self._sha256(path)
            for path in (
                output_dir / "users.jsonl",
                output_dir / "items.jsonl",
                output_dir / "scenarios.jsonl",
                oracle_dir / "oracle.jsonl",
                neptune_dir / "nodes.csv",
                neptune_dir / "static_edges.csv",
            )
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    def _generate_users(self) -> list[dict[str, Any]]:
        users: list[dict[str, Any]] = []
        for index in range(self.config.users):
            category_raw = [self.rng.gammavariate(0.8, 1.0) for _ in CATEGORIES]
            category_preferences = self._normalize_positive(category_raw)
            latent = category_preferences + [
                self.rng.gauss(0.0, 0.35)
                for _ in range(self.config.latent_dimensions - len(CATEGORIES))
            ]
            latent = self._unit(latent)
            lifecycle = self.rng.choices(
                LIFECYCLE_STAGES, weights=(0.18, 0.34, 0.38, 0.10), k=1
            )[0]
            spending_power = self.rng.lognormvariate(math.log(14_000), 0.75)
            price_sensitivity = clamp(self.rng.betavariate(2.4, 2.0))
            discount_sensitivity = clamp(self.rng.betavariate(2.0, 1.7))
            pickiness = clamp(self.rng.betavariate(2.2, 2.2))
            engagement = clamp(self.rng.betavariate(2.3, 1.8))
            novelty = clamp(self.rng.betavariate(1.8, 2.3))
            social_conformity = clamp(self.rng.betavariate(2.0, 2.0))
            impulsivity = clamp(self.rng.betavariate(1.6, 2.5))
            favorite_character = self.rng.choice(CHARACTERS)
            random_effect = self.rng.gauss(0.0, 0.55)
            returning_player = (
                self.rng.random()
                < self.assumptions.returning_player_probability
            )
            initial_owned_item_count = max(
                0,
                min(
                    24,
                    int(
                        round(
                            self._sample_lognormal_from_median_p90(
                                self.assumptions.initial_owned_item_count_median,
                                self.assumptions.initial_owned_item_count_p90,
                            )
                        )
                    )
                    - 1,
                ),
            )
            initial_purchase_age_hours = self.rng.lognormvariate(
                math.log(self.assumptions.purchase_cooldown_decay_hours * 2.0),
                0.75,
            )
            category_mapping = {
                category: round(category_preferences[position], 8)
                for position, category in enumerate(CATEGORIES)
            }
            users.append(
                {
                    "user_id": f"user-{index:06d}",
                    "lifecycle_stage": lifecycle,
                    "favorite_character": favorite_character,
                    "spending_power": round(spending_power, 4),
                    "price_sensitivity": round(price_sensitivity, 8),
                    "discount_sensitivity": round(discount_sensitivity, 8),
                    "pickiness": round(pickiness, 8),
                    "engagement": round(engagement, 8),
                    "novelty_affinity": round(novelty, 8),
                    "social_conformity": round(social_conformity, 8),
                    "impulsivity": round(impulsivity, 8),
                    "returning_player": returning_player,
                    "initial_owned_item_count": initial_owned_item_count,
                    "initial_purchase_age_hours": round(
                        initial_purchase_age_hours,
                        4,
                    ),
                    "initial_currency_multiplier": round(
                        self.rng.lognormvariate(math.log(2.4), 0.55),
                        8,
                    ),
                    "daily_currency_refill_rate": round(
                        self.rng.uniform(0.08, 0.24),
                        8,
                    ),
                    "category_preferences": category_mapping,
                    "latent_vector": [round(value, 8) for value in latent],
                    "random_effect": round(random_effect, 8),
                    "persona_summary": self._persona_summary(
                        category_mapping,
                        favorite_character,
                        lifecycle,
                        price_sensitivity,
                        pickiness,
                    ),
                }
            )
        return users

    def _generate_items(self) -> list[dict[str, Any]]:
        base_prices = {
            "upgrade": 9_900,
            "cosmetic": 12_900,
            "currency": 19_900,
            "subscription": 14_900,
            "convenience": 7_900,
            "bundle": 29_900,
            "social": 4_900,
            "collectible": 16_900,
        }
        items: list[dict[str, Any]] = []
        category_weights = [
            self.assumptions.bundle_category_weight if category == "bundle" else 1.0
            for category in CATEGORIES
        ]
        for index in range(self.config.items):
            primary = self.rng.choices(CATEGORIES, weights=category_weights, k=1)[0]
            primary_index = CATEGORIES.index(primary)
            secondary = (
                self.rng.choice([value for value in CATEGORIES if value != primary])
                if self.rng.random() < 0.25
                else None
            )
            categories = [primary] + ([secondary] if secondary else [])
            vector = [self.rng.gauss(0.0, 0.18) for _ in range(self.config.latent_dimensions)]
            vector[primary_index] += 1.0
            if secondary:
                vector[CATEGORIES.index(secondary)] += 0.45
            vector = self._unit(vector)
            quality = clamp(self.rng.betavariate(3.0, 1.8))
            utility = clamp(
                self.rng.betavariate(2.8, 1.7)
                if primary in {"upgrade", "currency", "convenience", "bundle"}
                else self.rng.betavariate(1.5, 2.8)
            )
            emotionality = clamp(
                self.rng.betavariate(2.8, 1.7)
                if primary in {"cosmetic", "social", "collectible"}
                else self.rng.betavariate(1.6, 2.6)
            )
            price = max(
                500.0,
                self.rng.lognormvariate(math.log(base_prices[primary]), 0.42),
            )
            discount = (
                self.rng.choice(self.assumptions.discount_levels)
                if self.rng.random() < self.assumptions.discount_probability
                else 0.0
            )
            release_day = self.rng.randrange(max(1, int(self.config.days * 0.75)))
            event_id = f"campaign-{release_day // 21:02d}"
            bundle_size = (
                max(2, int(round(self.rng.lognormvariate(
                    math.log(self.assumptions.bundle_size_median), 0.45
                ))))
                if primary == "bundle"
                else 1
            )
            items.append(
                {
                    "item_id": f"item-{index:06d}",
                    "categories": categories,
                    "character": self.rng.choice(CHARACTERS),
                    "event_id": event_id,
                    "price": round(price, 2),
                    "discount_rate": discount,
                    "bundle_size": bundle_size,
                    "quality": round(quality, 8),
                    "utility": round(utility, 8),
                    "emotionality": round(emotionality, 8),
                    "release_day": release_day,
                    "latent_vector": [round(value, 8) for value in vector],
                    "random_effect": round(self.rng.gauss(0.0, 0.45), 8),
                }
            )
        return items

    def _generate_interactions(
        self,
        users: Sequence[Mapping[str, Any]],
        items: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scenarios: list[dict[str, Any]] = []
        oracle_rows: list[dict[str, Any]] = []
        item_stats: dict[str, Counter[str]] = defaultdict(Counter)
        user_weights = [0.15 + float(user["engagement"]) for user in users]

        for index in range(self.config.impressions):
            fraction = index / max(1, self.config.impressions - 1)
            elapsed_seconds = fraction * self.config.days * 86400
            timestamp = self.start_time + timedelta(
                seconds=elapsed_seconds + self.rng.uniform(0.0, 1800.0)
            )
            day = min(self.config.days - 1, int((timestamp - self.start_time).days))
            user = self.rng.choices(users, weights=user_weights, k=1)[0]
            available = [item for item in items if int(item["release_day"]) <= day]
            if len(available) < 2:
                available = list(items)
            item, exposure_propensity, policy = self._select_exposure(
                user, available, item_stats, day
            )
            session_index = index // 6
            context = self._context(user, item, day, index % 6)
            impression_id = f"imp-{index:09d}"
            session_id = f"session-{session_index:09d}"
            split = self._split(day)
            scenarios.append({
                "impression_id": impression_id,
                "session_id": session_id,
                "timestamp": timestamp.isoformat(),
                "day": day,
                "split": split,
                "user_id": user["user_id"],
                "item_id": item["item_id"],
                "surface": context["surface"],
                "session_fatigue": round(context["session_fatigue"], 8),
                "progression_need": round(context["progression_need"], 8),
                "recent_failure_intensity": round(
                    context["recent_failure_intensity"],
                    8,
                ),
                "event_urgency": round(context["event_urgency"], 8),
                "current_goals": list(context["current_goals"]),
                "active_character": context["active_character"],
                "event_active": context["event_active"],
                "session_product_views": context[
                    "session_product_views"
                ],
                "session_duration_seconds": round(
                    context["session_duration_seconds"],
                    4,
                ),
                "weekend": context["weekend"],
                "exposure_policy": policy,
                "exposure_propensity": round(exposure_propensity, 8),
            })
            oracle_rows.append({
                "impression_id": impression_id,
                "oracle_version": StatefulPurchaseOracle.version,
                "latent_shock": round(self.rng.gauss(0.0, 1.0), 8),
            })

            item_stats[item["item_id"]]["VIEWED"] += 1

        return scenarios, oracle_rows

    def _select_exposure(
        self,
        user: Mapping[str, Any],
        available: Sequence[Mapping[str, Any]],
        item_stats: Mapping[str, Counter[str]],
        day: int,
    ) -> tuple[Mapping[str, Any], float, str]:
        pool_size = min(self.config.candidate_pool_size, len(available))
        pool = self.rng.sample(list(available), pool_size)
        policy = self.rng.choices(
            ("personalized", "popular", "exploration"), weights=(0.62, 0.25, 0.13), k=1
        )[0]
        logits: list[float] = []
        for item in pool:
            preference = self._affinity(user["latent_vector"], item["latent_vector"])
            statistics = item_stats.get(item["item_id"], Counter())
            popularity = math.log1p(statistics.get("PURCHASED", 0)) + 0.2 * math.log1p(
                statistics.get("CLICKED", 0)
            )
            freshness = math.exp(-max(0, day - int(item["release_day"])) / 28.0)
            if policy == "personalized":
                score = 3.2 * preference + 0.25 * popularity + 0.3 * freshness
            elif policy == "popular":
                score = 1.5 * popularity + 0.5 * preference
            else:
                score = 0.5 * preference + 1.2 * freshness + self.rng.gammavariate(1.0, 1.0)
            logits.append(score)
        probabilities = self._softmax(logits, temperature=0.72)
        item = self.rng.choices(pool, weights=probabilities, k=1)[0]
        propensity = probabilities[pool.index(item)]
        return item, propensity, policy

    def _context(
        self,
        user: Mapping[str, Any],
        item: Mapping[str, Any],
        day: int,
        session_position: int,
    ) -> dict[str, Any]:
        surface = self.rng.choices(
            SURFACES, weights=(0.30, 0.16, 0.14, 0.14, 0.18, 0.08), k=1
        )[0]
        base_need = 0.65 if surface == "failure_recovery" else 0.35
        progression_need = clamp(base_need + self.rng.gauss(0.0, 0.20))
        session_product_views = max(
            1,
            int(
                round(
                    self._sample_lognormal_from_median_p90(
                        self.assumptions.session_product_views_median,
                        self.assumptions.session_product_views_p90,
                    )
                )
            ),
        )
        session_duration_seconds = self._sample_lognormal_from_median_p90(
            self.assumptions.session_duration_seconds_median,
            self.assumptions.session_duration_seconds_p90,
        )
        position_pressure = min(
            1.0,
            session_position / max(1.0, session_product_views),
        )
        fatigue = clamp(
            0.45 * position_pressure
            + self.assumptions.baseline_exit_pressure
            + self.rng.betavariate(1.5, 5.0)
        )
        active_character = (
            user["favorite_character"]
            if self.rng.random() < 0.68
            else self.rng.choice(CHARACTERS)
        )
        event_active = item["event_id"] == f"campaign-{day // 21:02d}"
        event_day = day % 21
        event_urgency = (
            clamp(event_day / 20.0 + self.rng.gauss(0.0, 0.08))
            if event_active
            else 0.0
        )
        if self.rng.random() < self.assumptions.special_event_boost_probability:
            event_urgency = max(
                event_urgency,
                clamp(self.rng.betavariate(3.0, 1.6)),
            )
        recent_failure_intensity = clamp(
            (0.65 if surface == "failure_recovery" else 0.18)
            + 0.30 * progression_need
            + self.rng.gauss(0.0, 0.16)
        )
        preferred_categories = sorted(
            user["category_preferences"],
            key=user["category_preferences"].get,
            reverse=True,
        )
        goal_category = (
            str(item["categories"][0])
            if progression_need > 0.68 and self.rng.random() < 0.55
            else str(preferred_categories[0])
        )
        current_goals = (
            f"progress:{goal_category}",
            f"character:{active_character}",
        )
        return {
            "surface": surface,
            "progression_need": progression_need,
            "recent_failure_intensity": recent_failure_intensity,
            "event_urgency": event_urgency,
            "current_goals": current_goals,
            "session_fatigue": fatigue,
            "active_character": active_character,
            "event_active": event_active,
            "session_product_views": session_product_views,
            "session_duration_seconds": session_duration_seconds,
            "weekend": (
                self.rng.random() < self.assumptions.weekend_probability
            ),
        }

    def _graph_static_records(
        self,
        users: Sequence[Mapping[str, Any]],
        items: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for category in CATEGORIES:
            nodes.append({"id": f"category:{category}", "nodeId": category, "label": "Category"})
        for character in CHARACTERS:
            nodes.append({"id": f"character:{character}", "nodeId": character, "label": "Character"})
        for event_index in range(math.ceil(self.config.days / 21)):
            event_id = f"campaign-{event_index:02d}"
            nodes.append({"id": f"event:{event_id}", "nodeId": event_id, "label": "Event"})
        for user in users:
            nodes.append(
                {
                    "id": f"user:{user['user_id']}",
                    "nodeId": user["user_id"],
                    "userId": user["user_id"],
                    "label": "User",
                    "priceSensitivity": user["price_sensitivity"],
                    "spendingPower": user["spending_power"],
                    "latentVector": json.dumps(user["latent_vector"]),
                }
            )
            edges.append(
                self._static_edge(
                    f"user:{user['user_id']}",
                    f"character:{user['favorite_character']}",
                    "PLAYS",
                )
            )
        for item in items:
            nodes.append(
                {
                    "id": f"item:{item['item_id']}",
                    "nodeId": item["item_id"],
                    "itemId": item["item_id"],
                    "label": "Item",
                    "price": item["price"],
                    "quality": item["quality"],
                    "utility": item["utility"],
                    "emotionality": item["emotionality"],
                    "latentVector": json.dumps(item["latent_vector"]),
                }
            )
            for category in item["categories"]:
                edges.append(
                    self._static_edge(
                        f"item:{item['item_id']}",
                        f"category:{category}",
                        "IN_CATEGORY",
                    )
                )
            edges.append(
                self._static_edge(
                    f"item:{item['item_id']}",
                    f"character:{item['character']}",
                    "TARGETS",
                )
            )
            edges.append(
                self._static_edge(
                    f"item:{item['item_id']}",
                    f"event:{item['event_id']}",
                    "AVAILABLE_IN",
                )
            )
        return nodes, edges

    def _static_edge(self, start: str, end: str, edge_type: str) -> dict[str, Any]:
        edge_id = hashlib.sha256(
            f"{start}|{edge_type}|{end}".encode()
        ).hexdigest()
        return {
            "id": f"static-{edge_id}",
            "start": start,
            "end": end,
            "type": edge_type,
            "timestamp": "",
            "weight": 1.0,
            "sessionId": "",
            "synthetic": True,
        }

    def _generation_report(
        self,
        users: Sequence[Mapping[str, Any]],
        items: Sequence[Mapping[str, Any]],
        scenarios: Sequence[Mapping[str, Any]],
        oracle_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        splits = Counter(row["split"] for row in scenarios)
        policies = Counter(row["exposure_policy"] for row in scenarios)
        propensities = [float(row["exposure_propensity"]) for row in scenarios]
        required_state_fields = (
            "progression_need",
            "recent_failure_intensity",
            "event_urgency",
            "current_goals",
        )
        state_coverage = {
            field: sum(field in row for row in scenarios)
            / max(1, len(scenarios))
            for field in required_state_fields
        }
        state_unique_values = {
            field: len(
                {
                    json.dumps(row.get(field), sort_keys=True)
                    for row in scenarios
                }
            )
            for field in required_state_fields
        }
        quality_gates = {
            "all_temporal_splits_present": all(
                splits.get(name, 0) > 0 for name in ("train", "validation", "test")
            ),
            "valid_exposure_propensity": min(propensities) > 0.0
            and max(propensities) <= 1.0,
            "complete_exogenous_state": all(
                value == 1.0 for value in state_coverage.values()
            ),
            "non_degenerate_exogenous_state": all(
                value > 1 for value in state_unique_values.values()
            ),
            "oracle_spec_only": all(
                row.get("oracle_version")
                == StatefulPurchaseOracle.version
                and "ground_truth_purchase_probability" not in row
                for row in oracle_rows
            ),
        }
        return {
            "generator": "state-rich-scenario-generator-v3",
            "config": asdict(self.config),
            "counts": {
                "users": len(users),
                "items": len(items),
                "scenarios": len(scenarios),
                "oracle_rows": len(oracle_rows),
            },
            "state_coverage": {
                key: round(value, 8)
                for key, value in state_coverage.items()
            },
            "state_unique_values": state_unique_values,
            "splits": dict(splits),
            "exposure_policies": dict(policies),
            "quality_gates": {
                **quality_gates,
                "all_passed": all(quality_gates.values()),
            },
        }

    def _split(self, day: int) -> str:
        fraction = day / max(1, self.config.days)
        if fraction < 0.70:
            return "train"
        if fraction < 0.85:
            return "validation"
        return "test"

    @staticmethod
    def _persona_summary(
        preferences: Mapping[str, float],
        character: str,
        lifecycle: str,
        price_sensitivity: float,
        pickiness: float,
    ) -> str:
        top = sorted(preferences, key=preferences.get, reverse=True)[:2]
        price_phrase = "가격에 민감하고" if price_sensitivity > 0.6 else "가격보다 효용을 중시하고"
        picky_phrase = "선택 기준이 엄격한" if pickiness > 0.6 else "새 상품을 비교적 쉽게 시도하는"
        return (
            f"{lifecycle} 단계의 {character} 중심 사용자로 {top[0]}와 {top[1]} 상품을 선호하며, "
            f"{price_phrase} {picky_phrase} 성향"
        )

    @staticmethod
    def _affinity(user_vector: Sequence[float], item_vector: Sequence[float]) -> float:
        cosine = sum(left * right for left, right in zip(user_vector, item_vector))
        return clamp((cosine + 1.0) / 2.0)

    @staticmethod
    def _normalize_positive(values: Sequence[float]) -> list[float]:
        total = sum(values)
        return [value / total for value in values]

    @staticmethod
    def _unit(values: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    @staticmethod
    def _softmax(values: Sequence[float], temperature: float) -> list[float]:
        scaled = [value / temperature for value in values]
        maximum = max(scaled)
        exponentials = [math.exp(value - maximum) for value in scaled]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def _sample_lognormal_from_median_p90(
        self,
        median: float,
        p90: float,
    ) -> float:
        sigma = max(
            0.05,
            math.log(max(p90, median) / median) / 1.2815515655446004,
        )
        return self.rng.lognormvariate(math.log(median), sigma)

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")

    @staticmethod
    def _write_neptune_nodes(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        fieldnames = (
            ":ID",
            "nodeId:String",
            "userId:String",
            "itemId:String",
            "price:Double",
            "quality:Double",
            "utility:Double",
            "emotionality:Double",
            "priceSensitivity:Double",
            "spendingPower:Double",
            "latentVector:String",
            ":LABEL",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        ":ID": row.get("id", ""),
                        "nodeId:String": row.get("nodeId", ""),
                        "userId:String": row.get("userId", ""),
                        "itemId:String": row.get("itemId", ""),
                        "price:Double": row.get("price", ""),
                        "quality:Double": row.get("quality", ""),
                        "utility:Double": row.get("utility", ""),
                        "emotionality:Double": row.get("emotionality", ""),
                        "priceSensitivity:Double": row.get("priceSensitivity", ""),
                        "spendingPower:Double": row.get("spendingPower", ""),
                        "latentVector:String": row.get("latentVector", ""),
                        ":LABEL": row.get("label", ""),
                    }
                )

    @staticmethod
    def _write_neptune_edges(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        fieldnames = (
            ":ID",
            ":START_ID",
            ":END_ID",
            ":TYPE",
            "timestamp:DateTime",
            "weight:Double",
            "sessionId:String",
            "synthetic:Bool",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        ":ID": row.get("id", ""),
                        ":START_ID": row.get("start", ""),
                        ":END_ID": row.get("end", ""),
                        ":TYPE": row.get("type", ""),
                        "timestamp:DateTime": row.get("timestamp", ""),
                        "weight:Double": row.get("weight", 1.0),
                        "sessionId:String": row.get("sessionId", ""),
                        "synthetic:Bool": str(bool(row.get("synthetic", True))).lower(),
                    }
                )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--items", type=int, default=250)
    parser.add_argument("--impressions", type=int, default=100_000)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--session-id")
    args = parser.parse_args()
    report = SyntheticDatasetGenerator(
        SyntheticConfig(
            users=args.users,
            items=args.items,
            impressions=args.impressions,
            days=args.days,
            seed=args.seed,
        )
    ).generate(args.output, session_id=args.session_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
