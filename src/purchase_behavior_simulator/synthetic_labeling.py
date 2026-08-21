from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .action_rollout import DEFAULT_ACTION_GRAPH
from .scoring import clamp
from .synthetic import SyntheticAssumptions
from .synthetic_oracle import (
    REPEATABLE_CATEGORIES,
    StatefulPurchaseOracle,
)


@dataclass(frozen=True)
class LabelingConfig:
    seed: int = 20260819


@dataclass
class UserDynamicState:
    currency_balance: float
    owned_item_ids: set[str]
    last_purchase_at: datetime | None
    last_seen_at: datetime | None = None
    progression_relief: float = 0.0


class SyntheticDatasetLabeler:
    def __init__(self, config: LabelingConfig | None = None) -> None:
        self.config = config or LabelingConfig()
        # Synthetic labels must be reproducible; this is not security randomness.
        self.rng = random.Random(  # nosec B311
            self.config.seed
        )

    def label(
        self,
        source_dir: Path,
        output_dir: Path,
        *,
        session_id: str | None = None,
    ) -> Mapping[str, Any]:
        source_dir = source_dir.resolve()
        output_dir = output_dir.resolve()
        if source_dir == output_dir:
            raise ValueError("generation and labeling outputs must use different directories")

        source_manifest = json.loads(
            (source_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if source_manifest.get("phase") != "generation":
            raise ValueError("source manifest must come from the generation phase")
        self._verify_source_files(source_dir, source_manifest["files"])

        generation_session_id = str(source_manifest["generation_session_id"])
        labeling_session_id = session_id or f"label-{uuid.uuid4()}"
        if labeling_session_id == generation_session_id:
            raise ValueError("generation and labeling session IDs must differ")

        scenarios = sorted(
            self._read_jsonl(source_dir / "scenarios.jsonl"),
            key=lambda row: str(row["timestamp"]),
        )
        users = {
            str(row["user_id"]): row
            for row in self._read_jsonl(source_dir / "users.jsonl")
        }
        items = {
            str(row["item_id"]): row
            for row in self._read_jsonl(source_dir / "items.jsonl")
        }
        assumptions = SyntheticAssumptions(
            **source_manifest.get("assumptions", {})
        )
        assumptions.validate()
        oracle_by_id = {
            str(row["impression_id"]): row
            for row in self._read_jsonl(source_dir / "oracle" / "oracle.jsonl")
        }
        if len(scenarios) != len(oracle_by_id):
            raise ValueError("scenario and oracle row counts differ")

        impressions: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        behavior_edges: list[dict[str, Any]] = []
        user_states = self._initialize_user_states(
            users,
            items,
            scenarios,
        )
        item_stats: dict[str, Counter[str]] = defaultdict(Counter)
        user_item_stats: dict[
            tuple[str, str],
            Counter[str],
        ] = defaultdict(Counter)
        counterfactual_checks: Counter[str] = Counter()
        for scenario in scenarios:
            impression_id = str(scenario["impression_id"])
            oracle = oracle_by_id.get(impression_id)
            if oracle is None:
                raise ValueError(f"missing oracle row for {impression_id}")
            if (
                oracle.get("oracle_version")
                != StatefulPurchaseOracle.version
            ):
                raise ValueError(
                    f"unsupported oracle version for {impression_id}"
                )

            user_id = str(scenario["user_id"])
            item_id = str(scenario["item_id"])
            user = users[user_id]
            item = items[item_id]
            timestamp = datetime.fromisoformat(
                str(scenario["timestamp"]).replace("Z", "+00:00")
            )
            state = user_states[user_id]
            self._advance_state(
                state,
                user,
                timestamp,
            )
            game_state = self._game_state(
                state,
                scenario,
                item,
                items,
                assumptions,
            )
            probabilities, components = (
                StatefulPurchaseOracle.probabilities(
                    user=user,
                    item=item,
                    scenario=scenario,
                    game_state=game_state,
                    item_stats=item_stats[item_id],
                    repeat_stats=user_item_stats[(user_id, item_id)],
                    day=int(scenario["day"]),
                    latent_shock=float(oracle["latent_shock"]),
                )
            )
            self._record_counterfactual_checks(
                counterfactual_checks,
                user=user,
                item=item,
                scenario=scenario,
                game_state=game_state,
                item_stats=item_stats[item_id],
                repeat_stats=user_item_stats[(user_id, item_id)],
                day=int(scenario["day"]),
                latent_shock=float(oracle["latent_shock"]),
            )

            starts_at_detail = str(scenario["surface"]) == "checkout"
            clicked = starts_at_detail or self.rng.random() < float(
                probabilities["click"]
            )
            purchase_probability = (
                float(probabilities["purchase_given_click"])
                if clicked
                else float(probabilities["direct_purchase"])
            )
            purchased = self.rng.random() < purchase_probability
            (
                observed_initial_state,
                observed_action_path,
                topped_up,
            ) = self._sample_observed_action_path(
                clicked=clicked,
                purchased=purchased,
                user=user,
                item=item,
                scenario=scenario,
                game_state=game_state,
            )
            oracle_payload = {
                "ground_truth_click_probability": round(
                    probabilities["click"],
                    8,
                ),
                "ground_truth_purchase_given_click_probability": round(
                    probabilities["purchase_given_click"],
                    8,
                ),
                "ground_truth_direct_purchase_probability": round(
                    probabilities["direct_purchase"],
                    8,
                ),
                "ground_truth_purchase_probability": round(
                    probabilities["purchase"],
                    8,
                ),
                "ground_truth_organic_probability": round(
                    probabilities["organic_purchase"],
                    8,
                ),
                "ground_truth_incremental_uplift": round(
                    probabilities["purchase"]
                    - probabilities["organic_purchase"],
                    8,
                ),
                "causal_components": {
                    key: round(value, 8)
                    for key, value in components.items()
                },
            }
            impressions.append(
                {
                    **scenario,
                    **oracle_payload,
                    "game_state": game_state,
                    "context_features": {
                        "progression_need": game_state[
                            "progression_need"
                        ],
                        "recent_failure_intensity": game_state[
                            "recent_failure_intensity"
                        ],
                        "inventory_overlap": game_state[
                            "inventory_overlap"
                        ],
                        "event_urgency": game_state["event_urgency"],
                        "purchase_cooldown": game_state[
                            "purchase_cooldown"
                        ],
                    },
                    "clicked": int(clicked),
                    "purchased": int(purchased),
                    "observed_initial_state": observed_initial_state,
                    "observed_action_path": observed_action_path,
                }
            )

            event_types = ["VIEWED", *observed_action_path]
            for offset, event_type in enumerate(event_types):
                normalized_event_type = {
                    "CLICK": "CLICKED",
                    "PAYMENT_SUCCESS": "PURCHASED",
                }.get(event_type, event_type)
                event = {
                    "event_id": f"evt-{len(events):010d}",
                    "impression_id": impression_id,
                    "session_id": scenario["session_id"],
                    "timestamp": (timestamp + timedelta(seconds=offset * 2)).isoformat(),
                    "user_id": scenario["user_id"],
                    "item_id": scenario["item_id"],
                    "event_type": normalized_event_type,
                    "synthetic": True,
                }
                events.append(event)
                behavior_edges.append(self._behavior_edge(event))
                item_stats[item_id][normalized_event_type] += 1
                user_item_stats[(user_id, item_id)][
                    normalized_event_type
                ] += 1

            if purchased:
                if topped_up:
                    self._apply_top_up(state, item)
                self._apply_purchase(
                    state,
                    item,
                    timestamp,
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        neptune_dir = output_dir / "neptune"
        neptune_dir.mkdir(parents=True, exist_ok=True)
        for relative_path in ("users.jsonl", "items.jsonl", "neptune/nodes.csv"):
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_dir / relative_path, destination)

        self._write_jsonl(output_dir / "impressions.jsonl", impressions)
        self._write_jsonl(output_dir / "events.jsonl", events)
        self._write_combined_edges(
            source_dir / "neptune" / "static_edges.csv",
            neptune_dir / "edges.csv",
            behavior_edges,
        )

        report = self._quality_report(
            impressions,
            events,
            counterfactual_checks,
        )
        report.update(
            {
                "phase": "labeling",
                "labeler": "stateful-independent-oracle-labeler-v2",
                "generation_session_id": generation_session_id,
                "labeling_session_id": labeling_session_id,
                "labeling_config": asdict(self.config),
                "source_manifest_sha256": self._sha256(
                    source_dir / "manifest.json"
                ),
            }
        )
        output_files = (
            output_dir / "users.jsonl",
            output_dir / "items.jsonl",
            output_dir / "impressions.jsonl",
            output_dir / "events.jsonl",
            neptune_dir / "nodes.csv",
            neptune_dir / "edges.csv",
        )
        report["files"] = {
            str(path.relative_to(output_dir)): self._sha256(path)
            for path in output_files
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    def _sample_observed_action_path(
        self,
        *,
        clicked: bool,
        purchased: bool,
        user: Mapping[str, Any],
        item: Mapping[str, Any],
        scenario: Mapping[str, Any],
        game_state: Mapping[str, Any],
    ) -> tuple[str, list[str], bool]:
        """Sample only UI and transaction events that can be instrumented."""
        starts_at_detail = str(scenario["surface"]) == "checkout"
        initial_state = "ITEM_DETAIL" if starts_at_detail else "ITEM_EXPOSURE"
        state = initial_state
        actions: list[str] = []

        def take(action: str) -> None:
            nonlocal state
            transition = DEFAULT_ACTION_GRAPH.transition(state, action)
            actions.append(action)
            state = transition.next_state

        if not starts_at_detail:
            if clicked:
                take("CLICK")
            elif purchased:
                take("PURCHASE_NOW")
            else:
                take(
                    "EXIT"
                    if self.rng.random()
                    < 0.25 + 0.45 * float(scenario["session_fatigue"])
                    else "SKIP"
                )
                return initial_state, actions, False

        effective_price = float(item["price"]) * (
            1.0 - float(item["discount_rate"])
        )
        insufficient = (
            float(game_state["currency_balance"]) + 1e-9
            < effective_price
        )
        hesitation = clamp(
            0.18
            + 0.30 * float(user["price_sensitivity"])
            + 0.18 * float(scenario["session_fatigue"])
            - 0.15 * float(game_state["event_urgency"])
        )

        if not purchased:
            if self.rng.random() >= 0.38 + 0.22 * hesitation:
                take("BACK")
                take("EXIT" if self.rng.random() < 0.35 else "SKIP")
                return initial_state, actions, False

            take("START_PURCHASE")
            if self.rng.random() < 0.42 + 0.35 * hesitation:
                take("CANCEL")
                take("BACK")
                take("EXIT" if self.rng.random() < 0.45 else "SKIP")
                return initial_state, actions, False

            take("CONFIRM_PURCHASE")
            if insufficient:
                take("INSUFFICIENT_CURRENCY")
                if self.rng.random() < 0.35:
                    take("OPEN_TOP_UP")
                    take("CANCEL_TOP_UP")
                    take("EXIT")
                else:
                    take("BACK_TO_ITEM")
                    take("BACK")
                    take("EXIT")
            else:
                take("PAYMENT_FAILED")
                take("CANCEL")
                take("BACK")
                take("EXIT")
            return initial_state, actions, False

        if state == "ITEM_DETAIL":
            take("START_PURCHASE")
        if self.rng.random() < 0.12 * hesitation:
            take("CANCEL")
            take("START_PURCHASE")
        take("CONFIRM_PURCHASE")
        topped_up = False
        if insufficient:
            take("INSUFFICIENT_CURRENCY")
            take("OPEN_TOP_UP")
            take("TOP_UP_SUCCESS")
            topped_up = True
            take("CONFIRM_PURCHASE")
        elif self.rng.random() < 0.04:
            take("PAYMENT_FAILED")
            take("CONFIRM_PURCHASE")
        take("PAYMENT_SUCCESS")
        return initial_state, actions, topped_up

    @staticmethod
    def _verify_source_files(
        source_dir: Path,
        expected_hashes: Mapping[str, str],
    ) -> None:
        for relative_path, expected_hash in expected_hashes.items():
            path = source_dir / relative_path
            actual_hash = SyntheticDatasetLabeler._sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(f"source file hash mismatch: {relative_path}")

    def _initialize_user_states(
        self,
        users: Mapping[str, Mapping[str, Any]],
        items: Mapping[str, Mapping[str, Any]],
        scenarios: Sequence[Mapping[str, Any]],
    ) -> dict[str, UserDynamicState]:
        first_scenario: dict[str, Mapping[str, Any]] = {}
        for scenario in scenarios:
            first_scenario.setdefault(
                str(scenario["user_id"]),
                scenario,
            )

        states: dict[str, UserDynamicState] = {}
        for user_id, user in sorted(users.items()):
            first = first_scenario.get(user_id)
            if first is None:
                continue
            first_day = int(first["day"])
            first_timestamp = datetime.fromisoformat(
                str(first["timestamp"]).replace("Z", "+00:00")
            )
            candidates = [
                item
                for item in items.values()
                if int(item.get("release_day", 0)) <= first_day
            ] or list(items.values())
            ranked = sorted(
                candidates,
                key=lambda item: (
                    max(
                        float(
                            user["category_preferences"].get(
                                category,
                                0.0,
                            )
                        )
                        for category in item["categories"]
                    )
                    + 0.08 * self.rng.random()
                ),
                reverse=True,
            )
            owned_count = min(
                int(user.get("initial_owned_item_count", 0)),
                len(ranked),
            )
            owned_item_ids = {
                str(item["item_id"])
                for item in ranked[:owned_count]
            }
            last_purchase_at = None
            if bool(user.get("returning_player")):
                last_purchase_at = first_timestamp - timedelta(
                    hours=float(
                        user.get(
                            "initial_purchase_age_hours",
                            720.0,
                        )
                    )
                )
            states[user_id] = UserDynamicState(
                currency_balance=max(
                    0.0,
                    float(user["spending_power"])
                    * float(
                        user.get(
                            "initial_currency_multiplier",
                            2.4,
                        )
                    ),
                ),
                owned_item_ids=owned_item_ids,
                last_purchase_at=last_purchase_at,
                last_seen_at=first_timestamp,
            )
        return states

    @staticmethod
    def _advance_state(
        state: UserDynamicState,
        user: Mapping[str, Any],
        timestamp: datetime,
    ) -> None:
        if state.last_seen_at is None:
            state.last_seen_at = timestamp
            return
        elapsed_hours = max(
            0.0,
            (timestamp - state.last_seen_at).total_seconds() / 3600.0,
        )
        refill = (
            float(user["spending_power"])
            * float(user.get("daily_currency_refill_rate", 0.12))
            * elapsed_hours
            / 24.0
        )
        state.currency_balance = min(
            float(user["spending_power"]) * 8.0,
            state.currency_balance + refill,
        )
        state.progression_relief *= math.exp(
            -elapsed_hours / 72.0
        )
        state.last_seen_at = timestamp

    @staticmethod
    def _game_state(
        state: UserDynamicState,
        scenario: Mapping[str, Any],
        item: Mapping[str, Any],
        items: Mapping[str, Mapping[str, Any]],
        assumptions: SyntheticAssumptions,
    ) -> dict[str, Any]:
        owned_categories: set[str] = set()
        for item_id in state.owned_item_ids:
            owned = items.get(item_id)
            if owned is not None:
                owned_categories.update(
                    str(value) for value in owned["categories"]
                )
        target_categories = {
            str(value) for value in item["categories"]
        }
        exact_ownership = item["item_id"] in state.owned_item_ids
        category_overlap = len(
            target_categories.intersection(owned_categories)
        ) / max(1, len(target_categories))
        inventory_overlap = (
            1.0
            if exact_ownership
            else 0.65 * category_overlap
        )
        timestamp = datetime.fromisoformat(
            str(scenario["timestamp"]).replace("Z", "+00:00")
        )
        if state.last_purchase_at is None:
            cooldown = 0.0
        else:
            elapsed_hours = max(
                0.0,
                (
                    timestamp - state.last_purchase_at
                ).total_seconds()
                / 3600.0,
            )
            cooldown = math.exp(
                -elapsed_hours
                / assumptions.purchase_cooldown_decay_hours
            )
        progression_need = clamp(
            float(scenario["progression_need"])
            * (1.0 - 0.55 * state.progression_relief)
        )
        failure_intensity = clamp(
            float(scenario["recent_failure_intensity"])
            * (1.0 - 0.45 * state.progression_relief)
        )
        return {
            "currency_balance": round(state.currency_balance, 4),
            "progression_need": round(progression_need, 8),
            "recent_failure_intensity": round(
                failure_intensity,
                8,
            ),
            "inventory_overlap": round(
                clamp(inventory_overlap),
                8,
            ),
            "event_urgency": round(
                float(scenario["event_urgency"]),
                8,
            ),
            "purchase_cooldown": round(clamp(cooldown), 8),
            "current_goals": list(scenario["current_goals"]),
            "owned_item_ids": sorted(state.owned_item_ids),
            "features": {
                "session_product_views": float(
                    scenario["session_product_views"]
                ),
                "session_duration_seconds": float(
                    scenario["session_duration_seconds"]
                ),
                "weekend": float(bool(scenario["weekend"])),
            },
        }

    @staticmethod
    def _apply_purchase(
        state: UserDynamicState,
        item: Mapping[str, Any],
        timestamp: datetime,
    ) -> None:
        effective_price = float(item["price"]) * (
            1.0 - float(item["discount_rate"])
        )
        state.currency_balance = max(
            0.0,
            state.currency_balance - effective_price,
        )
        categories = {
            str(value) for value in item["categories"]
        }
        if not REPEATABLE_CATEGORIES.intersection(categories):
            state.owned_item_ids.add(str(item["item_id"]))
        state.last_purchase_at = timestamp
        state.progression_relief = max(
            state.progression_relief,
            0.75 * float(item.get("utility", 0.0)),
        )

    @staticmethod
    def _apply_top_up(
        state: UserDynamicState,
        item: Mapping[str, Any],
    ) -> None:
        effective_price = float(item["price"]) * (
            1.0 - float(item["discount_rate"])
        )
        state.currency_balance = max(
            state.currency_balance,
            effective_price * 1.5,
        )

    @staticmethod
    def _record_counterfactual_checks(
        checks: Counter[str],
        *,
        user: Mapping[str, Any],
        item: Mapping[str, Any],
        scenario: Mapping[str, Any],
        game_state: Mapping[str, Any],
        item_stats: Counter[str],
        repeat_stats: Counter[str],
        day: int,
        latent_shock: float,
    ) -> None:
        def probability(
            *,
            state_updates: Mapping[str, Any] | None = None,
            item_updates: Mapping[str, Any] | None = None,
        ) -> float:
            state = {
                **game_state,
                **dict(state_updates or {}),
            }
            candidate_item = {
                **item,
                **dict(item_updates or {}),
            }
            values, _ = StatefulPurchaseOracle.probabilities(
                user=user,
                item=candidate_item,
                scenario=scenario,
                game_state=state,
                item_stats=item_stats,
                repeat_stats=repeat_stats,
                day=day,
                latent_shock=latent_shock,
            )
            return float(values["purchase"])

        effective_price = float(item["price"]) * (
            1.0 - float(item["discount_rate"])
        )
        comparisons = {
            "currency_balance": (
                probability(
                    state_updates={
                        "currency_balance": 3.0 * effective_price
                    }
                ),
                probability(
                    state_updates={
                        "currency_balance": 0.1 * effective_price
                    }
                ),
            ),
            "price": (
                probability(),
                probability(
                    item_updates={
                        "price": 1.5 * float(item["price"])
                    }
                ),
            ),
            "progression_need": (
                probability(
                    state_updates={"progression_need": 1.0}
                ),
                probability(
                    state_updates={"progression_need": 0.0}
                ),
            ),
            "event_urgency": (
                probability(
                    state_updates={"event_urgency": 1.0}
                ),
                probability(
                    state_updates={"event_urgency": 0.0}
                ),
            ),
            "purchase_cooldown": (
                probability(
                    state_updates={"purchase_cooldown": 0.0}
                ),
                probability(
                    state_updates={"purchase_cooldown": 1.0}
                ),
            ),
        }
        if not REPEATABLE_CATEGORIES.intersection(
            str(value) for value in item["categories"]
        ):
            comparisons["ownership"] = (
                probability(
                    state_updates={
                        "inventory_overlap": 0.0,
                        "owned_item_ids": [
                            value
                            for value in game_state[
                                "owned_item_ids"
                            ]
                            if value != item["item_id"]
                        ],
                    }
                ),
                probability(
                    state_updates={
                        "inventory_overlap": 1.0,
                        "owned_item_ids": [
                            *game_state["owned_item_ids"],
                            item["item_id"],
                        ],
                    }
                ),
            )
        for name, (expected_high, expected_low) in comparisons.items():
            checks[f"{name}:total"] += 1
            if expected_high + 1e-12 >= expected_low:
                checks[f"{name}:passed"] += 1

    @staticmethod
    def _behavior_edge(event: Mapping[str, Any]) -> dict[str, Any]:
        return {
            ":ID": event["event_id"],
            ":START_ID": f"user:{event['user_id']}",
            ":END_ID": f"item:{event['item_id']}",
            ":TYPE": event["event_type"],
            "timestamp:DateTime": event["timestamp"],
            "weight:Double": 1.0,
            "sessionId:String": event["session_id"],
            "synthetic:Bool": "true",
        }

    @staticmethod
    def _write_combined_edges(
        static_path: Path,
        output_path: Path,
        behavior_edges: Sequence[Mapping[str, Any]],
    ) -> None:
        with static_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = tuple(reader.fieldnames or ())
            if not fieldnames:
                raise ValueError("static edge CSV has no header")
            with output_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(reader)
                writer.writerows(behavior_edges)

    @staticmethod
    def _quality_report(
        impressions: Sequence[Mapping[str, Any]],
        events: Sequence[Mapping[str, Any]],
        counterfactual_checks: Counter[str],
    ) -> dict[str, Any]:
        count = max(1, len(impressions))
        purchase_rate = sum(int(row["purchased"]) for row in impressions) / count
        click_rate = sum(int(row["clicked"]) for row in impressions) / count
        expected_purchase_rate = sum(
            float(row["ground_truth_purchase_probability"])
            for row in impressions
        ) / count
        brier = sum(
            (
                float(row["ground_truth_purchase_probability"])
                - int(row["purchased"])
            )
            ** 2
            for row in impressions
        ) / count
        log_loss = -sum(
            int(row["purchased"])
            * math.log(max(float(row["ground_truth_purchase_probability"]), 1e-8))
            + (1 - int(row["purchased"]))
            * math.log(
                max(1 - float(row["ground_truth_purchase_probability"]), 1e-8)
            )
            for row in impressions
        ) / count
        sorted_by_affinity = sorted(
            impressions, key=lambda row: row["causal_components"]["affinity"]
        )
        decile = max(1, count // 10)
        low_rate = sum(
            int(row["purchased"]) for row in sorted_by_affinity[:decile]
        ) / decile
        high_rate = sum(
            int(row["purchased"]) for row in sorted_by_affinity[-decile:]
        ) / decile
        low_expected_rate = sum(
            float(row["ground_truth_purchase_probability"])
            for row in sorted_by_affinity[:decile]
        ) / decile
        high_expected_rate = sum(
            float(row["ground_truth_purchase_probability"])
            for row in sorted_by_affinity[-decile:]
        ) / decile
        splits = Counter(str(row["split"]) for row in impressions)
        state_fields = (
            "currency_balance",
            "progression_need",
            "recent_failure_intensity",
            "inventory_overlap",
            "event_urgency",
            "purchase_cooldown",
            "current_goals",
            "owned_item_ids",
        )
        state_coverage = {
            field: sum(
                field in row.get("game_state", {})
                for row in impressions
            )
            / count
            for field in state_fields
        }
        state_unique_values = {
            field: len(
                {
                    json.dumps(
                        row["game_state"].get(field),
                        sort_keys=True,
                    )
                    for row in impressions
                }
            )
            for field in state_fields
        }
        counterfactual_pass_rates = {
            name.removesuffix(":total"): round(
                counterfactual_checks[
                    f"{name.removesuffix(':total')}:passed"
                ]
                / max(1, total),
                8,
            )
            for name, total in counterfactual_checks.items()
            if name.endswith(":total")
        }
        path_lengths = [
            len(row.get("observed_action_path", ()))
            for row in impressions
        ]
        path_signatures = Counter(
            " → ".join(row.get("observed_action_path", ()))
            for row in impressions
        )
        quality_gates = {
            "non_degenerate_purchase_rate": 0.005 < purchase_rate < 0.40,
            "affinity_direction": high_expected_rate > low_expected_rate,
            "all_temporal_splits_present": all(
                splits.get(name, 0) > 0 for name in ("train", "validation", "test")
            ),
            "complete_game_state": all(
                value == 1.0 for value in state_coverage.values()
            ),
            "non_degenerate_game_state": all(
                state_unique_values[field] > 1
                for field in (
                    "currency_balance",
                    "progression_need",
                    "recent_failure_intensity",
                    "inventory_overlap",
                    "event_urgency",
                    "purchase_cooldown",
                    "owned_item_ids",
                )
            ),
            "counterfactual_direction": all(
                value == 1.0
                for value in counterfactual_pass_rates.values()
            ),
            "long_action_paths_present": any(
                length >= 4 for length in path_lengths
            ),
            "diverse_action_paths": len(path_signatures) >= 8,
        }
        return {
            "counts": {
                "impressions": len(impressions),
                "events": len(events),
            },
            "rates": {
                "click": round(click_rate, 8),
                "purchase": round(purchase_rate, 8),
                "expected_purchase": round(
                    expected_purchase_rate,
                    8,
                ),
                "high_affinity_purchase": round(high_rate, 8),
                "low_affinity_purchase": round(low_rate, 8),
                "high_affinity_expected_purchase": round(
                    high_expected_rate,
                    8,
                ),
                "low_affinity_expected_purchase": round(
                    low_expected_rate,
                    8,
                ),
            },
            "oracle_metrics": {
                "brier": round(brier, 8),
                "log_loss": round(log_loss, 8),
            },
            "splits": dict(splits),
            "state_coverage": {
                key: round(value, 8)
                for key, value in state_coverage.items()
            },
            "state_unique_values": state_unique_values,
            "counterfactual_pass_rates": (
                counterfactual_pass_rates
            ),
            "action_paths": {
                "unique": len(path_signatures),
                "mean_length": round(
                    sum(path_lengths) / max(1, len(path_lengths)),
                    8,
                ),
                "max_length": max(path_lengths, default=0),
                "length_at_least_4_rate": round(
                    sum(length >= 4 for length in path_lengths)
                    / max(1, len(path_lengths)),
                    8,
                ),
                "top_signatures": dict(
                    path_signatures.most_common(20)
                ),
            },
            "quality_gates": {
                **quality_gates,
                "all_passed": all(quality_gates.values()),
            },
        }

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--session-id")
    args = parser.parse_args()
    report = SyntheticDatasetLabeler(
        LabelingConfig(seed=args.seed)
    ).label(args.input, args.output, session_id=args.session_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
