from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .synthetic_oracle import REPEATABLE_CATEGORIES


REQUIRED_GAME_STATE_FIELDS = frozenset(
    {
        "currency_balance",
        "progression_need",
        "recent_failure_intensity",
        "inventory_overlap",
        "event_urgency",
        "purchase_cooldown",
        "current_goals",
        "owned_item_ids",
    }
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def build_state_counterfactual_pairs(
    blind_cases: Sequence[Mapping[str, Any]],
    *,
    base_case_limit: int = 50,
    seed: int = 20260820,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid: list[Mapping[str, Any]] = []
    missing: Counter[str] = Counter()
    for case in blind_cases:
        request = case.get("payload", {}).get("request", {})
        state = request.get("game_state", {})
        absent = REQUIRED_GAME_STATE_FIELDS.difference(state)
        if absent:
            missing.update(absent)
            continue
        valid.append(case)
    if not valid:
        raise ValueError(
            "no blind cases contain a complete GameStateSnapshot"
        )

    # Deterministic sampling is intentional for reproducible evaluation fixtures.
    rng = random.Random(seed)  # nosec B311
    nonrepeatable = [
        case
        for case in valid
        if not REPEATABLE_CATEGORIES.intersection(
            str(value)
            for value in case["payload"]["request"]["item"].get(
                "categories",
                (),
            )
        )
    ]
    nonrepeatable_target = min(
        len(nonrepeatable),
        max(1, base_case_limit // 2),
    )
    selected_nonrepeatable = rng.sample(
        nonrepeatable,
        nonrepeatable_target,
    )
    selected_ids = {
        str(case["case_id"])
        for case in selected_nonrepeatable
    }
    remaining = [
        case
        for case in valid
        if str(case["case_id"]) not in selected_ids
    ]
    remaining_target = min(
        len(remaining),
        max(0, base_case_limit - len(selected_nonrepeatable)),
    )
    selected = [
        *selected_nonrepeatable,
        *rng.sample(remaining, remaining_target),
    ]
    selected.sort(key=lambda row: str(row["case_id"]))

    pairs: list[dict[str, Any]] = []
    dimension_counts: Counter[str] = Counter()
    for case in selected:
        request = case["payload"]["request"]
        item = request["item"]
        state = request["game_state"]
        price = float(item.get("price", 0.0)) * (
            1.0 - float(item.get("discount_rate", 0.0))
        )
        perturbations: list[
            tuple[
                str,
                Mapping[str, Any],
                Mapping[str, Any],
                Mapping[str, Any],
                Mapping[str, Any],
            ]
        ] = [
            (
                "currency_balance",
                {"currency_balance": 3.0 * max(1.0, price)},
                {"currency_balance": 0.1 * max(1.0, price)},
                {},
                {},
            ),
            (
                "progression_need",
                {
                    "progression_need": 1.0,
                    "recent_failure_intensity": max(
                        0.7,
                        float(state["recent_failure_intensity"]),
                    ),
                },
                {
                    "progression_need": 0.0,
                    "recent_failure_intensity": 0.0,
                },
                {},
                {},
            ),
            (
                "event_urgency",
                {"event_urgency": 1.0},
                {"event_urgency": 0.0},
                {},
                {},
            ),
            (
                "purchase_cooldown",
                {"purchase_cooldown": 0.0},
                {"purchase_cooldown": 1.0},
                {},
                {},
            ),
            (
                "price",
                {},
                {},
                {"price": 0.70 * float(item.get("price", 0.0))},
                {"price": 1.50 * float(item.get("price", 0.0))},
            ),
        ]
        categories = {
            str(value) for value in item.get("categories", ())
        }
        if not REPEATABLE_CATEGORIES.intersection(categories):
            owned = [
                str(value)
                for value in state["owned_item_ids"]
            ]
            perturbations.append(
                (
                    "ownership",
                    {
                        "inventory_overlap": 0.0,
                        "owned_item_ids": [
                            value
                            for value in owned
                            if value != item["item_id"]
                        ],
                    },
                    {
                        "inventory_overlap": 1.0,
                        "owned_item_ids": sorted(
                            {*owned, str(item["item_id"])}
                        ),
                    },
                    {},
                    {},
                )
            )

        for (
            dimension,
            favorable_state,
            adverse_state,
            favorable_item,
            adverse_item,
        ) in perturbations:
            favorable = copy.deepcopy(case["payload"])
            adverse = copy.deepcopy(case["payload"])
            favorable_request = favorable["request"]
            adverse_request = adverse["request"]
            favorable_request["game_state"].update(
                favorable_state
            )
            adverse_request["game_state"].update(adverse_state)
            favorable_request["item"].update(favorable_item)
            adverse_request["item"].update(adverse_item)
            pair_id = f"{case['case_id']}:{dimension}"
            favorable_request["request_id"] = f"{pair_id}:favorable"
            adverse_request["request_id"] = f"{pair_id}:adverse"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "base_case_id": case["case_id"],
                    "dimension": dimension,
                    "expected_relation": (
                        "favorable_purchase_probability_gte_adverse"
                    ),
                    "favorable_payload": favorable,
                    "adverse_payload": adverse,
                }
            )
            dimension_counts[dimension] += 1

    report = {
        "schema": "purchase-behavior.state-counterfactual.v1",
        "seed": seed,
        "available_base_cases": len(valid),
        "selected_base_cases": len(selected),
        "pairs": len(pairs),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "excluded_incomplete_cases": len(blind_cases) - len(valid),
        "missing_field_counts": dict(sorted(missing.items())),
        "model_called": False,
        "quality_gates": {
            "complete_game_state": len(valid) == len(blind_cases),
            "all_dimensions_present": all(
                dimension_counts[name] > 0
                for name in (
                    "currency_balance",
                    "progression_need",
                    "event_urgency",
                    "purchase_cooldown",
                    "price",
                    "ownership",
                )
            ),
            "ownership_coverage": (
                dimension_counts["ownership"]
                >= min(len(selected), max(1, base_case_limit // 2))
            ),
        },
    }
    report["quality_gates"]["all_passed"] = all(
        report["quality_gates"].values()
    )
    return pairs, report
