from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import binary_metrics


REPEATABLE_LIKE_CATEGORIES = frozenset(
    {"currency", "subscription", "convenience"}
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def enrich_predictions(
    predictions: Sequence[Mapping[str, Any]],
    blind_cases: Sequence[Mapping[str, Any]],
    bootstrap_payloads: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    actor_by_case = {
        str(row["case_id"]): str(row["actor_id"])
        for row in blind_cases
    }
    history_by_actor = {
        str(row["observation"]["user_id"]): list(
            row["observation"].get("events", ())
        )
        for row in bootstrap_payloads
    }
    enriched = []
    for source in predictions:
        row = dict(source)
        case_id = str(row["case_id"])
        actor_id = actor_by_case[case_id]
        history = history_by_actor.get(actor_id, [])
        item_id = str(row["item_id"])
        features = dict(row.get("analysis_features", {}))
        item_categories = set(features.get("item_categories", ()))
        exact_events = [
            event
            for event in history
            if str(event.get("item_id")) == item_id
        ]
        category_purchases = [
            event
            for event in history
            if event.get("event_type") == "purchase"
            and item_categories.intersection(event.get("categories", ()))
        ]
        preferences = features.get("category_preferences", {})
        preference_values = [
            float(preferences.get(category, 0.0))
            for category in item_categories
        ]
        budget = features.get("budget_reference")
        price = features.get("price")
        price_to_budget = (
            float(price) / float(budget)
            if price is not None and budget not in (None, 0)
            else None
        )
        enriched.append(
            {
                **row,
                "actor_id": actor_id,
                "diagnostics": {
                    "exact_history_interaction": bool(exact_events),
                    "exact_history_purchase": any(
                        event.get("event_type") == "purchase"
                        for event in exact_events
                    ),
                    "exact_history_event_types": sorted(
                        {
                            str(event.get("event_type"))
                            for event in exact_events
                        }
                    ),
                    "same_category_purchase_count": len(category_purchases),
                    "repeatable_like": bool(
                        item_categories.intersection(
                            REPEATABLE_LIKE_CATEGORIES
                        )
                    ),
                    "mean_category_preference": (
                        sum(preference_values) / len(preference_values)
                        if preference_values
                        else None
                    ),
                    "price_to_budget": price_to_budget,
                },
            }
        )
    return enriched


def signal_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    agent = [
        float(row["result"]["components"]["agent"])
        for row in rows
    ]
    final = [float(row["result"]["probability"]) for row in rows]
    return {
        "count": len(rows),
        "agent_likelihood": binary_metrics(labels, agent),
        "final_purchase_probability": binary_metrics(labels, final),
    }


def summarize_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    strata = {
        "all": list(rows),
        "exact_history_purchase": [
            row
            for row in rows
            if row["diagnostics"]["exact_history_purchase"]
        ],
        "no_exact_history_purchase": [
            row
            for row in rows
            if not row["diagnostics"]["exact_history_purchase"]
        ],
        "exact_interaction_without_purchase": [
            row
            for row in rows
            if row["diagnostics"]["exact_history_interaction"]
            and not row["diagnostics"]["exact_history_purchase"]
        ],
        "no_exact_history_interaction": [
            row
            for row in rows
            if not row["diagnostics"]["exact_history_interaction"]
        ],
        "same_category_purchase": [
            row
            for row in rows
            if row["diagnostics"]["same_category_purchase_count"] > 0
        ],
        "no_same_category_purchase": [
            row
            for row in rows
            if row["diagnostics"]["same_category_purchase_count"] == 0
        ],
        "repeatable_like": [
            row for row in rows if row["diagnostics"]["repeatable_like"]
        ],
        "one_time_like": [
            row for row in rows if not row["diagnostics"]["repeatable_like"]
        ],
    }
    by_user: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_user[str(row["original_user_id"])].append(row)

    false_positives = sorted(
        (
            row
            for row in rows
            if int(row["label"]) == 0
        ),
        key=lambda row: float(row["result"]["components"]["agent"]),
        reverse=True,
    )[:10]
    false_negatives = sorted(
        (
            row
            for row in rows
            if int(row["label"]) == 1
        ),
        key=lambda row: float(row["result"]["components"]["agent"]),
    )[:10]

    return {
        "strata": {
            name: signal_metrics(selected)
            for name, selected in strata.items()
            if selected
        },
        "users": {
            user_id: signal_metrics(selected)
            for user_id, selected in sorted(by_user.items())
        },
        "top_agent_false_positives": [
            diagnostic_case(row) for row in false_positives
        ],
        "top_agent_false_negatives": [
            diagnostic_case(row) for row in false_negatives
        ],
    }


def select_tuning_case_ids(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    selected = [
        str(row["case_id"])
        for row in rows
        if row["diagnostics"]["exact_history_purchase"]
    ]
    selected_set = set(selected)
    high_false_positives = sorted(
        (
            row
            for row in rows
            if int(row["label"]) == 0
            and not row["diagnostics"]["exact_history_purchase"]
        ),
        key=lambda row: float(row["result"]["components"]["agent"]),
        reverse=True,
    )
    supported_positives = sorted(
        (
            row
            for row in rows
            if int(row["label"]) == 1
            and float(row["oracle_probability"]) >= 0.15
        ),
        key=lambda row: float(row["result"]["components"]["agent"]),
    )
    for candidates in (high_false_positives, supported_positives):
        added = 0
        for row in candidates:
            case_id = str(row["case_id"])
            if case_id in selected_set:
                continue
            selected.append(case_id)
            selected_set.add(case_id)
            added += 1
            if added == 4:
                break
    return tuple(selected)


def diagnostic_case(row: Mapping[str, Any]) -> dict[str, Any]:
    result = row["result"]
    return {
        "case_id": row["case_id"],
        "label": row["label"],
        "agent_likelihood": result["components"]["agent"],
        "final_probability": result["probability"],
        "diagnostics": row["diagnostics"],
        "item_categories": row["analysis_features"]["item_categories"],
        "price": row["analysis_features"]["price"],
        "budget_reference": row["analysis_features"]["budget_reference"],
        "reasons": result["reasons"],
        "contradictions": result["contradictions"],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Sol full evaluation diagnostics",
        "",
        "> This is a diagnostic/dev evaluation on independently sampled "
        "synthetic outcomes, not real customer purchases.",
        "",
        "| Stratum | Cases | Positive rate | Agent AUC | Agent Brier | "
        "Final AUC | Final Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["strata"].items():
        agent = values["agent_likelihood"]
        final = values["final_purchase_probability"]
        lines.append(
            f"| {name} | {values['count']} | "
            f"{agent['observed_rate']:.3f} | {agent['roc_auc']:.4f} | "
            f"{agent['brier']:.5f} | {final['roc_auc']:.4f} | "
            f"{final['brier']:.5f} |"
        )
    lines.extend(("", "## Highest-scored negatives", ""))
    for row in report["top_agent_false_positives"][:5]:
        lines.append(
            f"- `{row['case_id']}` agent={row['agent_likelihood']:.3f}, "
            f"exact_purchase={row['diagnostics']['exact_history_purchase']}, "
            f"categories={','.join(row['item_categories'])}"
        )
    lines.extend(("", "## Lowest-scored positives", ""))
    for row in report["top_agent_false_negatives"][:5]:
        lines.append(
            f"- `{row['case_id']}` agent={row['agent_likelihood']:.3f}, "
            f"exact_purchase={row['diagnostics']['exact_history_purchase']}, "
            f"categories={','.join(row['item_categories'])}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = enrich_predictions(
        read_jsonl(args.evaluation_dir / "predictions.jsonl"),
        read_jsonl(args.protocol_dir / "blind_cases.jsonl"),
        read_jsonl(args.protocol_dir / "bootstrap.jsonl"),
    )
    report = summarize_diagnostics(rows)
    tuning_case_ids = select_tuning_case_ids(rows)
    (args.evaluation_dir / "diagnostic-cases.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    (args.evaluation_dir / "diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.evaluation_dir / "DIAGNOSTICS.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    (args.evaluation_dir / "tuning-case-ids.txt").write_text(
        "\n".join(tuning_case_ids) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
