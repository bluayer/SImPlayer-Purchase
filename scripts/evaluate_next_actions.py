from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purchase_behavior_simulator.evaluation import prepare_protocol
from purchase_behavior_simulator.models import SimulationRequest
from purchase_behavior_simulator.next_action_evaluation import (
    evaluate_next_actions,
    evaluate_next_actions_by_slice,
)
from purchase_behavior_simulator.product_needs import resolve_product_need_profile


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")


def action_distributions(row: Mapping[str, Any]) -> Mapping[str, Mapping[str, float]]:
    trace = row.get("trace", {})
    events = trace.get("events", ()) if isinstance(trace, Mapping) else ()
    for stage in ("scoring", "eligibility_short_circuit"):
        for event in reversed(events):
            if (
                isinstance(event, Mapping)
                and event.get("stage") == stage
                and isinstance(event.get("action_distributions"), Mapping)
            ):
                return event["action_distributions"]
    raise ValueError(f"missing scoring action distribution for {row.get('case_id')}")


def case_slice(blind: Mapping[str, Any]) -> dict[str, str]:
    request = SimulationRequest.from_dict(
        blind["payload"]["request"]
    )
    profile = resolve_product_need_profile(request.item).to_dict()
    return {
        "motivation_segment": str(profile["dominant_need"]),
        "product_type": request.item.product_type,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081802)
    parser.add_argument(
        "--exclude-users",
        nargs="*",
        default=(),
    )
    args = parser.parse_args(argv)

    answer_rows = read_jsonl(args.protocol_dir / "answer_key.jsonl")
    blind_rows = {
        str(row["case_id"]): row
        for row in read_jsonl(args.protocol_dir / "blind_cases.jsonl")
    }
    answer_key = {
        str(row["case_id"]): row
        for row in answer_rows
    }
    prediction_rows = {
        str(row["case_id"]): row for row in read_jsonl(args.predictions)
    }

    joined: list[dict[str, Any]] = []
    labels_are_saved = all(
        "observed_initial_state" in row and "observed_next_action" in row
        for row in answer_rows
    )
    if labels_are_saved:
        for case_id, prediction in prediction_rows.items():
            saved = answer_key.get(case_id)
            blind = blind_rows.get(case_id)
            if saved is None or blind is None:
                raise ValueError(f"prediction references unknown case {case_id}")
            joined.append(
                {
                    "case_id": case_id,
                    "observed_initial_state": saved["observed_initial_state"],
                    "observed_next_action": saved["observed_next_action"],
                    "observed_detail_action": saved.get(
                        "observed_detail_action"
                    ),
                    "action_distributions": action_distributions(prediction),
                    **case_slice(blind),
                }
            )
    else:
        if args.data_dir is None:
            raise ValueError(
                "--data-dir is required when the answer key does not contain "
                "saved action labels"
            )
        protocol = prepare_protocol(
            args.data_dir,
            selected_users=10,
            history_fraction=0.5,
            history_limit=50,
            seed=args.seed,
            excluded_user_ids=args.exclude_users,
            compute_natural_metrics=False,
        )
        for case in protocol.cases:
            if case.case_id not in prediction_rows:
                continue
            saved = answer_key.get(case.case_id)
            prediction = prediction_rows.get(case.case_id)
            if saved is None or prediction is None:
                raise ValueError(f"missing case {case.case_id}")
            if int(saved["label"]) != case.label:
                raise ValueError(f"label mismatch for {case.case_id}")
            if (
                abs(
                    float(saved["oracle_probability"])
                    - case.oracle_probability
                )
                > 1e-10
            ):
                raise ValueError(
                    f"protocol reconstruction mismatch for {case.case_id}"
                )
            joined.append(
                {
                    "case_id": case.case_id,
                    "observed_initial_state": case.observed_initial_state,
                    "observed_next_action": case.observed_next_action,
                    "observed_detail_action": case.observed_detail_action,
                    "action_distributions": action_distributions(prediction),
                    **case_slice(blind_rows[case.case_id]),
                }
            )
    if len(joined) != len(prediction_rows):
        raise ValueError("not every prediction could be joined to the protocol")

    report = {
        "protocol": {
            "source": str(args.protocol_dir),
            "data": (
                str(args.data_dir)
                if args.data_dir is not None
                else "not_required_saved_answer_key"
            ),
            "seed": args.seed,
            "excluded_users": list(args.exclude_users),
            "prediction_file": str(args.predictions),
            "action_labels_source": (
                "saved_answer_key"
                if labels_are_saved
                else "deterministic_protocol_reconstruction"
            ),
        },
        **evaluate_next_actions(joined),
        "slices": evaluate_next_actions_by_slice(joined),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "cases.jsonl", joined)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
