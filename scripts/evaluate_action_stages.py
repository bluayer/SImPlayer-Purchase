from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purchase_behavior_simulator.next_action_evaluation import (
    evaluate_next_actions,
)


STAGES = {
    "actor": ("action_assessment_round", ()),
    "grounding": ("transition_grounding", ("scoring",)),
    "commitment": (
        "commitment_gate",
        ("transition_grounding", "scoring"),
    ),
    "critic": ("action_validator", ("scoring",)),
    "counterfactual": (
        "counterfactual_validator",
        ("action_validator", "scoring"),
    ),
    "final": ("scoring", ()),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stage_distribution(
    row: Mapping[str, Any],
    stage: str,
    *,
    fallback_stages: tuple[str, ...] = (),
) -> Mapping[str, Mapping[str, float]]:
    events = row.get("trace", {}).get("events", ())
    for event in reversed(events):
        if (
            isinstance(event, Mapping)
            and event.get("stage") == "eligibility_short_circuit"
            and isinstance(event.get("action_distributions"), Mapping)
        ):
            return event["action_distributions"]
    for candidate_stage in (stage, *fallback_stages):
        for event in reversed(events):
            if (
                isinstance(event, Mapping)
                and event.get("stage") == candidate_stage
                and isinstance(event.get("action_distributions"), Mapping)
            ):
                return event["action_distributions"]
    result = row.get("result", {})
    if isinstance(result, Mapping) and isinstance(
        result.get("action_distributions"),
        Mapping,
    ):
        return result["action_distributions"]
    raise ValueError(f"missing {stage} distributions for {row.get('case_id')}")


def trace_uses_fallback(row: Mapping[str, Any]) -> bool:
    def visit(value: Any) -> bool:
        if isinstance(value, Mapping):
            if value.get("fallback") is True:
                return True
            if value.get("fallback_used") is True:
                return True
            return any(visit(child) for child in value.values())
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            return any(visit(child) for child in value)
        return False

    return visit(row.get("trace", {}))


def hard_constraint_distribution(
    row: Mapping[str, Any],
) -> Mapping[str, Mapping[str, float]]:
    events = row.get("trace", {}).get("events", ())
    validator = next(
        (
            event
            for event in reversed(events)
            if isinstance(event, Mapping)
            and event.get("stage") == "action_validator"
        ),
        None,
    )
    if validator is None:
        return stage_distribution(
            row,
            "transition_grounding",
            fallback_stages=("scoring",),
        )
    checks = validator.get("checks", {})
    if (
        checks.get("repeat_purchase_valid") is False
        or checks.get("price_budget_consistent") is False
    ):
        return validator["action_distributions"]
    return stage_distribution(row, "transition_grounding")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    answers = {
        str(row["case_id"]): row
        for row in read_jsonl(args.protocol_dir / "answer_key.jsonl")
    }
    predictions = read_jsonl(args.predictions)
    reports: dict[str, Any] = {}
    for name, (trace_stage, fallback_stages) in STAGES.items():
        joined = []
        for prediction in predictions:
            case_id = str(prediction["case_id"])
            answer = answers.get(case_id)
            if answer is None:
                raise ValueError(f"missing answer for {case_id}")
            joined.append(
                {
                    "case_id": case_id,
                    "observed_initial_state": answer[
                        "observed_initial_state"
                    ],
                    "observed_next_action": answer["observed_next_action"],
                    "observed_detail_action": answer.get(
                        "observed_detail_action"
                    ),
                    "observed_action_path": answer.get(
                        "observed_action_path",
                        (),
                    ),
                    "action_distributions": stage_distribution(
                        prediction,
                        trace_stage,
                        fallback_stages=fallback_stages,
                    ),
                }
            )
        reports[name] = evaluate_next_actions(joined)
    hard_policy_rows = []
    for prediction in predictions:
        case_id = str(prediction["case_id"])
        answer = answers[case_id]
        hard_policy_rows.append(
            {
                "case_id": case_id,
                "observed_initial_state": answer["observed_initial_state"],
                "observed_next_action": answer["observed_next_action"],
                "observed_detail_action": answer.get(
                    "observed_detail_action"
                ),
                "action_distributions": hard_constraint_distribution(
                    prediction
                ),
            }
        )
    reports["hard_constraint_policy"] = evaluate_next_actions(hard_policy_rows)
    reports["advisory_critic_policy"] = reports["grounding"]

    output = {
        "protocol": str(args.protocol_dir),
        "predictions": str(args.predictions),
        "fallback_case_ids": [
            str(prediction["case_id"])
            for prediction in predictions
            if trace_uses_fallback(prediction)
        ],
        "stages": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                name: {
                    "primary_metrics": report["primary_metrics"],
                    "proper_scoring": report["proper_scoring"],
                    "argmax_purchase_f1": report["argmax_diagnostic"][
                        "terminal_purchase"
                    ]["f1"],
                }
                for name, report in reports.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
