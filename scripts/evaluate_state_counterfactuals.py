from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_rollout_local import (
    SnapshotMemoryProvider,
    build_service,
    read_jsonl,
    rescore_result,
    sanitize_legacy_observation,
)
from purchase_behavior_simulator.models import (
    ObservationBatch,
    SimulationRequest,
)


def select_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_dimension: int | None,
) -> list[Mapping[str, Any]]:
    if per_dimension is None:
        return list(rows)
    selected: list[Mapping[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        dimension = str(row["dimension"])
        if counts.get(dimension, 0) >= per_dimension:
            continue
        selected.append(row)
        counts[dimension] = counts.get(dimension, 0) + 1
    return selected


def rollout_purchase_probability(result: Mapping[str, Any]) -> float:
    components = result.get("components", {})
    return float(components.get("rollout", result["probability"]))


def summarize_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dimensions: dict[str, dict[str, int]] = {}
    for row in rows:
        dimension = str(row["dimension"])
        bucket = dimensions.setdefault(
            dimension,
            {"pairs": 0, "passed": 0},
        )
        bucket["pairs"] += 1
        bucket["passed"] += int(bool(row["passed"]))
    for bucket in dimensions.values():
        bucket["pass_rate"] = round(
            bucket["passed"] / max(1, bucket["pairs"]),
            8,
        )
    passed = sum(bool(row["passed"]) for row in rows)
    return {
        "pairs": len(rows),
        "passed": passed,
        "pass_rate": round(passed / max(1, len(rows)), 8),
        "dimensions": dimensions,
    }


def summarize_saved_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rescored_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        favorable = dict(row["favorable"])
        adverse = dict(row["adverse"])
        favorable["result"] = rescore_result(favorable["result"])
        adverse["result"] = rescore_result(adverse["result"])
        favorable_probability = float(
            favorable["result"]["probability"]
        )
        adverse_probability = float(adverse["result"]["probability"])
        updated.update(
            {
                "favorable": favorable,
                "adverse": adverse,
                "difference": round(
                    favorable_probability - adverse_probability,
                    8,
                ),
                "passed": (
                    favorable_probability + 1e-9
                    >= adverse_probability
                ),
            }
        )
        rescored_rows.append(updated)
    return rescored_rows, summarize_pairs(rescored_rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired GameState changes with the same user, item, "
            "history, and model."
        )
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--per-dimension", type=int)
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Re-score the existing predictions.jsonl without model calls.",
    )
    args = parser.parse_args(argv)

    prediction_path = args.output_dir / "predictions.jsonl"
    if args.reuse_predictions:
        output_rows, summary = summarize_saved_predictions(
            read_jsonl(prediction_path)
        )
        report = {
            "model_id": args.model_id,
            "answer_key_sent_to_model": False,
            "checkpoint_results_rescored": True,
            "summary": summary,
        }
        prediction_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in output_rows
            ),
            encoding="utf-8",
        )
        (args.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    pairs = select_pairs(
        read_jsonl(args.pairs),
        per_dimension=args.per_dimension,
    )
    if not pairs:
        raise ValueError("no counterfactual pairs selected")
    batches = {
        str(row["observation"]["user_id"]): ObservationBatch.from_dict(
            sanitize_legacy_observation(row["observation"])
        )
        for row in read_jsonl(args.protocol_dir / "bootstrap.jsonl")
    }
    memory_provider = SnapshotMemoryProvider(batches)
    output_rows: list[dict[str, Any]] = []
    for pair in pairs:
        side_results: dict[str, Any] = {}
        for side in ("favorable", "adverse"):
            trace_events: list[dict[str, Any]] = []
            started = time.monotonic()
            payload = pair[f"{side}_payload"]
            request = SimulationRequest.from_dict(payload["request"])
            result = build_service(
                model_id=args.model_id,
                memory_provider=memory_provider,
                trace_events=trace_events,
            ).evaluate_snapshot(request).to_dict()
            result = rescore_result(result)
            side_results[side] = {
                "result": result,
                "purchase_probability": float(result["probability"]),
                "rollout_purchase_probability": (
                    rollout_purchase_probability(result)
                ),
                "latency_seconds": round(
                    time.monotonic() - started,
                    6,
                ),
                "trace": {
                    "trace_kind": "observable_structured",
                    "raw_chain_of_thought_captured": False,
                    "events": trace_events,
                },
            }
        favorable = side_results["favorable"]["purchase_probability"]
        adverse = side_results["adverse"]["purchase_probability"]
        favorable_rollout = side_results["favorable"][
            "rollout_purchase_probability"
        ]
        adverse_rollout = side_results["adverse"][
            "rollout_purchase_probability"
        ]
        output_rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "base_case_id": str(pair["base_case_id"]),
                "dimension": str(pair["dimension"]),
                "expected_relation": str(pair["expected_relation"]),
                **side_results,
                "difference": round(favorable - adverse, 8),
                "passed": favorable + 1e-9 >= adverse,
                "rollout_difference": round(
                    favorable_rollout - adverse_rollout,
                    8,
                ),
                "rollout_passed": (
                    favorable_rollout + 1e-9 >= adverse_rollout
                ),
            }
        )
        print(
            json.dumps(
                {
                    "completed": len(output_rows),
                    "total": len(pairs),
                    "pair_id": pair["pair_id"],
                    "passed": output_rows[-1]["passed"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    report = {
        "model_id": args.model_id,
        "answer_key_sent_to_model": False,
        "summary": summarize_pairs(output_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in output_rows
        ),
        encoding="utf-8",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
