from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_action_stages import main as evaluate_stages
from scripts.evaluate_action_stages import trace_uses_fallback
from scripts.evaluate_next_actions import main as evaluate_next_actions
from scripts.evaluate_rollout_local import (
    main as evaluate_simulations,
    read_jsonl,
    read_resume_rows,
)


def has_complete_checkpoint(
    protocol_dir: Path,
    simulation_dir: Path,
) -> bool:
    expected_ids = {
        str(row["case_id"])
        for row in read_jsonl(protocol_dir / "blind_cases.jsonl")
    }
    completed_ids = {
        str(row["case_id"]) for row in read_resume_rows(simulation_dir)
    }
    return bool(expected_ids) and completed_ids == expected_ids


def write_retry_case_ids(
    predictions: Path,
    output: Path,
) -> tuple[str, ...]:
    case_ids = tuple(
        str(row["case_id"])
        for row in read_jsonl(predictions)
        if row.get("result") is None or trace_uses_fallback(row)
    )
    if case_ids:
        output.write_text("\n".join(case_ids) + "\n", encoding="utf-8")
    elif output.exists():
        output.unlink()
    return case_ids


def replace_prediction_rows(
    predictions: Path,
    replacements: Path,
) -> None:
    replacement_by_case = {
        str(row["case_id"]): row for row in read_jsonl(replacements)
    }
    rows = list(read_jsonl(predictions))
    known_case_ids = {str(row["case_id"]) for row in rows}
    unknown_case_ids = set(replacement_by_case).difference(known_case_ids)
    if unknown_case_ids:
        raise ValueError(
            "retry output contains cases outside the main prediction set: "
            + ", ".join(sorted(unknown_case_ids)[:5])
        )
    predictions.write_text(
        "".join(
            json.dumps(
                replacement_by_case.get(str(row["case_id"]), row),
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete purchase behavior simulator evaluation with "
            "checkpoint/resume support."
        )
    )
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Canonical dataset used only when the protocol answer key does "
            "not already contain saved action labels."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID enabled in the AWS account running the evaluation.",
    )
    parser.add_argument("--seed", type=int, default=2026081802)
    parser.add_argument("--exclude-users", nargs="*", default=())
    parser.add_argument(
        "--confirm-model-cost",
        action="store_true",
        help="Required for an unbounded model-backed full-suite run.",
    )
    parser.add_argument(
        "--fallback-retries",
        type=int,
        default=0,
        help=(
            "Retry failed or fallback cases this many times, replace them in "
            "the main checkpoint, and regenerate reports. Additional model "
            "calls may incur cost."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.fallback_retries < 0:
        raise SystemExit("--fallback-retries must be zero or positive")
    simulation_dir = args.output_dir / "simulation"
    if (
        args.limit is None
        and args.case_ids_file is None
        and not args.confirm_model_cost
        and not has_complete_checkpoint(args.protocol_dir, simulation_dir)
    ):
        raise SystemExit(
            "Refusing an unbounded model-backed run without "
            "--confirm-model-cost. Use --limit 1 for smoke."
        )
    action_dir = args.output_dir / "action-metrics"
    predictions = simulation_dir / "predictions.jsonl"

    simulation_args = [
        "--protocol-dir",
        str(args.protocol_dir),
        "--output-dir",
        str(simulation_dir),
        "--workers",
        str(args.workers),
        "--model-id",
        args.model_id,
    ]
    if args.case_ids_file is not None:
        simulation_args.extend(
            ["--case-ids-file", str(args.case_ids_file)]
        )
    if args.limit is not None:
        simulation_args.extend(["--limit", str(args.limit)])
    evaluate_simulations(simulation_args)

    retry_case_file = args.output_dir / "retry-case-ids.txt"
    retry_case_ids = write_retry_case_ids(predictions, retry_case_file)
    for attempt in range(1, args.fallback_retries + 1):
        if not retry_case_ids:
            break
        retry_root = (
            args.output_dir
            / "retries"
            / f"attempt-{attempt}-{uuid4().hex[:8]}"
        )
        retry_simulation_dir = retry_root / "simulation"
        retry_args = [
            "--protocol-dir",
            str(args.protocol_dir),
            "--output-dir",
            str(retry_simulation_dir),
            "--case-ids-file",
            str(retry_case_file),
            "--workers",
            str(args.workers),
            "--model-id",
            args.model_id,
        ]
        evaluate_simulations(retry_args)
        replace_prediction_rows(
            predictions,
            retry_simulation_dir / "predictions.jsonl",
        )
        # Rebuild the main simulation report from the replaced checkpoint
        # without issuing another model call.
        evaluate_simulations(simulation_args)
        retry_case_ids = write_retry_case_ids(
            predictions,
            retry_case_file,
        )

    next_action_args = [
        "--protocol-dir",
        str(args.protocol_dir),
        "--predictions",
        str(predictions),
        "--output-dir",
        str(action_dir),
        "--seed",
        str(args.seed),
    ]
    if args.data_dir is not None:
        next_action_args.extend(["--data-dir", str(args.data_dir)])
    if args.exclude_users:
        next_action_args.append("--exclude-users")
        next_action_args.extend(args.exclude_users)
    evaluate_next_actions(next_action_args)

    evaluate_stages(
        [
            "--protocol-dir",
            str(args.protocol_dir),
            "--predictions",
            str(predictions),
            "--output",
            str(action_dir / "stage-report-advisory.json"),
        ]
    )
    if retry_case_ids:
        print(
            "Transient fallback cases require retry: "
            + ", ".join(retry_case_ids)
        )


if __name__ == "__main__":
    main()
