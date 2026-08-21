from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purchase_behavior_simulator.dataset_adapter import (
    CanonicalJsonlDatasetAdapter,
    EvaluationProtocolBuilder,
    ProtocolBuildConfig,
    build_path_coverage_protocol,
)
from purchase_behavior_simulator.evaluation import save_protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: create a leakage-safe internal evaluation protocol from "
            "a canonical dataset. Output includes an inference-isolated "
            "answer key."
        )
    )
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--coverage-output-dir",
        type=Path,
        help=(
            "Optional separate protocol containing rare observable paths. "
            "Its frequencies are diagnostic and not natural-distribution metrics."
        ),
    )
    parser.add_argument("--coverage-cases", type=int, default=100)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--cases-per-user", type=int, default=25)
    parser.add_argument("--history-fraction", type=float, default=0.5)
    parser.add_argument("--history-limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--exclude-users", nargs="*", default=())
    parser.add_argument("--allow-incomplete-exposure", action="store_true")
    parser.add_argument(
        "--require-game-state",
        action="store_true",
        help=(
            "Reject datasets without complete, varying GameStateSnapshot "
            "fields before any model-backed evaluation"
        ),
    )
    args = parser.parse_args()

    dataset = CanonicalJsonlDatasetAdapter(args.canonical_dir).load()
    protocol = EvaluationProtocolBuilder(
        dataset,
        ProtocolBuildConfig(
            selected_users=args.users or None,
            cases_per_user=args.cases_per_user or None,
            history_fraction=args.history_fraction,
            history_limit=args.history_limit,
            seed=args.seed,
            excluded_user_ids=frozenset(args.exclude_users),
            allow_incomplete_exposure=args.allow_incomplete_exposure,
            require_game_state=args.require_game_state,
        ),
    ).build()
    save_protocol(protocol, args.output_dir)
    coverage_protocol = None
    if args.coverage_output_dir is not None:
        coverage_protocol = build_path_coverage_protocol(
            protocol,
            max_cases=args.coverage_cases,
        )
        save_protocol(coverage_protocol, args.coverage_output_dir)
    print(
        json.dumps(
            {
                "phase": "internal_eval_preparation",
                "output_dir": str(args.output_dir),
                "users": protocol.users,
                "cases": len(protocol.cases),
                "purchase_rate": protocol.natural_metrics["protocol"][
                    "purchase_rate"
                ],
                "answer_key_created": True,
                "coverage_output_dir": (
                    str(args.coverage_output_dir)
                    if args.coverage_output_dir is not None
                    else None
                ),
                "coverage_cases": (
                    len(coverage_protocol.cases)
                    if coverage_protocol is not None
                    else 0
                ),
                "game_state": protocol.natural_metrics["protocol"][
                    "validation"
                ]["game_state"],
                "aws_called": False,
                "model_called": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
