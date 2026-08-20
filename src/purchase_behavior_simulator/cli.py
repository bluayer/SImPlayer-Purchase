from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .bootstrap import build_service
from .runtime_api import handle_request
from .synthetic import SyntheticConfig, SyntheticDatasetGenerator
from .synthetic_labeling import LabelingConfig, SyntheticDatasetLabeler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="purchase-behavior-simulator",
        description="Purchase-related user behavior simulator prototype",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser(
        "simulate",
        help="Run a local simulation request from JSON",
    )
    simulate.add_argument("request", type=Path)

    generate = commands.add_parser(
        "generate-synthetic",
        help=(
            "Generate label-free synthetic scenarios and an "
            "inference-isolated oracle"
        ),
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--users", type=int, default=500)
    generate.add_argument("--items", type=int, default=250)
    generate.add_argument("--impressions", type=int, default=100_000)
    generate.add_argument("--days", type=int, default=120)
    generate.add_argument("--seed", type=int, default=20260818)
    generate.add_argument("--session-id")

    label = commands.add_parser(
        "label-synthetic",
        help="Sample observed actions in an isolated labeling session",
    )
    label.add_argument("--input", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)
    label.add_argument("--seed", type=int, default=20260819)
    label.add_argument("--session-id")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "simulate":
        payload = json.loads(args.request.read_text(encoding="utf-8"))
        result = handle_request(payload, service=build_service())
    elif args.command == "generate-synthetic":
        result = SyntheticDatasetGenerator(
            SyntheticConfig(
                users=args.users,
                items=args.items,
                impressions=args.impressions,
                days=args.days,
                seed=args.seed,
            )
        ).generate(args.output, session_id=args.session_id)
    elif args.command == "label-synthetic":
        result = SyntheticDatasetLabeler(
            LabelingConfig(seed=args.seed)
        ).label(args.input, args.output, session_id=args.session_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
