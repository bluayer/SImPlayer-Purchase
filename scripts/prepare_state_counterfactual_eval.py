from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purchase_behavior_simulator.state_counterfactuals import (
    build_state_counterfactual_pairs,
    read_jsonl,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create paired state perturbations from a state-rich blind "
            "evaluation protocol. This command does not call AWS or a model."
        )
    )
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-cases", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    blind_cases = read_jsonl(
        args.protocol_dir / "blind_cases.jsonl"
    )
    pairs, report = build_state_counterfactual_pairs(
        blind_cases,
        base_case_limit=args.base_cases,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "blind_pairs.jsonl",
        pairs,
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
