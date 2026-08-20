from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purchase_behavior_simulator.dataset_adapter import (
    CanonicalJsonlDatasetAdapter,
    MappedTabularDatasetAdapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: normalize an external dataset into the canonical "
            "SimUSER dataset contract. This command does not call AWS or a model."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    adapter = (
        MappedTabularDatasetAdapter(args.config)
        if args.config
        else CanonicalJsonlDatasetAdapter(
            args.canonical_dir,
            dataset_name=args.dataset_name,
        )
    )
    dataset = adapter.load()
    dataset.write(args.output_dir)
    print(
        json.dumps(
            {
                "phase": "configuration",
                "output_dir": str(args.output_dir),
                "validation": dataset.validate(),
                "aws_called": False,
                "model_called": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
