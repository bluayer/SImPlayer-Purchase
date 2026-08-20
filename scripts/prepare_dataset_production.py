from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purchase_behavior_simulator.dataset_adapter import (
    CanonicalJsonlDatasetAdapter,
    ProductionExportBuilder,
    ProductionExportConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3a: create label-free, pseudonymized Memory and Neptune "
            "import artifacts. This command does not deploy AWS resources."
        )
    )
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--since",
        help=(
            "Optional exclusive lower bound for an incremental artifact. "
            "The range is since < timestamp <= as-of."
        ),
    )
    parser.add_argument(
        "--identity-salt-env",
        default="PURCHASE_BEHAVIOR_IDENTITY_SALT",
        help="Environment variable containing the HMAC identity salt.",
    )
    parser.add_argument("--history-limit-per-user", type=int)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Only for local smoke tests. Do not use for a production deployment.",
    )
    args = parser.parse_args()

    salt = os.environ.get(args.identity_salt_env, "")
    dataset = CanonicalJsonlDatasetAdapter(args.canonical_dir).load()
    report = ProductionExportBuilder(
        dataset,
        ProductionExportConfig(
            as_of=datetime.fromisoformat(args.as_of.replace("Z", "+00:00")),
            since=(
                datetime.fromisoformat(args.since.replace("Z", "+00:00"))
                if args.since
                else None
            ),
            identity_salt=salt,
            history_limit_per_user=args.history_limit_per_user,
            allow_synthetic=args.allow_synthetic,
        ),
    ).write(args.output_dir)
    print(
        json.dumps(
            {
                "phase": "production_artifact_preparation",
                "output_dir": str(args.output_dir),
                "aws_called": False,
                "deployed": False,
                **report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
