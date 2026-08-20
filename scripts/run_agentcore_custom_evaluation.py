from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from typing import Sequence

from bedrock_agentcore.evaluation import EvaluationClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an AgentCore custom evaluator against one traced session."
    )
    parser.add_argument(
        "--evaluator-id",
        default=os.getenv("AGENTCORE_EVALUATOR_ID"),
        help="Custom evaluator ID or ARN. Defaults to AGENTCORE_EVALUATOR_ID.",
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--agent-id",
        default=os.getenv("AGENTCORE_RUNTIME_ID"),
        help="Runtime ID. Defaults to AGENTCORE_RUNTIME_ID.",
    )
    parser.add_argument(
        "--log-group-name",
        help="Use an explicit CloudWatch log group instead of deriving it from agent ID.",
    )
    parser.add_argument("--trace-id")
    parser.add_argument("--look-back-hours", type=float, default=24.0)
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.evaluator_id:
        parser.error("--evaluator-id or AGENTCORE_EVALUATOR_ID is required")
    if not args.agent_id and not args.log_group_name:
        parser.error(
            "--agent-id/AGENTCORE_RUNTIME_ID or --log-group-name is required"
        )
    if args.look_back_hours <= 0:
        parser.error("--look-back-hours must be positive")

    client = EvaluationClient(region_name=args.region)
    results = client.run(
        evaluator_ids=[args.evaluator_id],
        session_id=args.session_id,
        agent_id=args.agent_id,
        log_group_name=args.log_group_name,
        trace_id=args.trace_id,
        look_back_time=timedelta(hours=args.look_back_hours),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
