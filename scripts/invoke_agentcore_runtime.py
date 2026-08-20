from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.deploy_agentcore import (
    DeploymentSettings,
    find_output,
    invoke_with_retries,
    stack_outputs,
    verify_aws_account,
    wait_ready,
)


LOCAL_DEPLOYMENT_CONFIG = (
    ROOT / "deployment" / "agentcore" / "deployment.local.json"
)


def runtime_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if "operation" in value:
        return dict(value)
    return {"operation": "simulate", "request": dict(value)}


def resolve_runtime(
    *,
    settings: DeploymentSettings,
    target: str,
    runtime_arn: str | None,
) -> tuple[str, str | None]:
    if runtime_arn:
        return runtime_arn, None
    outputs = stack_outputs(
        settings.region,
        f"AgentCore-BehaviorSim-{target}",
    )
    runtime_id = find_output(
        outputs,
        "PurchaseBehaviorSimulatorRuntimeIdOutput",
    )
    resolved_arn = find_output(
        outputs,
        "PurchaseBehaviorSimulatorRuntimeArnOutput",
    )
    wait_ready(runtime_id, settings.region)
    return resolved_arn, runtime_id


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Invoke a deployed SimPlayer Purchase AgentCore Runtime with a "
            "JSON request."
        )
    )
    parser.add_argument("payload", type=Path)
    parser.add_argument("--target", default="default")
    parser.add_argument("--config", type=Path, default=LOCAL_DEPLOYMENT_CONFIG)
    parser.add_argument("--runtime-arn")
    parser.add_argument("--user-id")
    parser.add_argument("--session-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    settings = DeploymentSettings.from_path(args.config)
    verify_aws_account(settings)
    payload_value = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(payload_value, Mapping):
        parser.error("payload must contain a JSON object")
    payload = runtime_payload(payload_value)
    request = payload.get("request", {})
    request_user = (
        request.get("user", {}).get("user_id")
        if isinstance(request, Mapping)
        and isinstance(request.get("user"), Mapping)
        else None
    )
    user_id = args.user_id or str(request_user or f"invoke-{uuid4().hex}")
    session_id = args.session_id or (
        f"simplayer-invoke-{uuid4().hex}-{uuid4().hex[:8]}"
    )
    runtime_arn, runtime_id = resolve_runtime(
        settings=settings,
        target=args.target,
        runtime_arn=args.runtime_arn,
    )
    result = invoke_with_retries(
        runtime_arn=runtime_arn,
        payload=payload,
        user_id=user_id,
        session_id=session_id,
        region=settings.region,
    )
    output = {
        "target": args.target,
        "runtime_id": runtime_id,
        "runtime_arn": runtime_arn,
        "user_id": user_id,
        "session_id": session_id,
        "result": result,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
