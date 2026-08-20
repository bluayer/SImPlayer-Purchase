from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import json
from pathlib import Path
# Deployment commands use fixed argument lists and never invoke a shell.
import subprocess  # nosec B404
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "deployment" / "agentcore"
AGENTCORE_CONFIG = PROJECT_DIR / "agentcore" / "agentcore.json"
AWS_TARGETS_CONFIG = PROJECT_DIR / "agentcore" / "aws-targets.json"
LOCAL_DEPLOYMENT_CONFIG = PROJECT_DIR / "deployment.local.json"
ACCESS_TEMPLATE = (
    ROOT / "deployment" / "neptune" / "runtime-data-access-policy.yaml"
)
RUNTIME_CLIENT_CONFIG = Config(
    connect_timeout=10,
    read_timeout=180,
    retries={"max_attempts": 0},
)


@dataclass(frozen=True)
class DeploymentSettings:
    account: str
    region: str
    model_id: str
    subnet_ids: tuple[str, ...]
    security_group_ids: tuple[str, ...]
    neptune_endpoint: str
    neptune_cluster_id: str

    @classmethod
    def from_path(cls, path: Path) -> DeploymentSettings:
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is required; copy deployment.example.json and "
                "fill in the target AWS environment"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        settings = cls(
            account=str(payload["account"]),
            region=str(payload["region"]),
            model_id=str(payload["model_id"]),
            subnet_ids=tuple(str(value) for value in payload["subnet_ids"]),
            security_group_ids=tuple(
                str(value) for value in payload["security_group_ids"]
            ),
            neptune_endpoint=str(payload["neptune_endpoint"]),
            neptune_cluster_id=str(payload["neptune_cluster_id"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if len(self.account) != 12 or not self.account.isdigit():
            raise ValueError("deployment account must be a 12-digit AWS account")
        if self.account == "000000000000":
            raise ValueError("deployment account still contains a placeholder")
        if not self.region:
            raise ValueError("deployment region is required")
        if not self.model_id or self.model_id == (
            "model-id-enabled-in-the-target-account"
        ):
            raise ValueError("a model enabled in the target account is required")
        if not self.subnet_ids or not all(
            value.startswith("subnet-") for value in self.subnet_ids
        ):
            raise ValueError("at least one valid subnet ID is required")
        if not self.security_group_ids or not all(
            value.startswith("sg-") for value in self.security_group_ids
        ):
            raise ValueError("at least one valid security group ID is required")
        if (
            not self.neptune_endpoint
            or "replace-me" in self.neptune_endpoint
            or "your-cluster" in self.neptune_endpoint
        ):
            raise ValueError("a deployed Neptune endpoint is required")
        if (
            not self.neptune_cluster_id
            or self.neptune_cluster_id == "your-neptune-cluster"
        ):
            raise ValueError("a deployed Neptune cluster ID is required")


def run(command: Sequence[str], *, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)  # nosec B603


def stack_outputs(region: str, stack_name: str) -> dict[str, str]:
    client = boto3.client("cloudformation", region_name=region)
    stack = client.describe_stacks(StackName=stack_name)["Stacks"][0]
    return {
        output["OutputKey"]: output["OutputValue"]
        for output in stack.get("Outputs", ())
    }


def find_output(outputs: Mapping[str, str], fragment: str) -> str:
    matches = [
        value for key, value in outputs.items() if fragment in key
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one stack output containing {fragment!r}, got {len(matches)}"
        )
    return matches[0]


def memory_strategy_ids(
    client: Any,
    memory_id: str,
) -> tuple[str, str]:
    response = client.get_memory(memoryId=memory_id)
    strategies = response.get("memory", {}).get("strategies", ())
    by_type = {
        str(strategy.get("type", "")).upper(): str(strategy["strategyId"])
        for strategy in strategies
        if strategy.get("status") == "ACTIVE"
    }
    try:
        return by_type["EPISODIC"], by_type["SEMANTIC"]
    except KeyError as exc:
        raise RuntimeError(
            "Memory must expose active EPISODIC and SEMANTIC strategies"
        ) from exc


@contextmanager
def configured_deployment_environment(
    settings: DeploymentSettings,
    *,
    target: str,
):
    original_agentcore = AGENTCORE_CONFIG.read_bytes()
    original_targets = AWS_TARGETS_CONFIG.read_bytes()
    agentcore = json.loads(original_agentcore)
    runtime = next(
        item
        for item in agentcore["runtimes"]
        if item["name"] == "PurchaseBehaviorSimulator"
    )
    environment = {
        item["name"]: item for item in runtime.get("envVars", ())
    }
    environment["AWS_REGION"]["value"] = settings.region
    environment["BEDROCK_MODEL_ID"]["value"] = settings.model_id
    environment["NEPTUNE_ENDPOINT"]["value"] = settings.neptune_endpoint
    runtime["networkConfig"] = {
        "subnets": list(settings.subnet_ids),
        "securityGroups": list(settings.security_group_ids),
    }

    targets = json.loads(original_targets)
    target_config = next(
        (item for item in targets if item.get("name") == target),
        None,
    )
    if target_config is None:
        target_config = {"name": target}
        targets.append(target_config)
    target_config.update(
        {
            "description": "Configured by scripts/deploy_agentcore.py",
            "account": settings.account,
            "region": settings.region,
        }
    )

    AGENTCORE_CONFIG.write_text(
        json.dumps(agentcore, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    AWS_TARGETS_CONFIG.write_text(
        json.dumps(targets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        yield
    finally:
        AGENTCORE_CONFIG.write_bytes(original_agentcore)
        AWS_TARGETS_CONFIG.write_bytes(original_targets)


@contextmanager
def configured_strategy_environment(
    strategy_ids: tuple[str, str],
):
    original = AGENTCORE_CONFIG.read_bytes()
    config = json.loads(original)
    runtime = next(
        item
        for item in config["runtimes"]
        if item["name"] == "PurchaseBehaviorSimulator"
    )
    names = {
        "PURCHASE_BEHAVIOR_EPISODIC_STRATEGY_ID",
        "PURCHASE_BEHAVIOR_TRANSITION_STRATEGY_ID",
    }
    environment = [
        item for item in runtime.get("envVars", ()) if item["name"] not in names
    ]
    environment.extend(
        [
            {
                "name": "PURCHASE_BEHAVIOR_EPISODIC_STRATEGY_ID",
                "value": strategy_ids[0],
            },
            {
                "name": "PURCHASE_BEHAVIOR_TRANSITION_STRATEGY_ID",
                "value": strategy_ids[1],
            },
        ]
    )
    runtime["envVars"] = environment
    AGENTCORE_CONFIG.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        yield
    finally:
        AGENTCORE_CONFIG.write_bytes(original)


def deploy_agentcore(
    *,
    target: str,
    strategy_ids: tuple[str, str] | None,
) -> None:
    context = (
        configured_strategy_environment(strategy_ids)
        if strategy_ids is not None
        else nullcontext()
    )
    with context:
        run(
            ["agentcore", "deploy", "--yes", "--target", target],
            cwd=PROJECT_DIR,
        )


def deployed_strategy_ids(
    *,
    region: str,
    stack_name: str,
) -> tuple[str, tuple[str, str]] | None:
    try:
        outputs = stack_outputs(region, stack_name)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ValidationError":
            return None
        raise
    memory_id = find_output(
        outputs,
        "MemoryPurchaseBehaviorSimulatorMemoryIdOutput",
    )
    control = boto3.client("bedrock-agentcore-control", region_name=region)
    return memory_id, memory_strategy_ids(control, memory_id)


def deploy_access_policy(
    *,
    region: str,
    role_arn: str,
    memory_id: str,
    neptune_cluster_id: str,
    stack_name: str,
) -> None:
    cloudformation = boto3.client("cloudformation", region_name=region)
    try:
        stack = cloudformation.describe_stacks(StackName=stack_name)["Stacks"][0]
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code != "ValidationError":
            raise
    else:
        if stack["StackStatus"] == "ROLLBACK_COMPLETE":
            cloudformation.delete_stack(StackName=stack_name)
            cloudformation.get_waiter("stack_delete_complete").wait(
                StackName=stack_name
            )

    neptune = boto3.client("neptune", region_name=region)
    cluster = neptune.describe_db_clusters(
        DBClusterIdentifier=neptune_cluster_id
    )["DBClusters"][0]
    run(
        [
            "aws",
            "cloudformation",
            "deploy",
            "--stack-name",
            stack_name,
            "--template-file",
            str(ACCESS_TEMPLATE),
            "--parameter-overrides",
            f"RuntimeRoleName={role_arn.rsplit('/', 1)[-1]}",
            f"NeptuneClusterResourceId={cluster['DbClusterResourceId']}",
            f"MemoryId={memory_id}",
            "--capabilities",
            "CAPABILITY_NAMED_IAM",
            "--region",
            region,
        ]
    )


def wait_ready(runtime_id: str, region: str) -> Mapping[str, Any]:
    client = boto3.client("bedrock-agentcore-control", region_name=region)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        runtime = client.get_agent_runtime(agentRuntimeId=runtime_id)
        status = runtime["status"]
        if status == "READY":
            return runtime
        if status in {"CREATE_FAILED", "UPDATE_FAILED", "DELETING"}:
            raise RuntimeError(f"Runtime entered terminal state {status}")
        time.sleep(10)
    raise TimeoutError("Runtime did not become READY within 15 minutes")


def verify_aws_account(settings: DeploymentSettings) -> None:
    actual = str(
        boto3.client("sts", region_name=settings.region)
        .get_caller_identity()["Account"]
    )
    if actual != settings.account:
        raise RuntimeError(
            "AWS credential account does not match deployment.local.json: "
            f"expected {settings.account}, got {actual}"
        )


def invoke(
    *,
    runtime_arn: str,
    payload: Mapping[str, Any],
    user_id: str,
    session_id: str,
    region: str,
) -> Mapping[str, Any]:
    client = boto3.client(
        "bedrock-agentcore",
        region_name=region,
        config=RUNTIME_CLIENT_CONFIG,
    )
    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        qualifier="DEFAULT",
        contentType="application/json",
        accept="application/json",
        runtimeUserId=user_id,
        runtimeSessionId=session_id,
        payload=json.dumps(payload).encode("utf-8"),
    )
    body = response["response"].read()
    if int(response["statusCode"]) != 200:
        raise RuntimeError(
            f"Runtime returned {response['statusCode']}: {body!r}"
        )
    return json.loads(body)


def invoke_with_retries(**kwargs: Any) -> Mapping[str, Any]:
    last_error: Exception | None = None
    base_session_id = str(kwargs["session_id"])
    for attempt in range(3):
        try:
            attempt_kwargs = dict(kwargs)
            attempt_kwargs["session_id"] = f"{base_session_id}-{attempt + 1}"
            return invoke(**attempt_kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(10)
    raise RuntimeError("Runtime smoke failed after retries") from last_error


def smoke(runtime_arn: str, region: str) -> None:
    smoke_dir = ROOT / "artifacts" / "smoke" / "current"
    observation = json.loads(
        (smoke_dir / "observation.json").read_text(encoding="utf-8")
    )
    request = json.loads(
        (smoke_dir / "request.json").read_text(encoding="utf-8")
    )
    run_id = uuid4().hex
    user_id = f"dummy-deploy-smoke-{run_id}"
    observation["observation"]["user_id"] = user_id
    observation["observation"]["session_id"] = f"observation-{run_id}"
    request["operation"] = "simulate"
    request["request"]["user"]["user_id"] = user_id
    request["request"]["request_id"] = f"deploy-smoke-{run_id}"
    request["request"]["memory_session_id"] = observation["observation"][
        "session_id"
    ]
    request["request"].pop("kg_evidence", None)
    receipt = invoke_with_retries(
        runtime_arn=runtime_arn,
        payload=observation,
        user_id=user_id,
        session_id=f"deploy-write-{run_id}",
        region=region,
    )
    result = invoke_with_retries(
        runtime_arn=runtime_arn,
        payload=request,
        user_id=user_id,
        session_id=f"deploy-read-{run_id}",
        region=region,
    )
    (smoke_dir / "observation-response.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (smoke_dir / "response.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if int(receipt.get("long_term_record_count", 0)) < 1:
        raise RuntimeError("Smoke did not persist long-term records")
    if "probability" not in result:
        raise RuntimeError("Smoke simulation did not return a probability")
    memory_events = result.get("components", {}).get(
        "episodic_memory_events",
        0,
    )
    if int(memory_events) < len(observation["observation"]["events"]):
        raise RuntimeError("Smoke simulation did not read the written events")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Deploy the complete, reproducible AgentCore prototype."
    )
    parser.add_argument("--region")
    parser.add_argument("--target", default="default")
    parser.add_argument(
        "--config",
        type=Path,
        default=LOCAL_DEPLOYMENT_CONFIG,
        help="Ignored local AWS deployment settings JSON.",
    )
    parser.add_argument(
        "--neptune-cluster-id",
        help="Override the cluster ID from the local deployment config.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate local AWS settings and AgentCore schema without deploying.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    settings = DeploymentSettings.from_path(args.config)
    if args.region and args.region != settings.region:
        settings = replace(settings, region=args.region)
    if args.neptune_cluster_id:
        settings = replace(
            settings,
            neptune_cluster_id=args.neptune_cluster_id,
        )
    settings.validate()
    verify_aws_account(settings)

    agentcore_stack = f"AgentCore-BehaviorSim-{args.target}"
    access_stack = f"BehaviorSim-{args.target}-runtime-data-access"
    with configured_deployment_environment(settings, target=args.target):
        run(["agentcore", "validate"], cwd=PROJECT_DIR)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "validated": True,
                        "account": settings.account,
                        "region": settings.region,
                        "target": args.target,
                    },
                    indent=2,
                )
            )
            return
        existing = deployed_strategy_ids(
            region=settings.region,
            stack_name=agentcore_stack,
        )
        deployed_ids = existing[1] if existing is not None else None
        deploy_agentcore(target=args.target, strategy_ids=deployed_ids)
        outputs = stack_outputs(settings.region, agentcore_stack)
        runtime_id = find_output(
            outputs,
            "PurchaseBehaviorSimulatorRuntimeIdOutput",
        )
        runtime_arn = find_output(
            outputs,
            "PurchaseBehaviorSimulatorRuntimeArnOutput",
        )
        role_arn = find_output(
            outputs,
            "PurchaseBehaviorSimulatorRoleArnOutput",
        )
        memory_id = find_output(
            outputs,
            "MemoryPurchaseBehaviorSimulatorMemoryIdOutput",
        )
        control = boto3.client(
            "bedrock-agentcore-control",
            region_name=settings.region,
        )
        current_ids = memory_strategy_ids(control, memory_id)
        if current_ids != deployed_ids:
            deploy_agentcore(target=args.target, strategy_ids=current_ids)
            outputs = stack_outputs(settings.region, agentcore_stack)
            runtime_id = find_output(
                outputs,
                "PurchaseBehaviorSimulatorRuntimeIdOutput",
            )
            runtime_arn = find_output(
                outputs,
                "PurchaseBehaviorSimulatorRuntimeArnOutput",
            )
            role_arn = find_output(
                outputs,
                "PurchaseBehaviorSimulatorRoleArnOutput",
            )
            memory_id = find_output(
                outputs,
                "MemoryPurchaseBehaviorSimulatorMemoryIdOutput",
            )
        deploy_access_policy(
            region=settings.region,
            role_arn=role_arn,
            memory_id=memory_id,
            neptune_cluster_id=settings.neptune_cluster_id,
            stack_name=access_stack,
        )
        wait_ready(runtime_id, settings.region)
        if args.smoke:
            smoke(runtime_arn, settings.region)
    print(
        json.dumps(
            {
                "runtime_id": runtime_id,
                "memory_id": memory_id,
                "access_stack": access_stack,
                "smoke": args.smoke,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
