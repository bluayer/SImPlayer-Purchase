from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from purchase_behavior_simulator.production_bootstrap import (
    ProductionArtifact,
    build_canary_request,
    extract_loader_id,
    extract_loader_status,
    import_memory_rows,
    load_bootstrap_lineage,
    load_bootstrap_state,
    record_artifact_lineage,
    validate_artifact_lineage,
    validate_canary_result,
    validate_production_artifact,
    write_bootstrap_state,
)
from scripts.deploy_agentcore import (
    DeploymentSettings,
    find_output,
    invoke_with_retries,
    run,
    stack_outputs,
    verify_aws_account,
    wait_ready,
)


LOADER_TEMPLATE = (
    ROOT / "deployment" / "neptune" / "neptune-bulk-loader.yaml"
)
LOCAL_DEPLOYMENT_CONFIG = (
    ROOT / "deployment" / "agentcore" / "deployment.local.json"
)
LOADER_TERMINAL_SUCCESS = {"LOAD_COMPLETED"}
LOADER_TERMINAL_FAILURE = {
    "LOAD_FAILED",
    "LOAD_CANCELLED_BY_USER",
    "LOAD_CANCELLED_DUE_TO_ERRORS",
    "LOAD_UNEXPECTED_ERROR",
}


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return normalized or "default"


def default_bucket_name(settings: DeploymentSettings, target: str) -> str:
    suffix = safe_name(target)[:20]
    return (
        f"simplayer-purchase-data-{settings.account}-"
        f"{settings.region}-{suffix}"
    )[:63].rstrip("-")


def invoke_lambda(
    client: Any,
    *,
    function_name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    body = json.loads(response["Payload"].read())
    if response.get("FunctionError"):
        raise RuntimeError(f"Neptune loader Lambda failed: {body}")
    if not isinstance(body, dict):
        raise RuntimeError("Neptune loader Lambda returned a non-object")
    return body


def deploy_loader_stack(
    *,
    artifact: ProductionArtifact,
    settings: DeploymentSettings,
    target: str,
    stack_name: str,
    bucket_name: str,
) -> dict[str, str]:
    cloudformation = boto3.client(
        "cloudformation",
        region_name=settings.region,
    )
    try:
        stacks = cloudformation.describe_stacks(StackName=stack_name)[
            "Stacks"
        ]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ValidationError":
            raise
        stacks = ()
    if stacks and stacks[0].get("StackStatus") == "ROLLBACK_COMPLETE":
        cloudformation.delete_stack(StackName=stack_name)
        cloudformation.get_waiter("stack_delete_complete").wait(
            StackName=stack_name
        )

    neptune = boto3.client("neptune", region_name=settings.region)
    cluster = neptune.describe_db_clusters(
        DBClusterIdentifier=settings.neptune_cluster_id
    )["DBClusters"][0]
    ec2 = boto3.client("ec2", region_name=settings.region)
    subnets = ec2.describe_subnets(
        SubnetIds=[settings.subnet_ids[0]]
    )["Subnets"]
    if not subnets:
        raise RuntimeError("could not resolve the VPC from the configured subnet")
    vpc_id = str(subnets[0]["VpcId"])
    route_table_ids = resolve_route_table_ids(
        ec2,
        vpc_id=vpc_id,
        subnet_ids=settings.subnet_ids,
    )
    create_s3_endpoint = prepare_s3_gateway_endpoint(
        ec2,
        vpc_id=vpc_id,
        region=settings.region,
        route_table_ids=route_table_ids,
    )
    run(
        [
            "aws",
            "cloudformation",
            "deploy",
            "--stack-name",
            stack_name,
            "--template-file",
            str(LOADER_TEMPLATE),
            "--parameter-overrides",
            f"EnvironmentName=simplayer-purchase-{safe_name(target)}",
            f"VpcId={vpc_id}",
            f"SubnetIds={','.join(settings.subnet_ids)}",
            f"RouteTableIds={','.join(route_table_ids)}",
            (
                "CreateS3GatewayEndpoint="
                f"{str(create_s3_endpoint).lower()}"
            ),
            f"ClientSecurityGroupId={settings.security_group_ids[0]}",
            f"NeptuneEndpoint={settings.neptune_endpoint}",
            (
                "NeptuneClusterResourceId="
                f"{cluster['DbClusterResourceId']}"
            ),
            f"DataBucketName={bucket_name}",
            "--capabilities",
            "CAPABILITY_NAMED_IAM",
            "--region",
            settings.region,
        ]
    )
    outputs = stack_outputs(settings.region, stack_name)
    role_arn = find_output(outputs, "LoadRoleArn")
    try:
        neptune.add_role_to_db_cluster(
            DBClusterIdentifier=settings.neptune_cluster_id,
            RoleArn=role_arn,
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in {
            "DBClusterRoleAlreadyExists",
            "DBClusterRoleAlreadyExistsFault",
        }:
            raise
    wait_for_neptune_load_role(
        neptune,
        cluster_id=settings.neptune_cluster_id,
        role_arn=role_arn,
    )
    return outputs


def wait_for_neptune_load_role(
    client: Any,
    *,
    cluster_id: str,
    role_arn: str,
    timeout_seconds: int = 300,
    poll_seconds: int = 5,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status = "MISSING"
    while time.monotonic() < deadline:
        cluster = client.describe_db_clusters(
            DBClusterIdentifier=cluster_id
        )["DBClusters"][0]
        matching = next(
            (
                role
                for role in cluster.get("AssociatedRoles", ())
                if str(role.get("RoleArn")) == role_arn
            ),
            None,
        )
        last_status = (
            str(matching.get("Status", "UNKNOWN")).upper()
            if matching
            else "MISSING"
        )
        if last_status == "ACTIVE":
            return
        if last_status in {"INVALID", "FAILED"}:
            raise RuntimeError(
                f"Neptune load role entered {last_status}: {role_arn}"
            )
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"Neptune load role did not become ACTIVE within {timeout_seconds} "
        f"seconds: {role_arn} ({last_status})"
    )


def resolve_route_table_ids(
    ec2: Any,
    *,
    vpc_id: str,
    subnet_ids: Sequence[str],
) -> tuple[str, ...]:
    route_tables: set[str] = set()
    main_route_table_id: str | None = None
    for subnet_id in subnet_ids:
        response = ec2.describe_route_tables(
            Filters=[
                {
                    "Name": "association.subnet-id",
                    "Values": [subnet_id],
                }
            ]
        )
        subnet_route_tables = {
            str(value["RouteTableId"])
            for value in response.get("RouteTables", ())
        }
        if subnet_route_tables:
            route_tables.update(subnet_route_tables)
            continue
        if main_route_table_id is None:
            main_response = ec2.describe_route_tables(
                Filters=[
                    {"Name": "vpc-id", "Values": [vpc_id]},
                    {"Name": "association.main", "Values": ["true"]},
                ]
            )
            main_route_tables = main_response.get("RouteTables", ())
            if not main_route_tables:
                raise RuntimeError(
                    "could not resolve the VPC main route table"
                )
            main_route_table_id = str(
                main_route_tables[0]["RouteTableId"]
            )
        route_tables.add(main_route_table_id)
    if not route_tables:
        raise RuntimeError(
            "could not resolve route tables for the loader S3 endpoint"
        )
    return tuple(sorted(route_tables))


def prepare_s3_gateway_endpoint(
    ec2: Any,
    *,
    vpc_id: str,
    region: str,
    route_table_ids: Sequence[str],
) -> bool:
    response = ec2.describe_vpc_endpoints(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {
                "Name": "service-name",
                "Values": [f"com.amazonaws.{region}.s3"],
            },
            {
                "Name": "vpc-endpoint-state",
                "Values": ["available", "pending"],
            },
        ]
    )
    endpoints = response.get("VpcEndpoints", ())
    if not endpoints:
        return True
    endpoint = endpoints[0]
    associated = set(str(value) for value in endpoint.get("RouteTableIds", ()))
    missing = sorted(set(route_table_ids) - associated)
    if missing:
        ec2.modify_vpc_endpoint(
            VpcEndpointId=str(endpoint["VpcEndpointId"]),
            AddRouteTableIds=missing,
        )
    return False


def upload_neptune_artifact(
    artifact: ProductionArtifact,
    *,
    region: str,
    bucket_name: str,
    prefix: str,
) -> None:
    s3 = boto3.client("s3", region_name=region)
    metadata = {"artifact-fingerprint": artifact.fingerprint}
    for path in (artifact.nodes_path, artifact.edges_path):
        s3.upload_file(
            str(path),
            bucket_name,
            f"{prefix}/{path.name}",
            ExtraArgs={
                "ContentType": "text/csv",
                "ServerSideEncryption": "AES256",
                "Metadata": metadata,
            },
        )


def wait_for_loader(
    *,
    lambda_client: Any,
    function_name: str,
    load_id: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = invoke_lambda(
            lambda_client,
            function_name=function_name,
            payload={"action": "status", "loadId": load_id},
        )
        status = extract_loader_status(last)
        if status in LOADER_TERMINAL_SUCCESS:
            return last
        if status in LOADER_TERMINAL_FAILURE:
            raise RuntimeError(
                f"Neptune loader {load_id} entered {status}: {last}"
            )
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"Neptune loader {load_id} did not finish within "
        f"{timeout_seconds} seconds; last response={last}"
    )


def load_neptune(
    artifact: ProductionArtifact,
    *,
    settings: DeploymentSettings,
    target: str,
    state: dict[str, Any],
    state_path: Path,
    bucket_name: str,
    loader_stack_name: str,
    timeout_seconds: int,
    poll_seconds: int,
    restart: bool,
) -> dict[str, Any]:
    neptune_state = state.setdefault("neptune", {})
    if (
        neptune_state.get("status") == "LOAD_COMPLETED"
        and not restart
    ):
        return dict(neptune_state)
    if restart:
        neptune_state.clear()

    outputs = deploy_loader_stack(
        artifact=artifact,
        settings=settings,
        target=target,
        stack_name=loader_stack_name,
        bucket_name=bucket_name,
    )
    function_name = find_output(outputs, "LoaderFunctionName")
    prefix = (
        f"behavior-graph/{safe_name(target)}/"
        f"{artifact.fingerprint[:24]}"
    )
    upload_neptune_artifact(
        artifact,
        region=settings.region,
        bucket_name=bucket_name,
        prefix=prefix,
    )
    neptune_state.update(
        {
            "bucket": bucket_name,
            "prefix": prefix,
            "loader_function": function_name,
        }
    )
    write_bootstrap_state(state_path, state)

    lambda_client = boto3.client("lambda", region_name=settings.region)
    load_id = str(neptune_state.get("load_id", ""))
    if not load_id:
        response = invoke_lambda(
            lambda_client,
            function_name=function_name,
            payload={
                "action": "start",
                "prefix": prefix,
                "mode": "NEW",
                "parallelism": "MEDIUM",
            },
        )
        load_id = extract_loader_id(response)
        neptune_state["load_id"] = load_id
        neptune_state["status"] = extract_loader_status(response)
        write_bootstrap_state(state_path, state)

    response = wait_for_loader(
        lambda_client=lambda_client,
        function_name=function_name,
        load_id=load_id,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    neptune_state["status"] = extract_loader_status(response)
    neptune_state["completed_at_epoch"] = int(time.time())
    write_bootstrap_state(state_path, state)
    return dict(neptune_state)


def wait_for_imported_data_canary(
    *,
    runtime_arn: str,
    artifact: ProductionArtifact,
    region: str,
    require_memory: bool,
    require_neptune: bool,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    user_id, imported_session_id, request = build_canary_request(artifact)
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        try:
            result = invoke_with_retries(
                runtime_arn=runtime_arn,
                payload=request,
                user_id=user_id,
                session_id=(
                    f"bootstrap-canary-{artifact.fingerprint[:20]}-"
                    f"{attempts:04d}"
                ),
                region=region,
            )
            evidence = validate_canary_result(
                result,
                require_memory=require_memory,
                require_neptune=require_neptune,
            )
            return {
                "completed": True,
                "attempts": attempts,
                "user_id": user_id,
                "imported_session_id": imported_session_id,
                "retrieval_session_id": request["request"][
                    "memory_session_id"
                ],
                **evidence,
            }
        except RuntimeError as exc:
            last_error = exc
            if time.monotonic() + poll_seconds >= deadline:
                break
            time.sleep(poll_seconds)
    raise TimeoutError(
        "imported-data canary did not observe long-term Memory and Neptune "
        f"evidence within {timeout_seconds} seconds"
    ) from last_error


def bootstrap_aws(
    artifact: ProductionArtifact,
    *,
    settings: DeploymentSettings,
    target: str,
    state_path: Path,
    lineage_path: Path,
    bucket_name: str,
    loader_stack_name: str,
    skip_neptune: bool,
    skip_memory: bool,
    skip_canary: bool,
    restart_neptune_load: bool,
    loader_timeout_seconds: int,
    loader_poll_seconds: int,
    canary_timeout_seconds: int,
    canary_poll_seconds: int,
) -> dict[str, Any]:
    verify_aws_account(settings)
    lineage = load_bootstrap_lineage(lineage_path)
    validate_artifact_lineage(artifact, lineage)
    state = load_bootstrap_state(
        state_path,
        artifact_fingerprint_value=artifact.fingerprint,
    )
    agentcore_stack = f"AgentCore-BehaviorSim-{target}"
    outputs = stack_outputs(settings.region, agentcore_stack)
    runtime_id = find_output(
        outputs,
        "PurchaseBehaviorSimulatorRuntimeIdOutput",
    )
    runtime_arn = find_output(
        outputs,
        "PurchaseBehaviorSimulatorRuntimeArnOutput",
    )
    wait_ready(runtime_id, settings.region)

    report: dict[str, Any] = {
        **artifact.summary(),
        "aws_called": True,
        "target": target,
        "runtime_id": runtime_id,
        "runtime_arn": runtime_arn,
        "state_path": str(state_path),
        "lineage_path": str(lineage_path),
    }
    if not skip_neptune:
        report["neptune"] = load_neptune(
            artifact,
            settings=settings,
            target=target,
            state=state,
            state_path=state_path,
            bucket_name=bucket_name,
            loader_stack_name=loader_stack_name,
            timeout_seconds=loader_timeout_seconds,
            poll_seconds=loader_poll_seconds,
            restart=restart_neptune_load,
        )

    if not skip_memory:
        def runtime_invoke(
            payload: Mapping[str, Any],
            user_id: str,
            session_id: str,
        ) -> Mapping[str, Any]:
            return invoke_with_retries(
                runtime_arn=runtime_arn,
                payload=payload,
                user_id=user_id,
                session_id=session_id,
                region=settings.region,
            )

        report["memory"] = import_memory_rows(
            artifact.memory_rows,
            invoke=runtime_invoke,
            state=state,
            state_path=state_path,
        )

    if not skip_canary:
        canary = wait_for_imported_data_canary(
            runtime_arn=runtime_arn,
            region=settings.region,
            artifact=artifact,
            require_memory=not skip_memory,
            require_neptune=not skip_neptune,
            timeout_seconds=canary_timeout_seconds,
            poll_seconds=canary_poll_seconds,
        )
        state["canary"] = canary
        write_bootstrap_state(state_path, state)
        report["canary"] = dict(canary)
    already_recorded = any(
        isinstance(value, Mapping)
        and value.get("fingerprint") == artifact.fingerprint
        for value in lineage.get("artifacts", ())
    )
    if not any((skip_neptune, skip_memory, skip_canary)):
        record_artifact_lineage(lineage_path, lineage, artifact)
        report["lineage_recorded"] = True
    elif already_recorded:
        report["lineage_recorded"] = True
    else:
        report["lineage_recorded"] = False
        report["lineage_note"] = (
            "partial bootstrap did not advance the artifact lineage"
        )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or bootstrap pseudonymized production artifacts into "
            "Neptune and AgentCore Memory."
        )
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--target", default="default")
    parser.add_argument(
        "--config",
        type=Path,
        default=LOCAL_DEPLOYMENT_CONFIG,
    )
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--lineage-path", type=Path)
    parser.add_argument("--data-bucket-name")
    parser.add_argument("--loader-stack-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-write", action="store_true")
    parser.add_argument("--skip-neptune", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-canary", action="store_true")
    parser.add_argument("--restart-neptune-load", action="store_true")
    parser.add_argument("--loader-timeout-seconds", type=int, default=3600)
    parser.add_argument("--loader-poll-seconds", type=int, default=10)
    parser.add_argument("--canary-timeout-seconds", type=int, default=900)
    parser.add_argument("--canary-poll-seconds", type=int, default=30)
    args = parser.parse_args(argv)

    if args.dry_run and args.confirm_write:
        parser.error("--dry-run and --confirm-write are mutually exclusive")
    if not args.dry_run and not args.confirm_write:
        parser.error(
            "remote bootstrap requires --confirm-write; use --dry-run for "
            "local validation"
        )

    artifact = validate_production_artifact(args.artifact_dir)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "validated": True,
                    **artifact.summary(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    settings = DeploymentSettings.from_path(args.config)
    state_dir = args.state_dir or (
        artifact.root / ".bootstrap" / safe_name(args.target)
    )
    state_path = state_dir / "state.json"
    lineage_path = args.lineage_path or (
        artifact.root.parent
        / ".bootstrap"
        / safe_name(args.target)
        / "lineage.json"
    )
    bucket_name = args.data_bucket_name or default_bucket_name(
        settings,
        args.target,
    )
    loader_stack_name = args.loader_stack_name or (
        f"SimPlayerPurchase-{safe_name(args.target)}-neptune-loader"
    )
    report = bootstrap_aws(
        artifact,
        settings=settings,
        target=args.target,
        state_path=state_path,
        lineage_path=lineage_path,
        bucket_name=bucket_name,
        loader_stack_name=loader_stack_name,
        skip_neptune=args.skip_neptune,
        skip_memory=args.skip_memory,
        skip_canary=args.skip_canary,
        restart_neptune_load=args.restart_neptune_load,
        loader_timeout_seconds=args.loader_timeout_seconds,
        loader_poll_seconds=args.loader_poll_seconds,
        canary_timeout_seconds=args.canary_timeout_seconds,
        canary_poll_seconds=args.canary_poll_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
