from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from purchase_behavior_simulator.dataset_adapter import (
    MappedTabularDatasetAdapter,
    ProductionExportBuilder,
    ProductionExportConfig,
)
from purchase_behavior_simulator.production_bootstrap import (
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
)
from scripts.bootstrap_production_data import main
from scripts.bootstrap_production_data import (
    bootstrap_aws,
    deploy_loader_stack,
    prepare_s3_gateway_endpoint,
    resolve_route_table_ids,
    wait_for_imported_data_canary,
    wait_for_neptune_load_role,
)
from scripts.deploy_agentcore import DeploymentSettings


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "dataset_adapter"


class ProductionBootstrapTest(unittest.TestCase):
    def _artifact(self, root: Path):
        dataset = MappedTabularDatasetAdapter(EXAMPLE / "config.json").load()
        ProductionExportBuilder(
            dataset,
            ProductionExportConfig(
                as_of=datetime(2026, 1, 4, tzinfo=timezone.utc),
                identity_salt="test-only-bootstrap-salt",
                allow_synthetic=True,
            ),
        ).write(root)
        return validate_production_artifact(root)

    def test_validates_complete_production_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))

        self.assertGreater(len(artifact.memory_rows), 0)
        self.assertGreater(artifact.node_count, 0)
        self.assertGreater(artifact.edge_count, 0)
        self.assertEqual(len(artifact.fingerprint), 64)

    def test_rejects_manifest_that_does_not_exclude_answer_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._artifact(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["contains_answer_key"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "exclude evaluation data",
            ):
                validate_production_artifact(root)

    def test_memory_import_checkpoints_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = self._artifact(root)
            state_path = root / ".bootstrap" / "state.json"
            state = load_bootstrap_state(
                state_path,
                artifact_fingerprint_value=artifact.fingerprint,
            )
            calls: list[str] = []

            def invoke(payload, user_id, session_id):
                calls.append(session_id)
                observation = payload["observation"]
                return {
                    "user_id": user_id,
                    "session_id": observation["session_id"],
                    "event_count": len(observation["events"]),
                    "long_term_record_count": 2,
                }

            first = import_memory_rows(
                artifact.memory_rows,
                invoke=invoke,
                state=state,
                state_path=state_path,
            )
            second = import_memory_rows(
                artifact.memory_rows,
                invoke=invoke,
                state=state,
                state_path=state_path,
            )

        self.assertEqual(first["imported_batches"], len(artifact.memory_rows))
        self.assertEqual(second["imported_batches"], 0)
        self.assertEqual(second["skipped_batches"], len(artifact.memory_rows))
        self.assertEqual(len(calls), len(artifact.memory_rows))

    def test_builds_canary_for_imported_actor_and_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            user_id, session_id, payload = build_canary_request(artifact)

        self.assertTrue(user_id.startswith("user-"))
        self.assertTrue(session_id.startswith("session-"))
        self.assertEqual(payload["operation"], "simulate")
        self.assertEqual(payload["request"]["user"]["user_id"], user_id)
        self.assertNotEqual(
            payload["request"]["memory_session_id"],
            session_id,
        )
        self.assertTrue(
            payload["request"]["memory_session_id"].startswith(
                "bootstrap-canary-"
            )
        )
        self.assertIn("target_product", payload["request"])

    def test_validates_long_term_memory_and_neptune_canary(self) -> None:
        evidence = validate_canary_result(
            {
                "scalar_purchase_probability": 0.1,
                "trajectory_purchase_probability": 0.08,
                "components": {
                    "episodic_memory_records": 2,
                    "episodic_memory_events": 3,
                    "observed_memory_transitions": 1,
                    "knowledge_graph_retrieval_support": 0.4,
                },
                "action_graph_id": "game_store_purchase",
            },
            require_memory=True,
            require_neptune=True,
        )

        self.assertEqual(evidence["episodic_memory_records"], 2)
        self.assertEqual(
            evidence["knowledge_graph_retrieval_support"],
            0.4,
        )

    def test_canary_waits_for_long_term_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            results = iter(
                (
                    {
                        "scalar_purchase_probability": 0.1,
                        "trajectory_purchase_probability": 0.08,
                        "components": {
                            "episodic_memory_records": 0,
                            "knowledge_graph_retrieval_support": 0.3,
                        },
                    },
                    {
                        "scalar_purchase_probability": 0.1,
                        "trajectory_purchase_probability": 0.08,
                        "components": {
                            "episodic_memory_records": 2,
                            "episodic_memory_events": 0,
                            "observed_memory_transitions": 1,
                            "knowledge_graph_retrieval_support": 0.3,
                        },
                        "action_graph_id": "game_store_purchase",
                    },
                )
            )
            with (
                patch(
                    "scripts.bootstrap_production_data.invoke_with_retries",
                    side_effect=lambda **_: next(results),
                ),
                patch("scripts.bootstrap_production_data.time.sleep"),
            ):
                result = wait_for_imported_data_canary(
                    runtime_arn="runtime-arn",
                    artifact=artifact,
                    region="us-east-1",
                    require_memory=True,
                    require_neptune=True,
                    timeout_seconds=60,
                    poll_seconds=1,
                )

        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["episodic_memory_records"], 2)

    def test_incremental_lineage_must_continue_previous_as_of(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_root = root / "snapshot"
            snapshot = self._artifact(snapshot_root)
            lineage_path = root / "lineage.json"
            lineage = load_bootstrap_lineage(lineage_path)
            validate_artifact_lineage(snapshot, lineage)
            record_artifact_lineage(lineage_path, lineage, snapshot)

            dataset = MappedTabularDatasetAdapter(EXAMPLE / "config.json").load()
            incremental_root = root / "incremental"
            ProductionExportBuilder(
                dataset,
                ProductionExportConfig(
                    since=datetime(2026, 1, 4, tzinfo=timezone.utc),
                    as_of=datetime(2026, 1, 6, tzinfo=timezone.utc),
                    identity_salt="test-only-bootstrap-salt",
                    allow_synthetic=True,
                ),
            ).write(incremental_root)
            incremental = validate_production_artifact(incremental_root)
            loaded = load_bootstrap_lineage(lineage_path)
            validate_artifact_lineage(incremental, loaded)

            invalid_root = root / "invalid"
            ProductionExportBuilder(
                dataset,
                ProductionExportConfig(
                    since=datetime(2026, 1, 3, tzinfo=timezone.utc),
                    as_of=datetime(2026, 1, 6, tzinfo=timezone.utc),
                    identity_salt="test-only-bootstrap-salt",
                    allow_synthetic=True,
                ),
            ).write(invalid_root)
            invalid = validate_production_artifact(invalid_root)
            with self.assertRaisesRegex(ValueError, "previous as_of"):
                validate_artifact_lineage(invalid, loaded)

            wrong_salt_root = root / "wrong-salt"
            ProductionExportBuilder(
                dataset,
                ProductionExportConfig(
                    since=datetime(2026, 1, 4, tzinfo=timezone.utc),
                    as_of=datetime(2026, 1, 6, tzinfo=timezone.utc),
                    identity_salt="different-salt",
                    allow_synthetic=True,
                ),
            ).write(wrong_salt_root)
            wrong_salt = validate_production_artifact(wrong_salt_root)
            with self.assertRaisesRegex(ValueError, "identity salt"):
                validate_artifact_lineage(wrong_salt, loaded)

    def test_extracts_neptune_loader_id_and_status(self) -> None:
        started = {
            "status": "200 OK",
            "payload": {"loadId": "load-123"},
        }
        completed = {
            "status": "200 OK",
            "payload": {
                "overallStatus": {"status": "LOAD_COMPLETED"}
            },
        }

        self.assertEqual(extract_loader_id(started), "load-123")
        self.assertEqual(extract_loader_status(completed), "LOAD_COMPLETED")

    def test_cli_dry_run_does_not_require_aws_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = self._artifact(root)
            output = io.StringIO()
            with redirect_stdout(output):
                main(["--artifact-dir", str(root), "--dry-run"])
            result = json.loads(output.getvalue())

        self.assertTrue(result["validated"])
        self.assertFalse(result["aws_called"])
        self.assertEqual(
            result["memory_batches"],
            len(artifact.memory_rows),
        )

    def test_resolves_subnet_route_tables_for_s3_endpoint(self) -> None:
        class FakeEc2:
            def describe_route_tables(self, *, Filters):
                subnet = next(
                    (
                        value["Values"][0]
                        for value in Filters
                        if value["Name"] == "association.subnet-id"
                    ),
                    None,
                )
                if subnet == "subnet-a":
                    return {"RouteTables": [{"RouteTableId": "rtb-a"}]}
                if subnet == "subnet-b":
                    return {"RouteTables": [{"RouteTableId": "rtb-b"}]}
                return {"RouteTables": []}

        self.assertEqual(
            resolve_route_table_ids(
                FakeEc2(),
                vpc_id="vpc-1",
                subnet_ids=("subnet-b", "subnet-a"),
            ),
            ("rtb-a", "rtb-b"),
        )

    def test_uses_main_route_table_for_unassociated_subnet(self) -> None:
        class FakeEc2:
            def describe_route_tables(self, *, Filters):
                if any(
                    value["Name"] == "association.main"
                    for value in Filters
                ):
                    return {
                        "RouteTables": [{"RouteTableId": "rtb-main"}]
                    }
                return {"RouteTables": []}

        self.assertEqual(
            resolve_route_table_ids(
                FakeEc2(),
                vpc_id="vpc-1",
                subnet_ids=("subnet-a", "subnet-b"),
            ),
            ("rtb-main",),
        )

    def test_reuses_existing_s3_gateway_endpoint_and_adds_routes(self) -> None:
        class FakeEc2:
            modified = None

            def describe_vpc_endpoints(self, *, Filters):
                return {
                    "VpcEndpoints": [
                        {
                            "VpcEndpointId": "vpce-s3",
                            "RouteTableIds": ["rtb-a"],
                        }
                    ]
                }

            def modify_vpc_endpoint(self, **kwargs):
                self.modified = kwargs

        client = FakeEc2()

        self.assertFalse(
            prepare_s3_gateway_endpoint(
                client,
                vpc_id="vpc-1",
                region="us-east-1",
                route_table_ids=("rtb-a", "rtb-b"),
            )
        )
        self.assertEqual(
            client.modified,
            {
                "VpcEndpointId": "vpce-s3",
                "AddRouteTableIds": ["rtb-b"],
            },
        )

    def test_deploy_loader_stack_associates_neptune_load_role(self) -> None:
        class FakeNeptune:
            role = None

            def describe_db_clusters(self, *, DBClusterIdentifier):
                return {
                    "DBClusters": [
                        {
                            "DbClusterResourceId": "cluster-resource-id",
                            "AssociatedRoles": (
                                [
                                    {
                                        "RoleArn": self.role["RoleArn"],
                                        "Status": "ACTIVE",
                                    }
                                ]
                                if self.role
                                else []
                            ),
                        }
                    ]
                }

            def add_role_to_db_cluster(self, **kwargs):
                self.role = kwargs

        class FakeEc2:
            def describe_subnets(self, *, SubnetIds):
                return {"Subnets": [{"VpcId": "vpc-1"}]}

            def describe_route_tables(self, *, Filters):
                return {"RouteTables": [{"RouteTableId": "rtb-1"}]}

            def describe_vpc_endpoints(self, *, Filters):
                return {"VpcEndpoints": []}

        class FakeCloudFormation:
            def describe_stacks(self, *, StackName):
                return {"Stacks": [{"StackStatus": "CREATE_COMPLETE"}]}

        neptune = FakeNeptune()
        ec2 = FakeEc2()
        cloudformation = FakeCloudFormation()
        settings = DeploymentSettings(
            account="123456789012",
            region="us-east-1",
            model_id="test-model",
            subnet_ids=("subnet-a",),
            security_group_ids=("sg-a",),
            neptune_endpoint="cluster.example.neptune.amazonaws.com",
            neptune_cluster_id="cluster-id",
        )
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._artifact(Path(temporary))
            with (
                patch(
                    "scripts.bootstrap_production_data.boto3.client",
                    side_effect=lambda service, **_: (
                        neptune
                        if service == "neptune"
                        else (
                            cloudformation
                            if service == "cloudformation"
                            else ec2
                        )
                    ),
                ),
                patch("scripts.bootstrap_production_data.run") as run_mock,
                patch(
                    "scripts.bootstrap_production_data.stack_outputs",
                    return_value={
                        "LoadRoleArn": (
                            "arn:aws:iam::123456789012:"
                            "role/simplayer-load"
                        )
                    },
                ),
            ):
                outputs = deploy_loader_stack(
                    artifact=artifact,
                    settings=settings,
                    target="default",
                    stack_name="loader-stack",
                    bucket_name="loader-bucket",
                )

        self.assertEqual(outputs["LoadRoleArn"].split("/")[-1], "simplayer-load")
        self.assertEqual(
            neptune.role,
            {
                "DBClusterIdentifier": "cluster-id",
                "RoleArn": (
                    "arn:aws:iam::123456789012:role/simplayer-load"
                ),
            },
        )
        command = run_mock.call_args.args[0]
        self.assertIn("RouteTableIds=rtb-1", command)
        self.assertIn("CreateS3GatewayEndpoint=true", command)

    def test_waits_for_neptune_load_role_to_become_active(self) -> None:
        class FakeNeptune:
            calls = 0

            def describe_db_clusters(self, *, DBClusterIdentifier):
                self.calls += 1
                return {
                    "DBClusters": [
                        {
                            "AssociatedRoles": [
                                {
                                    "RoleArn": "role-arn",
                                    "Status": (
                                        "PENDING"
                                        if self.calls == 1
                                        else "ACTIVE"
                                    ),
                                }
                            ]
                        }
                    ]
                }

        client = FakeNeptune()
        with patch("scripts.bootstrap_production_data.time.sleep"):
            wait_for_neptune_load_role(
                client,
                cluster_id="cluster-id",
                role_arn="role-arn",
                timeout_seconds=60,
                poll_seconds=1,
            )
        self.assertEqual(client.calls, 2)

    def test_bootstrap_aws_orders_runtime_data_and_canary_steps(self) -> None:
        settings = DeploymentSettings(
            account="123456789012",
            region="us-east-1",
            model_id="test-model",
            subnet_ids=("subnet-a",),
            security_group_ids=("sg-a",),
            neptune_endpoint="cluster.example.neptune.amazonaws.com",
            neptune_cluster_id="cluster-id",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = self._artifact(root)
            state_path = root / ".bootstrap" / "state.json"
            lineage_path = root / ".bootstrap" / "lineage.json"
            calls: list[str] = []

            def mark(name, value):
                calls.append(name)
                return value

            with (
                patch(
                    "scripts.bootstrap_production_data.verify_aws_account",
                    side_effect=lambda _: calls.append("account"),
                ),
                patch(
                    "scripts.bootstrap_production_data.stack_outputs",
                    return_value={
                        "PurchaseBehaviorSimulatorRuntimeIdOutput": "runtime-id",
                        "PurchaseBehaviorSimulatorRuntimeArnOutput": "runtime-arn",
                    },
                ),
                patch(
                    "scripts.bootstrap_production_data.wait_ready",
                    side_effect=lambda *_: calls.append("ready"),
                ),
                patch(
                    "scripts.bootstrap_production_data.load_neptune",
                    side_effect=lambda *_, **__: mark(
                        "neptune",
                        {"status": "LOAD_COMPLETED"},
                    ),
                ),
                patch(
                    "scripts.bootstrap_production_data.import_memory_rows",
                    side_effect=lambda *_, **__: mark(
                        "memory",
                        {"completed_batches": len(artifact.memory_rows)},
                    ),
                ),
                patch(
                    "scripts.bootstrap_production_data.wait_for_imported_data_canary",
                    side_effect=lambda **_: mark(
                        "canary",
                        {
                            "completed": True,
                            "episodic_memory_records": 1,
                            "knowledge_graph_retrieval_support": 0.2,
                            "action_graph_id": "game_store_purchase",
                        },
                    ),
                ),
            ):
                report = bootstrap_aws(
                    artifact,
                    settings=settings,
                    target="default",
                    state_path=state_path,
                    lineage_path=lineage_path,
                    bucket_name="bucket",
                    loader_stack_name="loader",
                    skip_neptune=False,
                    skip_memory=False,
                    skip_canary=False,
                    restart_neptune_load=False,
                    loader_timeout_seconds=60,
                    loader_poll_seconds=1,
                    canary_timeout_seconds=60,
                    canary_poll_seconds=1,
                )

        self.assertEqual(
            calls,
            ["account", "ready", "neptune", "memory", "canary"],
        )
        self.assertEqual(report["runtime_id"], "runtime-id")
        self.assertTrue(report["canary"]["completed"])


if __name__ == "__main__":
    unittest.main()
