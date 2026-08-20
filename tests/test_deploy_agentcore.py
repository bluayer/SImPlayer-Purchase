from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from scripts.deploy_agentcore import (
    AGENTCORE_CONFIG,
    AWS_TARGETS_CONFIG,
    DeploymentSettings,
    configured_deployment_environment,
    configured_strategy_environment,
    memory_strategy_ids,
    verify_aws_account,
)


class FakeControlPlaneClient:
    def get_memory(self, *, memoryId):
        return {
            "memory": {
                "strategies": [
                    {
                        "strategyId": "preference-id",
                        "type": "USER_PREFERENCE",
                        "status": "ACTIVE",
                    },
                    {
                        "strategyId": "episodic-id",
                        "type": "EPISODIC",
                        "status": "ACTIVE",
                    },
                    {
                        "strategyId": "semantic-id",
                        "type": "SEMANTIC",
                        "status": "ACTIVE",
                    },
                ]
            }
        }


class DeployAgentCoreTest(unittest.TestCase):
    def test_temporarily_injects_local_aws_environment(self) -> None:
        settings = DeploymentSettings(
            account="123456789012",
            region="ap-northeast-2",
            model_id="test-model",
            subnet_ids=("subnet-0123456789abcdef0",),
            security_group_ids=("sg-0123456789abcdef0",),
            neptune_endpoint=(
                "customer.cluster-example.ap-northeast-2.neptune.amazonaws.com"
            ),
            neptune_cluster_id="customer-neptune",
        )
        original_agentcore = AGENTCORE_CONFIG.read_bytes()
        original_targets = AWS_TARGETS_CONFIG.read_bytes()

        with configured_deployment_environment(settings, target="customer"):
            agentcore = json.loads(
                AGENTCORE_CONFIG.read_text(encoding="utf-8")
            )
            runtime = agentcore["runtimes"][0]
            environment = {
                item["name"]: item["value"] for item in runtime["envVars"]
            }
            targets = json.loads(
                AWS_TARGETS_CONFIG.read_text(encoding="utf-8")
            )
            target = next(
                item for item in targets if item["name"] == "customer"
            )

            self.assertEqual(environment["AWS_REGION"], "ap-northeast-2")
            self.assertEqual(environment["BEDROCK_MODEL_ID"], "test-model")
            self.assertEqual(
                environment["NEPTUNE_ENDPOINT"],
                settings.neptune_endpoint,
            )
            self.assertEqual(
                runtime["networkConfig"]["subnets"],
                list(settings.subnet_ids),
            )
            self.assertEqual(target["account"], settings.account)

        self.assertEqual(AGENTCORE_CONFIG.read_bytes(), original_agentcore)
        self.assertEqual(AWS_TARGETS_CONFIG.read_bytes(), original_targets)

    def test_rejects_credentials_for_a_different_account(self) -> None:
        settings = DeploymentSettings(
            account="123456789012",
            region="us-east-1",
            model_id="test-model",
            subnet_ids=("subnet-0123456789abcdef0",),
            security_group_ids=("sg-0123456789abcdef0",),
            neptune_endpoint=(
                "customer.cluster-example.us-east-1.neptune.amazonaws.com"
            ),
            neptune_cluster_id="customer-neptune",
        )
        client = Mock()
        client.get_caller_identity.return_value = {
            "Account": "210987654321"
        }
        with patch(
            "scripts.deploy_agentcore.boto3.client",
            return_value=client,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "does not match deployment.local.json",
            ):
                verify_aws_account(settings)

    def test_resolves_generated_memory_strategy_ids(self) -> None:
        episodic, transition = memory_strategy_ids(
            FakeControlPlaneClient(),
            "generated-memory-id",
        )

        self.assertEqual(episodic, "episodic-id")
        self.assertEqual(transition, "semantic-id")

    def test_temporarily_injects_generated_strategy_ids(self) -> None:
        original = AGENTCORE_CONFIG.read_bytes()
        with configured_strategy_environment(
            ("episodic-id", "semantic-id")
        ):
            config = json.loads(
                AGENTCORE_CONFIG.read_text(encoding="utf-8")
            )
            runtime = config["runtimes"][0]
            environment = {
                item["name"]: item["value"] for item in runtime["envVars"]
            }
            self.assertEqual(
                environment["PURCHASE_BEHAVIOR_EPISODIC_STRATEGY_ID"],
                "episodic-id",
            )
            self.assertEqual(
                environment["PURCHASE_BEHAVIOR_TRANSITION_STRATEGY_ID"],
                "semantic-id",
            )
        self.assertEqual(AGENTCORE_CONFIG.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
