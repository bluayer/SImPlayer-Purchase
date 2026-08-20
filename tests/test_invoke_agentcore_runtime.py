from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.invoke_agentcore_runtime import main, runtime_payload


class InvokeAgentCoreRuntimeTest(unittest.TestCase):
    def test_wraps_raw_simulation_request(self) -> None:
        payload = runtime_payload({"request_id": "request-1"})
        self.assertEqual(payload["operation"], "simulate")
        self.assertEqual(payload["request"]["request_id"], "request-1")

    def test_preserves_explicit_operation(self) -> None:
        payload = runtime_payload(
            {"operation": "record_observations", "observation": {}}
        )
        self.assertEqual(payload["operation"], "record_observations")

    def test_cli_resolves_runtime_and_prints_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "deployment.json"
            config.write_text(
                json.dumps(
                    {
                        "account": "123456789012",
                        "region": "us-east-1",
                        "model_id": "test-model",
                        "subnet_ids": ["subnet-a"],
                        "security_group_ids": ["sg-a"],
                        "neptune_endpoint": (
                            "cluster.example.neptune.amazonaws.com"
                        ),
                        "neptune_cluster_id": "cluster-id",
                    }
                ),
                encoding="utf-8",
            )
            payload = root / "request.json"
            payload.write_text(
                json.dumps(
                    {
                        "request_id": "request-1",
                        "user": {"user_id": "user-1"},
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch(
                    "scripts.invoke_agentcore_runtime.verify_aws_account"
                ),
                patch(
                    "scripts.invoke_agentcore_runtime.resolve_runtime",
                    return_value=("runtime-arn", "runtime-id"),
                ),
                patch(
                    "scripts.invoke_agentcore_runtime.invoke_with_retries",
                    return_value={"scalar_purchase_probability": 0.1},
                ) as invoke_mock,
                redirect_stdout(output),
            ):
                main(
                    [
                        str(payload),
                        "--config",
                        str(config),
                        "--session-id",
                        "session-" + "x" * 40,
                    ]
                )

        result = json.loads(output.getvalue())
        self.assertEqual(result["runtime_id"], "runtime-id")
        self.assertEqual(
            result["result"]["scalar_purchase_probability"],
            0.1,
        )
        sent = invoke_mock.call_args.kwargs["payload"]
        self.assertEqual(sent["operation"], "simulate")
        self.assertEqual(sent["request"]["request_id"], "request-1")


if __name__ == "__main__":
    unittest.main()
