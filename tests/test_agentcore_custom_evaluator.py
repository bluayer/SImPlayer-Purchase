from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
)


EVALUATOR_PATH = (
    Path(__file__).parents[1]
    / "deployment"
    / "agentcore"
    / "evaluators"
    / "runtime-health"
    / "lambda_function.py"
)
SPEC = importlib.util.spec_from_file_location(
    "purchase_behavior_runtime_health_evaluator",
    EVALUATOR_PATH,
)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class AgentCoreCustomEvaluatorTest(unittest.TestCase):
    def test_passes_clean_spans_and_reports_tokens(self) -> None:
        result = EVALUATOR.evaluate_runtime_quality(
            EvaluatorInput(
                evaluation_level="SESSION",
                session_spans=[
                    {
                        "name": "model",
                        "status": {"code": "UNSET"},
                        "attributes": {
                            "gen_ai.usage.input_tokens": 120,
                            "gen_ai.usage.output_tokens": 30,
                        },
                    },
                    {
                        "name": "runtime",
                        "status": {"code": "OK"},
                        "attributes": {"http.response.status_code": 200},
                    },
                ],
            )
        )

        self.assertEqual(result.label, "Pass")
        self.assertEqual(result.value, 1.0)
        self.assertIn("input_tokens=120", result.explanation)
        self.assertIn("output_tokens=30", result.explanation)

    def test_fails_error_status_and_explicit_fallback(self) -> None:
        result = EVALUATOR.evaluate_runtime_quality(
            EvaluatorInput(
                evaluation_level="SESSION",
                session_spans=[
                    {
                        "name": "runtime",
                        "status": {"code": "ERROR"},
                        "attributes": {"simuser.neutral_fallback": True},
                    },
                    {
                        "name": "request",
                        "attributes": {"http.response.status_code": 503},
                    },
                ],
            )
        )

        self.assertEqual(result.label, "Fail")
        self.assertEqual(result.value, 0.0)
        self.assertIn("runtime:error-status", result.explanation)
        self.assertIn("runtime:fallback", result.explanation)
        self.assertIn("request:http-5xx", result.explanation)

    def test_supports_otlp_attribute_value_wrappers(self) -> None:
        result = EVALUATOR.evaluate_runtime_quality(
            EvaluatorInput(
                evaluation_level="SESSION",
                session_spans=[
                    {
                        "name": "model",
                        "attributes": [
                            {
                                "key": "gen_ai.usage.input_tokens",
                                "value": {"intValue": 42},
                            },
                            {
                                "key": "simuser.fallback_used",
                                "value": {"boolValue": False},
                            },
                        ],
                    }
                ],
            )
        )

        self.assertEqual(result.label, "Pass")
        self.assertIn("input_tokens=42", result.explanation)

    def test_returns_contract_error_when_spans_are_missing(self) -> None:
        result = EVALUATOR.evaluate_runtime_quality(
            EvaluatorInput(
                evaluation_level="SESSION",
                session_spans=[],
            )
        )

        self.assertEqual(result.errorCode, "NO_SPANS")
        self.assertIsNone(result.label)

    def test_decorated_handler_uses_agentcore_contract(self) -> None:
        response = EVALUATOR.handler(
            {
                "evaluationLevel": "SESSION",
                "evaluationInput": {
                    "sessionSpans": [
                        {
                            "name": "runtime",
                            "status": {"code": "OK"},
                            "attributes": {},
                        }
                    ]
                },
                "evaluationTarget": {},
                "evaluatorName": "PurchaseBehaviorRuntimeHealth",
            },
            None,
        )

        self.assertEqual(response["label"], "Pass")
        self.assertEqual(response["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
