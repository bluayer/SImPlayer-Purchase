from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from purchase_behavior_simulator.model_factory import (
    _install_jiter_import_fallback,
    effective_max_tokens,
    is_openai_bedrock_model,
    mantle_model_id,
    supports_temperature,
)


class ModelFactoryTest(unittest.TestCase):
    def test_routes_openai_models_to_mantle(self) -> None:
        self.assertTrue(is_openai_bedrock_model("openai.gpt-5.6-terra"))
        self.assertTrue(is_openai_bedrock_model("global.openai.gpt-5.6-sol"))
        self.assertTrue(is_openai_bedrock_model("openai.gpt-5.6-luna"))
        self.assertFalse(
            is_openai_bedrock_model("global.anthropic.claude-opus-5")
        )

    def test_removes_inference_profile_prefix_for_mantle(self) -> None:
        self.assertEqual(
            mantle_model_id("global.openai.gpt-5.6-terra"),
            "openai.gpt-5.6-terra",
        )
        self.assertEqual(
            mantle_model_id("openai.gpt-5.6-sol"),
            "openai.gpt-5.6-sol",
        )

    def test_omits_deprecated_temperature_for_claude_opus_5(self) -> None:
        self.assertFalse(
            supports_temperature("global.anthropic.claude-opus-5")
        )
        self.assertTrue(
            supports_temperature("amazon.nova-micro-v1:0")
        )
        self.assertEqual(
            effective_max_tokens("global.anthropic.claude-opus-5", 700),
            2048,
        )
        self.assertEqual(
            effective_max_tokens("openai.gpt-5.6-sol", 700),
            700,
        )

    def test_installs_strict_stdlib_fallback_when_jiter_is_absent(self) -> None:
        real_jiter = sys.modules.pop("jiter", None)
        try:
            with patch(
                "purchase_behavior_simulator.model_factory.importlib.import_module",
                side_effect=ModuleNotFoundError(
                    "No module named 'jiter'",
                    name="jiter",
                ),
            ):
                _install_jiter_import_fallback()
            fallback = sys.modules["jiter"]
            self.assertEqual(
                fallback.from_json(b'{"value": 3}'),
                {"value": 3},
            )
            with self.assertRaises(RuntimeError):
                fallback.from_json(b'{"value":', partial_mode=True)
        finally:
            sys.modules.pop("jiter", None)
            if real_jiter is not None:
                sys.modules["jiter"] = real_jiter


if __name__ == "__main__":
    unittest.main()
