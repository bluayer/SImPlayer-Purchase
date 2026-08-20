from __future__ import annotations

import unittest
from unittest.mock import patch

from purchase_behavior_simulator.bootstrap import (
    build_reflection_provider,
    build_service,
)
from purchase_behavior_simulator.episodic_reasoning import (
    DeterministicReflectionProvider,
    StrandsReflectionProvider,
)


class BootstrapTest(unittest.TestCase):
    def test_service_loads_bundled_action_graph_by_file_name(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "PURCHASE_BEHAVIOR_ACTION_GRAPH": (
                    "game-store-purchase.json"
                )
            },
            clear=True,
        ):
            service = build_service()

        self.assertEqual(
            service.action_graph.graph_id,
            "game_store_purchase",
        )

    def test_reflection_is_deterministic_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            provider = build_reflection_provider()

        self.assertIsInstance(provider, DeterministicReflectionProvider)

    def test_llm_reflection_is_explicit_opt_in(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BEDROCK_MODEL_ID": "openai.gpt-5.6-sol",
                "PURCHASE_BEHAVIOR_REFLECTION_MODE": "llm",
            },
            clear=True,
        ):
            provider = build_reflection_provider()

        self.assertIsInstance(provider, StrandsReflectionProvider)

    def test_llm_reflection_requires_a_model(self) -> None:
        with patch.dict(
            "os.environ",
            {"PURCHASE_BEHAVIOR_REFLECTION_MODE": "llm"},
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                build_reflection_provider()


if __name__ == "__main__":
    unittest.main()
