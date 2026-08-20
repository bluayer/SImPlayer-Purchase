from __future__ import annotations

import unittest
from unittest.mock import patch

from purchase_behavior_simulator.episodic_reasoning import (
    StrandsReflectionProvider,
    StrandsSelfAskQueryPlanner,
)
from purchase_behavior_simulator.models import ObservationBatch, SimulationRequest


class EpisodicReasoningTest(unittest.TestCase):
    def test_self_ask_model_initialization_failure_uses_fallback(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1", "categories": ["upgrade"]},
            }
        )
        planner = StrandsSelfAskQueryPlanner(model_id="openai.test")

        with patch(
            "purchase_behavior_simulator.episodic_reasoning.build_strands_model",
            side_effect=ImportError("openai SDK unavailable"),
        ):
            questions = planner.plan(request, "initial")

        self.assertEqual(len(questions), 3)

    def test_reflection_model_initialization_failure_uses_fallback(self) -> None:
        batch = ObservationBatch.from_dict(
            {
                "user_id": "u1",
                "session_id": "s1",
                "events": [{"event_type": "purchase", "item_id": "i1"}],
            }
        )
        provider = StrandsReflectionProvider(model_id="openai.test")

        with patch(
            "purchase_behavior_simulator.episodic_reasoning.build_strands_model",
            side_effect=ImportError("openai SDK unavailable"),
        ):
            reflection = provider.reflect(batch)

        self.assertIn("purchase", reflection)

    def test_self_ask_payload_excludes_prediction_and_label(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
                "base_model_probability": 0.99,
                "agent_assessment": {"likelihood": 0.99},
            }
        )

        payload = StrandsSelfAskQueryPlanner._payload(request, "initial")
        text = str(payload).lower()

        self.assertNotIn("base_model_probability", text)
        self.assertNotIn("agent_assessment", text)
        self.assertNotIn("synthetic_label", text)

    def test_reflection_payload_contains_only_external_observation(self) -> None:
        batch = ObservationBatch.from_dict(
            {
                "user_id": "u1",
                "session_id": "s1",
                "events": [{"event_type": "refund", "item_id": "i1"}],
            }
        )

        payload = StrandsReflectionProvider._payload(batch)
        text = str(payload).lower()

        self.assertIn("refund", text)
        self.assertNotIn("probability", text)
        self.assertNotIn("prediction", text)


if __name__ == "__main__":
    unittest.main()
