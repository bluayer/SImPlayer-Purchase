from __future__ import annotations

import unittest

from purchase_behavior_simulator.hybrid_assessment import (
    HybridAssessmentProvider,
)
from purchase_behavior_simulator.models import (
    AgentAssessment,
    SimulationRequest,
)


class FixedAssessmentProvider:
    def __init__(self, assessment: AgentAssessment) -> None:
        self.assessment = assessment

    def assess(self, request, memory_evidence):
        return self.assessment


class HybridAssessmentTest(unittest.TestCase):
    def test_preserves_probability_assessment_and_adds_rollout_state(self) -> None:
        trace_events = []
        probability = AgentAssessment(
            likelihood=0.08,
            relative_preference_score=0.7,
            confidence=0.6,
            reasons=("legacy probability reason",),
            contradictions=("legacy contradiction",),
        )
        rollout = AgentAssessment(
            likelihood=0.2,
            rollout_probability=0.2,
            action_distributions={
                "ITEM_EXPOSURE": {
                    "CLICK": 0.4,
                    "SKIP": 0.4,
                    "EXIT": 0.1,
                    "PURCHASE_NOW": 0.1,
                }
            },
            validator_adjusted=True,
            transition_grounding_strength=0.4,
            confidence=0.9,
            reasons=("ephemeral rollout reason",),
            contradictions=("rollout contradiction",),
        )
        provider = HybridAssessmentProvider(
            probability_provider=FixedAssessmentProvider(probability),
            rollout_provider=FixedAssessmentProvider(rollout),
            trace_events=trace_events,
        )

        result = provider.assess(
            SimulationRequest.from_dict(
                {
                    "user": {"user_id": "u1"},
                    "item": {"item_id": "i1"},
                }
            ),
            ("observed memory",),
        )

        self.assertEqual(result.likelihood, 0.08)
        self.assertEqual(result.relative_preference_score, 0.7)
        self.assertEqual(result.confidence, 0.6)
        self.assertEqual(result.rollout_probability, 0.2)
        self.assertEqual(result.rollout_confidence, 0.9)
        self.assertEqual(
            result.reasons,
            ("legacy probability reason",),
        )
        self.assertEqual(
            result.contradictions,
            ("legacy contradiction", "rollout contradiction"),
        )
        self.assertEqual(trace_events[-1]["stage"], "hybrid_assessment")


if __name__ == "__main__":
    unittest.main()
