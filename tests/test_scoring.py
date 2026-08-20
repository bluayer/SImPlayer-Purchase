from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from purchase_behavior_simulator.models import (
    AgentAssessment,
    BehaviorEvent,
    Item,
    KnowledgeGraphEvidence,
    ExposureContext,
    UserProfile,
)
from purchase_behavior_simulator.scoring import (
    PlattCalibration,
    BehaviorSimulationScorer,
    ScoringConfig,
    episodic_affinity,
    logit,
    pathsim,
    sigmoid,
)


NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


class ScoringTest(unittest.TestCase):
    def test_pathsim_matches_paper_formula(self) -> None:
        self.assertAlmostEqual(pathsim(4, 8, 10), 8 / 18)

    def test_recent_positive_behavior_increases_affinity(self) -> None:
        item = Item(item_id="target", categories=("upgrade",))
        recent = BehaviorEvent(
            event_type="purchase",
            timestamp=NOW - timedelta(days=1),
            item_id="other",
            categories=("upgrade",),
        )
        old = BehaviorEvent(
            event_type="purchase",
            timestamp=NOW - timedelta(days=120),
            item_id="other",
            categories=("upgrade",),
        )
        scorer = BehaviorSimulationScorer()
        recent_score, _ = episodic_affinity((recent,), item, NOW, scorer.config)
        old_score, _ = episodic_affinity((old,), item, NOW, scorer.config)
        self.assertGreaterEqual(recent_score, old_score)

    def test_refund_with_low_rating_is_negative_evidence(self) -> None:
        item = Item(item_id="target", categories=("upgrade",))
        refund = BehaviorEvent(
            event_type="refund",
            timestamp=NOW - timedelta(days=1),
            item_id="other",
            categories=("upgrade",),
            rating=1,
        )

        score, confidence = episodic_affinity(
            (refund,),
            item,
            NOW,
            BehaviorSimulationScorer().config,
        )

        self.assertLess(score, 0.5)
        self.assertGreater(confidence, 0.0)

    def test_negative_episode_offsets_positive_episode(self) -> None:
        item = Item(item_id="target", categories=("upgrade",))
        purchase = BehaviorEvent(
            event_type="purchase",
            timestamp=NOW - timedelta(days=2),
            item_id="positive",
            categories=("upgrade",),
            rating=5,
        )
        refund = BehaviorEvent(
            event_type="refund",
            timestamp=NOW - timedelta(days=1),
            item_id="negative",
            categories=("upgrade",),
            rating=1,
        )
        config = BehaviorSimulationScorer().config

        positive_score, _ = episodic_affinity((purchase,), item, NOW, config)
        mixed_score, _ = episodic_affinity((purchase, refund), item, NOW, config)

        self.assertLess(mixed_score, positive_score)

    def test_positive_evidence_keeps_probability_bounded_and_raises_base(self) -> None:
        scorer = BehaviorSimulationScorer()
        result = scorer.score(
            user=UserProfile(
                user_id="u1",
                category_preferences={"upgrade": 0.9},
                pickiness=0.3,
            ),
            item=Item(
                item_id="i1",
                categories=("upgrade",),
                price=5000,
                discount_rate=0.2,
            ),
            context=ExposureContext(
                surface="checkout",
                budget_reference=10000,
                timestamp=NOW,
            ),
            interactions=(
                BehaviorEvent(
                    event_type="purchase",
                    timestamp=NOW - timedelta(days=2),
                    item_id="i0",
                    categories=("upgrade",),
                ),
            ),
            kg_evidence=KnowledgeGraphEvidence(
                item_shared_paths=4,
                item_source_self_paths=5,
                item_target_self_paths=5,
                user_shared_paths=4,
                user_self_paths=5,
                user_target_self_paths=5,
            ),
            base_model_probability=0.08,
            agent_assessment=AgentAssessment(likelihood=0.8, confidence=0.8),
        )
        self.assertGreater(result.probability, 0.08)
        self.assertLessEqual(result.probability, 1.0)

    def test_ineligible_item_has_zero_probability(self) -> None:
        result = BehaviorSimulationScorer().score(
            user=UserProfile(user_id="u1"),
            item=Item(item_id="i1", eligible=False),
            context=ExposureContext(timestamp=NOW),
            interactions=(),
            kg_evidence=KnowledgeGraphEvidence(),
            base_model_probability=0.2,
            agent_assessment=AgentAssessment(likelihood=0.9, confidence=1.0),
        )
        self.assertEqual(result.probability, 0.0)
        self.assertFalse(result.eligible)

    def test_calibration_flag_requires_version(self) -> None:
        scorer = BehaviorSimulationScorer(
            calibration=PlattCalibration(slope=0.8, intercept=-0.1, version="cal-1")
        )
        result = scorer.score(
            user=UserProfile(user_id="u1"),
            item=Item(item_id="i1"),
            context=ExposureContext(timestamp=NOW),
            interactions=(),
            kg_evidence=KnowledgeGraphEvidence(),
            base_model_probability=0.2,
            agent_assessment=AgentAssessment(),
        )
        self.assertTrue(result.is_calibrated)
        self.assertEqual(result.calibration_version, "cal-1")

    def test_base_referenced_agent_does_not_double_count_same_probability(
        self,
    ) -> None:
        common = {
            "user": UserProfile(user_id="u1"),
            "item": Item(item_id="i1"),
            "context": ExposureContext(timestamp=NOW),
            "interactions": (),
            "kg_evidence": KnowledgeGraphEvidence(),
            "base_model_probability": 0.1,
            "agent_assessment": AgentAssessment(
                likelihood=0.1,
                confidence=1.0,
            ),
        }
        without_agent_weight = BehaviorSimulationScorer(
            config=ScoringConfig(
                agent_logit_weight=0.0,
                agent_logit_reference="base",
            )
        ).score(**common)
        relative_agent = BehaviorSimulationScorer(
            config=ScoringConfig(agent_logit_reference="base")
        ).score(**common)

        self.assertEqual(
            relative_agent.probability,
            without_agent_weight.probability,
        )

    def test_relative_preference_is_reported_but_not_used_as_probability(
        self,
    ) -> None:
        common = {
            "user": UserProfile(user_id="u1"),
            "item": Item(item_id="i1"),
            "context": ExposureContext(timestamp=NOW),
            "interactions": (),
            "kg_evidence": KnowledgeGraphEvidence(),
            "base_model_probability": 0.1,
        }
        low_rank = BehaviorSimulationScorer(
            config=ScoringConfig(agent_logit_reference="base")
        ).score(
            **common,
            agent_assessment=AgentAssessment(
                likelihood=0.08,
                relative_preference_score=0.2,
                confidence=1.0,
            ),
        )
        high_rank = BehaviorSimulationScorer(
            config=ScoringConfig(agent_logit_reference="base")
        ).score(
            **common,
            agent_assessment=AgentAssessment(
                likelihood=0.08,
                relative_preference_score=0.9,
                confidence=1.0,
            ),
        )

        self.assertEqual(low_rank.probability, high_rank.probability)
        self.assertEqual(
            high_rank.components["agent_ranking_score"],
            0.9,
        )

    def test_rollout_is_a_separate_base_referenced_fusion_component(self) -> None:
        common = {
            "user": UserProfile(user_id="u1"),
            "item": Item(item_id="i1"),
            "context": ExposureContext(timestamp=NOW),
            "interactions": (),
            "kg_evidence": KnowledgeGraphEvidence(),
            "base_model_probability": 0.1,
        }
        neutral = BehaviorSimulationScorer(
            config=ScoringConfig(
                agent_logit_weight=0.0,
                rollout_logit_weight=0.5,
                rollout_logit_reference="base",
            )
        ).score(
            **common,
            agent_assessment=AgentAssessment(
                likelihood=0.1,
                rollout_probability=0.1,
                confidence=1.0,
            ),
        )
        stronger = BehaviorSimulationScorer(
            config=ScoringConfig(
                agent_logit_weight=0.0,
                rollout_logit_weight=0.5,
                rollout_logit_reference="base",
            )
        ).score(
            **common,
            agent_assessment=AgentAssessment(
                likelihood=0.3,
                rollout_probability=0.3,
                confidence=1.0,
                validator_adjusted=True,
            ),
        )

        self.assertGreater(stronger.probability, neutral.probability)
        self.assertEqual(stronger.components["rollout"], 0.3)
        self.assertEqual(
            stronger.components["rollout_validator_adjusted"],
            1.0,
        )

    def test_action_output_keeps_rollout_primary_and_fusion_as_calibration(self) -> None:
        result = BehaviorSimulationScorer(
            config=ScoringConfig(
                agent_logit_weight=0.0,
                rollout_logit_weight=0.5,
                rollout_logit_reference="base",
                rollout_output_blend=0.75,
            )
        ).score(
            user=UserProfile(user_id="u1"),
            item=Item(item_id="i1"),
            context=ExposureContext(timestamp=NOW),
            interactions=(),
            kg_evidence=KnowledgeGraphEvidence(),
            base_model_probability=0.1,
            agent_assessment=AgentAssessment(
                likelihood=0.2,
                rollout_probability=0.2,
                confidence=1.0,
            ),
        )

        expected = (
            0.25 * result.components["raw_fusion"]
            + 0.75 * result.components["rollout"]
        )
        self.assertAlmostEqual(result.probability, expected, places=5)
        self.assertEqual(result.components["rollout_output_blend"], 0.75)

    def test_hybrid_uses_separate_rollout_confidence_and_prior_shrink(
        self,
    ) -> None:
        result = BehaviorSimulationScorer(
            config=ScoringConfig(
                episodic_logit_weight=0.0,
                kg_logit_weight=0.0,
                context_logit_weight=0.0,
                agent_logit_weight=0.0,
                rollout_logit_weight=0.1,
                rollout_logit_reference="base",
                fusion_logit_shrink=0.5,
                fusion_prior_anchor=0.1,
            )
        ).score(
            user=UserProfile(user_id="u1"),
            item=Item(item_id="i1"),
            context=ExposureContext(timestamp=NOW),
            interactions=(),
            kg_evidence=KnowledgeGraphEvidence(),
            base_model_probability=0.1,
            agent_assessment=AgentAssessment(
                likelihood=0.1,
                rollout_probability=0.4,
                confidence=0.0,
                rollout_confidence=1.0,
            ),
        )

        rollout_fused = sigmoid(
            logit(0.1)
            + 0.1 * (logit(0.4) - logit(0.1))
        )
        expected = sigmoid(
            logit(0.1)
            + 0.5 * (logit(rollout_fused) - logit(0.1))
        )
        self.assertAlmostEqual(result.probability, expected, places=5)
        self.assertEqual(result.components["rollout_confidence"], 1.0)
        self.assertEqual(result.components["fusion_logit_shrink"], 0.5)


if __name__ == "__main__":
    unittest.main()
