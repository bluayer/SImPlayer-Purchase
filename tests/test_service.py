from __future__ import annotations

import unittest
from datetime import datetime, timezone

from purchase_behavior_simulator.models import (
    BehaviorEvent,
    EpisodicMemoryEvidence,
    KnowledgeGraphEvidence,
    MemoryDocument,
    ObservationBatch,
    ObservationReceipt,
    ObservedStateTransition,
    SimulationRequest,
)
from purchase_behavior_simulator.episodic_memory import (
    serialize_observed_transitions,
)
from purchase_behavior_simulator.service import BehaviorSimulationService


class FixedMemoryProvider:
    def retrieve(self, user_id, queries, item, now, session_id=None):
        event = BehaviorEvent(
            event_type="purchase",
            timestamp=datetime(2026, 8, 17, tzinfo=timezone.utc),
            item_id="old-upgrade",
            categories=("upgrade",),
        )
        return EpisodicMemoryEvidence(
            queries=tuple(queries),
            documents=(MemoryDocument(content="observed upgrade purchase"),),
            interactions=(event,),
        )

    def record_observations(self, batch, reflection):
        return ObservationReceipt(
            user_id=batch.user_id,
            session_id=batch.session_id,
            event_count=len(batch.events),
            long_term_record_count=2,
            source=batch.source,
            reflection=reflection,
        )


class ContrastiveMemoryProvider:
    def retrieve(self, user_id, queries, item, now, session_id=None):
        def document(action, next_state):
            return MemoryDocument(
                content=serialize_observed_transitions(
                    (
                        ObservedStateTransition(
                            state=(
                                "ITEM_DETAIL"
                                if action in {"PURCHASE", "BACK"}
                                else "ITEM_EXPOSURE"
                            ),
                            action=action,
                            next_state=next_state,
                            timestamp=now,
                            item_id=item.item_id,
                            categories=item.categories,
                        ),
                    )
                )
            )

        return EpisodicMemoryEvidence(
            queries=tuple(queries),
            documents=(
                document("PURCHASE", "PURCHASED"),
                document("BACK", "EXITED"),
            ),
        )

    def record_observations(self, batch, reflection):
        raise NotImplementedError


class CapturingAssessmentProvider:
    def __init__(self):
        self.documents = ()
        self.request = None

    def assess(self, request, memory_evidence):
        from purchase_behavior_simulator.models import AgentAssessment

        self.request = request
        self.documents = tuple(memory_evidence)
        return AgentAssessment()


class FixedActionAssessmentProvider:
    def assess(self, request, memory_evidence):
        from purchase_behavior_simulator.models import AgentAssessment

        return AgentAssessment(
            likelihood=0.2,
            rollout_probability=0.26,
            action_distributions={
                "ITEM_EXPOSURE": {
                    "CLICK": 0.4,
                    "SKIP": 0.3,
                    "EXIT": 0.1,
                    "PURCHASE_NOW": 0.2,
                },
                "ITEM_DETAIL": {
                    "PURCHASE": 0.15,
                    "BACK": 0.65,
                    "EXIT": 0.2,
                },
            },
            decision_state={"feasibility": 0.8},
            intention_distribution={
                "BUY_NOW": 0.2,
                "EXPLORE": 0.4,
                "DEFER": 0.3,
                "REJECT": 0.1,
            },
            confidence=0.7,
        )


class FixedGraphProvider:
    def get_evidence(self, request):
        return KnowledgeGraphEvidence(
            precomputed_affinity=0.8,
            precomputed_confidence=0.4,
            retrieved_evidence=(
                "<simuser-kg-evidence>{\"source_item_id\":\"old\"}</simuser-kg-evidence>",
            ),
            retrieval_support=0.5,
            retrieval_coverage=0.75,
            retrieval_consistency=1.0,
        )


class ServiceTest(unittest.TestCase):
    def test_owned_item_short_circuit_records_non_purchase_actions(self) -> None:
        trace_events = []
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {
                    "item_id": "skin-1",
                    "categories": ["cosmetic"],
                    "price": 1000,
                },
                "game_state": {
                    "currency_balance": 5000,
                    "owned_item_ids": ["skin-1"],
                },
            }
        )

        result = BehaviorSimulationService(
            trace_events=trace_events
        ).simulate(request)

        self.assertFalse(result.eligible)
        self.assertEqual(result.probability, 0.0)
        event = trace_events[-1]
        self.assertEqual(event["stage"], "eligibility_short_circuit")
        self.assertEqual(
            event["action_distributions"]["ITEM_EXPOSURE"]["SKIP"],
            0.98,
        )
        self.assertEqual(
            event["action_distributions"]["ITEM_DETAIL"]["PURCHASE"],
            0.0,
        )

    def test_parses_and_scores_request(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1", "category_preferences": {"skin": 0.8}},
                "item": {"item_id": "skin-1", "categories": ["skin"], "price": 1000},
                "context": {"surface": "character_screen", "budget_reference": 3000},
                "base_model_probability": 0.05,
            }
        )
        result = BehaviorSimulationService().simulate(request)
        self.assertTrue(result.eligible)
        self.assertGreater(result.probability, 0.0)
        self.assertIn("base", result.components)
        self.assertFalse(result.is_calibrated)

    def test_public_result_exposes_both_scores_and_behavior_paths(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "skin-1", "price": 1000},
                "context": {"surface": "store_home"},
            }
        )

        result = BehaviorSimulationService(
            assessment_provider=FixedActionAssessmentProvider()
        ).simulate(request)
        payload = result.to_dict()

        self.assertEqual(
            payload["scalar_purchase_probability"],
            payload["probability"],
        )
        self.assertAlmostEqual(
            payload["trajectory_purchase_probability"],
            0.2 + 0.4 * 0.15,
        )
        self.assertEqual(
            payload["action_distributions"]["ITEM_EXPOSURE"]["CLICK"],
            0.4,
        )
        self.assertTrue(payload["likely_trajectories"])
        self.assertEqual(payload["action_graph_id"], "game_store_purchase")
        self.assertIn("feasibility", payload["decision_state"])

    def test_target_product_scenario_overrides_catalog_values(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "target_product": {
                    "product_id": "bundle-1",
                    "product_type": "bundle",
                    "categories": ["bundle"],
                    "price": 30000,
                    "discount_rate": 0.0,
                    "components": [
                        {"product_id": "currency-1", "quantity": 2},
                        "upgrade-1",
                    ],
                },
                "product_scenario": {
                    "price_override": 24900,
                    "discount_rate_override": 0.17,
                    "add_categories": ["limited_offer"],
                    "attribute_overrides": {"event_id": "summer-2026"},
                },
                "exposure_scenario": {
                    "surface": "store_home",
                    "budget_reference": 40000,
                },
            }
        )

        self.assertEqual(request.item.item_id, "bundle-1")
        self.assertEqual(request.item.product_type, "bundle")
        self.assertEqual(request.item.price, 24900)
        self.assertEqual(request.item.discount_rate, 0.17)
        self.assertEqual(
            request.item.categories,
            ("bundle", "limited_offer"),
        )
        self.assertEqual(
            [component.item_id for component in request.item.components],
            ["currency-1", "upgrade-1"],
        )
        self.assertEqual(request.item.attributes["scenario_base_price"], 30000)
        self.assertEqual(request.context.surface, "store_home")

    def test_target_product_accepts_explicit_need_profile(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "target_product": {
                    "product_id": "mage-summer-bundle",
                    "product_type": "bundle",
                    "categories": ["upgrade", "skin"],
                    "price": 24900,
                    "components": ["upgrade-1", "skin-1"],
                    "need_profile": {
                        "rational": 0.6,
                        "emotional": 0.4,
                        "rational_aspects": ["magic"],
                        "emotional_aspects": ["blue", "summer-style"],
                    },
                },
            }
        )

        self.assertIsNotNone(request.item.need_profile)
        self.assertEqual(request.item.need_profile.rational, 0.6)
        self.assertEqual(
            request.item.need_profile.emotional_aspects,
            ("blue", "summer-style"),
        )

    def test_retrieved_episode_affects_deterministic_episodic_score(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "target", "categories": ["upgrade"]},
                "context": {"timestamp": "2026-08-18T00:00:00+00:00"},
                "base_model_probability": 0.05,
            }
        )

        result = BehaviorSimulationService(
            memory_provider=FixedMemoryProvider()
        ).simulate(request)

        self.assertGreater(result.components["episodic"], 0.5)
        self.assertEqual(result.components["episodic_memory_events"], 1.0)
        self.assertEqual(result.components["episodic_memory_queries"], 4.0)

    def test_record_observations_is_separate_from_prediction(self) -> None:
        service = BehaviorSimulationService(memory_provider=FixedMemoryProvider())
        batch = ObservationBatch.from_dict(
            {
                "user_id": "u1",
                "session_id": "s1",
                "events": [{"event_type": "view"}],
            }
        )

        receipt = service.record_observations(batch)

        self.assertEqual(receipt.event_count, 1)

    def test_evaluation_snapshot_uses_request_graph_evidence(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "eval-u1"},
                "item": {"item_id": "target", "categories": ["upgrade"]},
                "context": {"timestamp": "2026-08-18T00:00:00+00:00"},
                "kg_evidence": {
                    "precomputed_affinity": 0.9,
                    "precomputed_confidence": 1.0,
                },
            }
        )

        result = BehaviorSimulationService(
            memory_provider=FixedMemoryProvider()
        ).evaluate_snapshot(request)

        self.assertEqual(result.components["knowledge_graph"], 0.9)
        self.assertEqual(result.components["knowledge_graph_confidence"], 1.0)

    def test_trace_contains_ranked_memory_metadata_not_full_internal_state(
        self,
    ) -> None:
        trace_events = []
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "eval-u1"},
                "item": {"item_id": "target", "categories": ["upgrade"]},
                "context": {"timestamp": "2026-08-18T00:00:00+00:00"},
            }
        )

        BehaviorSimulationService(
            memory_provider=FixedMemoryProvider(),
            trace_events=trace_events,
        ).evaluate_snapshot(request)

        retrieval = next(
            event
            for event in trace_events
            if event["stage"] == "memory_retrieval"
        )
        document = retrieval["documents"][0]
        self.assertEqual(len(document["document_id"]), 16)
        self.assertEqual(
            document["content_preview"],
            "observed upgrade purchase",
        )
        self.assertNotIn("label", str(trace_events).lower())

    def test_pathsim_evidence_is_forwarded_to_assessment(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "eval-u1"},
                "item": {"item_id": "target", "categories": ["upgrade"]},
                "context": {"timestamp": "2026-08-18T00:00:00+00:00"},
            }
        )
        assessment = CapturingAssessmentProvider()

        result = BehaviorSimulationService(
            memory_provider=FixedMemoryProvider(),
            graph_provider=FixedGraphProvider(),
            assessment_provider=assessment,
        ).simulate(request)

        self.assertTrue(
            any("simuser-kg-evidence" in value for value in assessment.documents)
        )
        self.assertEqual(
            result.components["knowledge_graph_retrieval_quality"],
            0.4,
        )

    def test_purchase_and_non_purchase_memory_reach_assessment(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "eval-u1"},
                "item": {"item_id": "target", "categories": ["upgrade"]},
                "context": {"timestamp": "2026-08-18T00:00:00+00:00"},
            }
        )
        assessment = CapturingAssessmentProvider()

        BehaviorSimulationService(
            memory_provider=ContrastiveMemoryProvider(),
            assessment_provider=assessment,
        ).simulate(request)

        self.assertEqual(len(assessment.documents), 2)
        self.assertTrue(
            any('"action":"PURCHASE"' in value for value in assessment.documents)
        )
        self.assertTrue(
            any('"action":"BACK"' in value for value in assessment.documents)
        )


if __name__ == "__main__":
    unittest.main()
