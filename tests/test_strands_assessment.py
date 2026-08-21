from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from purchase_behavior_simulator.action_rollout import (
    rollout_purchase_probability,
)

from purchase_behavior_simulator.action_rollout import ActionGraph
from purchase_behavior_simulator.episodic_memory import (
    serialize_observed_transitions,
)
from purchase_behavior_simulator.models import (
    ObservedStateTransition,
    SimulationRequest,
)
from purchase_behavior_simulator.strands_assessment import StrandsAssessmentProvider


class StrandsAssessmentTest(unittest.TestCase):
    def test_custom_action_graph_drives_payload_and_output_mapping(self) -> None:
        graph = ActionGraph.from_dict(
            {
                "graph_id": "popup",
                "version": "1",
                "default_initial_state": "OFFER",
                "state_output_fields": {
                    "OFFER": "offer",
                    "CONFIRM": "confirm",
                },
                "terminal_outcomes": {
                    "PURCHASED": "purchase",
                    "CLOSED": "exit",
                },
                "transitions": [
                    {
                        "state": "OFFER",
                        "action": "OPEN",
                        "next_state": "CONFIRM",
                    },
                    {
                        "state": "OFFER",
                        "action": "CLOSE",
                        "next_state": "CLOSED",
                    },
                    {
                        "state": "CONFIRM",
                        "action": "BUY",
                        "next_state": "PURCHASED",
                    },
                    {
                        "state": "CONFIRM",
                        "action": "CANCEL",
                        "next_state": "CLOSED",
                    },
                ],
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="openai.test",
            mode="actions",
            action_graph=graph,
        )
        request = SimulationRequest.from_dict(
            {"user": {"user_id": "u1"}, "item": {"item_id": "i1"}}
        )

        payload = provider._rollout_round_payload(
            request=request,
            new_memory_evidence=(),
            previous_assessment=None,
            round_number=1,
            remaining_document_count=0,
        )
        distributions = provider._action_distributions(
            SimpleNamespace(
                offer=SimpleNamespace(open=0.4, close=0.6),
                confirm=SimpleNamespace(buy=0.25, cancel=0.75),
            )
        )

        self.assertEqual(
            payload["deterministic_environment"]["transitions"]["OFFER"],
            {"OPEN": "CONFIRM", "CLOSE": "CLOSED"},
        )
        self.assertEqual(distributions["CONFIRM"]["BUY"], 0.25)

    def test_agent_factory_initialization_failure_returns_neutral(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )

        def failing_factory():
            raise ImportError("openai SDK unavailable")

        provider = StrandsAssessmentProvider(
            model_id="openai.test",
            agent_factory=failing_factory,
            mode="probability",
        )

        result = provider.assess(request, ())

        self.assertEqual(result.likelihood, 0.5)
        self.assertEqual(result.relative_preference_score, 0.5)
        self.assertEqual(result.confidence, 0.0)

    def test_rollout_factory_initialization_failure_returns_neutral(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )

        def failing_factory():
            raise ImportError("openai SDK unavailable")

        provider = StrandsAssessmentProvider(
            model_id="openai.test",
            agent_factory=failing_factory,
            mode="actions",
        )

        result = provider.assess(request, ())

        self.assertIsNotNone(result.rollout_probability)
        self.assertEqual(result.confidence, 0.0)

    def test_agent_payload_excludes_model_prediction_and_synthetic_label(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
                "base_model_probability": 0.99,
                "agent_assessment": {
                    "likelihood": 0.99,
                    "confidence": 1.0,
                },
            }
        )
        payload = StrandsAssessmentProvider._payload(request, ())

        self.assertNotIn("base_model_probability", payload)
        self.assertNotIn("agent_assessment", payload)
        self.assertNotIn("purchased", payload)

    def test_payload_marks_repeat_purchase_semantics(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {
                    "item_id": "i1",
                    "categories": ["currency"],
                },
            }
        )

        payload = StrandsAssessmentProvider._payload(request, ())

        self.assertTrue(
            payload["item"]["purchase_semantics"]["repeatable_likely"]
        )

    def test_payload_disentangles_need_profile_without_routing_by_type(self) -> None:
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {
                    "item_id": "mixed-bundle",
                    "product_type": "bundle",
                    "categories": ["upgrade", "skin"],
                    "components": ["upgrade-1", "skin-1"],
                },
            }
        )

        payload = StrandsAssessmentProvider._payload(request, ())

        self.assertEqual(
            payload["item"]["need_profile"]["dominant_need"],
            "mixed",
        )
        self.assertIn("bundle_rule", payload["item"]["need_assessment_contract"])

    def test_reviews_memory_in_at_most_three_rounds(self) -> None:
        calls = []

        def fake_agent(prompt, **kwargs):
            calls.append(kwargs)
            round_number = len(calls)
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=round_number == 3,
                    likelihood=0.6,
                    confidence=0.5,
                    reasons=["observed evidence"],
                    contradictions=["alternative cause"],
                    additional_memory_question="more history",
                )
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: fake_agent,
        )

        provider.assess(request, tuple(f"memory-{index}" for index in range(12)))

        self.assertEqual(len(calls), 3)

    def test_stops_when_the_first_round_is_sufficient(self) -> None:
        calls = []

        def fake_agent(prompt, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=True,
                    likelihood=0.5,
                    confidence=0.2,
                    reasons=[],
                    contradictions=[],
                    additional_memory_question="",
                )
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: fake_agent,
        )

        provider.assess(request, ("m1", "m2", "m3", "m4"))

        self.assertEqual(len(calls), 1)

    def test_probability_returns_separate_natural_and_ranking_scores(self) -> None:
        def fake_agent(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=True,
                    likelihood=0.08,
                    relative_preference_score=0.73,
                    confidence=0.6,
                    reasons=["rare purchase but strong relative fit"],
                    contradictions=[],
                    additional_memory_question="",
                )
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: fake_agent,
            mode="probability",
        )

        result = provider.assess(request, ("m1",))

        self.assertEqual(result.likelihood, 0.08)
        self.assertEqual(result.relative_preference_score, 0.73)

    def test_action_computes_probability_from_validated_action_paths(self) -> None:
        def actor(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=True,
                    relative_preference_score=0.7,
                    confidence=0.6,
                    exposure=SimpleNamespace(
                        click=0.4,
                        skip=0.3,
                        exit=0.1,
                        purchase_now=0.2,
                    ),
                    detail=SimpleNamespace(
                        purchase=0.2,
                        back=0.5,
                        exit=0.3,
                    ),
                    counterfactuals=[],
                    reasons=["path comparison"],
                    contradictions=[],
                    additional_memory_question="",
                )
            )

        def critic(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    persona_consistent=True,
                    recent_behavior_consistent=True,
                    repeat_purchase_valid=True,
                    price_budget_consistent=True,
                    weak_action_sequence_valid=True,
                    exposure=SimpleNamespace(
                        click=0.4,
                        skip=0.3,
                        exit=0.1,
                        purchase_now=0.2,
                    ),
                    detail=SimpleNamespace(
                        purchase=0.2,
                        back=0.5,
                        exit=0.3,
                    ),
                    confidence=0.65,
                    contradictions=[],
                    revision_summary="",
                )
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: actor,
            critic_agent_factory=lambda: critic,
            mode="actions",
        )

        result = provider.assess(request, ("m1",))

        self.assertAlmostEqual(
            result.rollout_probability,
            rollout_purchase_probability(
                result.action_distributions,
                surface="store_home",
            ).purchase_probability,
        )
        self.assertEqual(result.likelihood, result.rollout_probability)
        self.assertFalse(result.validator_adjusted)
        self.assertFalse(result.commitment_adjusted)
        self.assertIn("DEFER", result.intention_distribution)
        self.assertIn("commitment_strength", result.decision_state)

    def test_action_second_round_serializes_only_public_decision_state(self) -> None:
        prompts = []

        def actor(prompt, **kwargs):
            prompts.append(prompt)
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=len(prompts) >= 2,
                    relative_preference_score=0.6,
                    confidence=0.7,
                    exposure=SimpleNamespace(
                        click=0.3,
                        skip=0.5,
                        exit=0.1,
                        purchase_now=0.1,
                    ),
                    detail=SimpleNamespace(
                        purchase=0.2,
                        back=0.6,
                        exit=0.2,
                    ),
                    decision_state=SimpleNamespace(
                        need_strength=0.6,
                        selection_strength=0.6,
                        feasibility=0.5,
                        urgency=0.4,
                        uncertainty=0.3,
                        hesitation=0.5,
                    ),
                    intentions=SimpleNamespace(
                        buy_now=0.1,
                        explore=0.3,
                        defer=0.5,
                        reject=0.1,
                    ),
                    counterfactuals=[],
                    reasons=["two-round comparison"],
                    contradictions=[],
                    additional_memory_question="compare another outcome",
                )
            )

        def critic(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    persona_consistent=True,
                    recent_behavior_consistent=True,
                    repeat_purchase_valid=True,
                    price_budget_consistent=True,
                    weak_action_sequence_valid=True,
                    exposure=SimpleNamespace(
                        click=0.3,
                        skip=0.5,
                        exit=0.1,
                        purchase_now=0.1,
                    ),
                    detail=SimpleNamespace(
                        purchase=0.2,
                        back=0.6,
                        exit=0.2,
                    ),
                    confidence=0.7,
                    contradictions=[],
                    revision_summary="",
                )
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        result = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: actor,
            critic_agent_factory=lambda: critic,
            mode="actions",
        ).assess(request, ("m1", "m2", "m3", "m4"))

        self.assertEqual(len(prompts), 2)
        self.assertNotIn("decision_state_object", prompts[1])
        self.assertIn('"decision_state"', prompts[1])
        self.assertIsNotNone(result.rollout_probability)

    def test_action_validator_can_reduce_direct_purchase_without_rewriting_reasoning(
        self,
    ) -> None:
        trace_events = []

        def actor(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=True,
                    relative_preference_score=0.8,
                    confidence=0.8,
                    exposure=SimpleNamespace(
                        click=0.1,
                        skip=0.1,
                        exit=0.0,
                        purchase_now=0.8,
                    ),
                    detail=SimpleNamespace(
                        purchase=0.8,
                        back=0.1,
                        exit=0.1,
                    ),
                    counterfactuals=[],
                    reasons=["strong fit"],
                    contradictions=[],
                    additional_memory_question="",
                )
            )

        def critic(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    persona_consistent=True,
                    recent_behavior_consistent=False,
                    repeat_purchase_valid=True,
                    price_budget_consistent=False,
                    weak_action_sequence_valid=False,
                    exposure=SimpleNamespace(
                        click=0.3,
                        skip=0.4,
                        exit=0.2,
                        purchase_now=0.1,
                    ),
                    detail=SimpleNamespace(
                        purchase=0.1,
                        back=0.5,
                        exit=0.4,
                    ),
                    confidence=0.6,
                    contradictions=["price exceeds budget"],
                    revision_summary="distribution revised once",
                )
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1", "price": 200},
                "context": {"budget_reference": 100},
            }
        )
        result = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: actor,
            critic_agent_factory=lambda: critic,
            trace_events=trace_events,
            mode="actions",
        ).assess(request, ("m1",))

        self.assertTrue(result.validator_adjusted)
        original_rollout = 0.8 + 0.1 * 0.8
        self.assertLess(result.rollout_probability, original_rollout)
        self.assertGreater(result.rollout_probability, 0.0)
        self.assertIn("price exceeds budget", result.contradictions)
        self.assertEqual(
            [event["stage"] for event in trace_events],
            [
                "action_assessment_round",
                "transition_grounding",
                "commitment_gate",
                "action_validator",
                "counterfactual_validator",
            ],
        )

    def test_action_recent_behavior_critique_is_advisory_after_grounding(
        self,
    ) -> None:
        trace_events = []

        def actor(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=True,
                    relative_preference_score=0.5,
                    confidence=0.7,
                    exposure=SimpleNamespace(
                        click=0.3,
                        skip=0.5,
                        exit=0.1,
                        purchase_now=0.1,
                    ),
                    detail=SimpleNamespace(
                        purchase=0.3,
                        back=0.6,
                        exit=0.1,
                    ),
                    counterfactuals=[],
                    reasons=[],
                    contradictions=[],
                    additional_memory_question="",
                )
            )

        def critic(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    persona_consistent=True,
                    recent_behavior_consistent=False,
                    repeat_purchase_valid=True,
                    price_budget_consistent=True,
                    weak_action_sequence_valid=False,
                    exposure=SimpleNamespace(
                        click=0.1,
                        skip=0.8,
                        exit=0.05,
                        purchase_now=0.05,
                    ),
                    detail=SimpleNamespace(
                        purchase=0.1,
                        back=0.8,
                        exit=0.1,
                    ),
                    confidence=0.6,
                    contradictions=["recent paths stopped before purchase"],
                    revision_summary="advisory critique only",
                )
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        result = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: actor,
            critic_agent_factory=lambda: critic,
            trace_events=trace_events,
            mode="actions",
        ).assess(request, ("m1",))

        self.assertFalse(result.validator_adjusted)
        validator = next(
            event
            for event in trace_events
            if event["stage"] == "action_validator"
        )
        self.assertFalse(validator["hard_revision_eligible"])
        self.assertFalse(
            validator["advisory_checks"]["recent_behavior_consistent"]
        )
        self.assertIn(
            "recent paths stopped before purchase",
            result.contradictions,
        )

    def test_action_grounding_is_not_reported_as_validator_adjustment(self) -> None:
        def actor(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=True,
                    relative_preference_score=0.5,
                    confidence=0.7,
                    exposure=SimpleNamespace(
                        click=0.1,
                        skip=0.1,
                        exit=0.0,
                        purchase_now=0.8,
                    ),
                    detail=SimpleNamespace(
                        purchase=0.8,
                        back=0.1,
                        exit=0.1,
                    ),
                    counterfactuals=[],
                    reasons=[],
                    contradictions=[],
                    additional_memory_question="",
                )
            )

        def failing_critic(prompt, **kwargs):
            raise ValueError("critic unavailable")

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1", "categories": ["upgrade"]},
            }
        )
        memory = serialize_observed_transitions(
            (
                ObservedStateTransition(
                    state="ITEM_EXPOSURE",
                    action="SKIP",
                    next_state="ITEM_EXPOSURE",
                    timestamp=datetime(2026, 8, 17, tzinfo=timezone.utc),
                    item_id="i1",
                    categories=("upgrade",),
                ),
            )
        )

        result = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: actor,
            critic_agent_factory=lambda: failing_critic,
            mode="actions",
        ).assess(request, (memory,))

        self.assertFalse(result.validator_adjusted)
        self.assertLess(result.rollout_probability, 0.872)

    def test_first_round_failure_returns_neutral_assessment(self) -> None:
        def failing_agent(prompt, **kwargs):
            raise ValueError("invalid structured output")

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: failing_agent,
        )

        with self.assertLogs(
            "purchase_behavior_simulator.strands_assessment",
            level="WARNING",
        ):
            result = provider.assess(request, ("m1",))

        self.assertEqual(result.likelihood, 0.5)
        self.assertEqual(result.confidence, 0.0)

    def test_probability_failure_returns_neutral_ranking_score(self) -> None:
        def failing_agent(prompt, **kwargs):
            raise ValueError("invalid structured output")

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: failing_agent,
            mode="probability",
        )

        with self.assertLogs(
            "purchase_behavior_simulator.strands_assessment",
            level="WARNING",
        ):
            result = provider.assess(request, ("m1",))

        self.assertEqual(result.likelihood, 0.5)
        self.assertEqual(result.relative_preference_score, 0.5)

    def test_evaluation_can_include_sanitized_failure_detail(self) -> None:
        def failing_agent(prompt, **kwargs):
            raise ValueError("invalid\nstructured output")

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: failing_agent,
            include_failure_details=True,
        )

        with self.assertLogs(
            "purchase_behavior_simulator.strands_assessment",
            level="WARNING",
        ):
            result = provider.assess(request, ("m1",))

        self.assertIn(
            "ValueError: invalid structured output",
            result.reasons[0],
        )

    def test_later_round_failure_keeps_previous_assessment(self) -> None:
        calls = 0

        def flaky_agent(prompt, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("invalid structured output")
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=False,
                    likelihood=0.65,
                    confidence=0.4,
                    reasons=["first round"],
                    contradictions=["uncertain"],
                    additional_memory_question="more",
                )
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: flaky_agent,
        )

        with self.assertLogs(
            "purchase_behavior_simulator.strands_assessment",
            level="WARNING",
        ):
            result = provider.assess(request, ("m1", "m2", "m3", "m4"))

        self.assertEqual(result.likelihood, 0.65)
        self.assertEqual(result.confidence, 0.4)

    def test_observable_trace_excludes_raw_model_message(self) -> None:
        trace_events = []

        def fake_agent(prompt, **kwargs):
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=True,
                    likelihood=0.62,
                    confidence=0.51,
                    reasons=["structured reason"],
                    contradictions=["structured contradiction"],
                    additional_memory_question="",
                ),
                stop_reason="end_turn",
                message={"content": [{"reasoningContent": "private reasoning"}]},
                metrics=SimpleNamespace(
                    cycle_count=1,
                    accumulated_usage={
                        "inputTokens": 100,
                        "outputTokens": 20,
                        "totalTokens": 120,
                    },
                    accumulated_metrics={"latencyMs": 250},
                ),
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: fake_agent,
            trace_events=trace_events,
        )

        provider.assess(request, ("m1",))

        self.assertEqual(trace_events[0]["likelihood"], 0.62)
        self.assertEqual(
            trace_events[0]["metrics"]["usage"]["totalTokens"],
            120,
        )
        self.assertNotIn("private reasoning", str(trace_events))

    def test_retries_transient_mantle_token_mint_failure(self) -> None:
        trace_events = []
        calls = 0

        def flaky_agent(prompt, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError(
                    "Failed to mint Bedrock Mantle bearer token for region"
                )
            return SimpleNamespace(
                structured_output=SimpleNamespace(
                    sufficient_evidence=True,
                    likelihood=0.55,
                    confidence=0.4,
                    reasons=[],
                    contradictions=[],
                    additional_memory_question="",
                ),
                stop_reason="end_turn",
                metrics=SimpleNamespace(
                    cycle_count=1,
                    accumulated_usage={
                        "inputTokens": 10,
                        "outputTokens": 5,
                        "totalTokens": 15,
                    },
                    accumulated_metrics={"latencyMs": 20},
                ),
            )

        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "u1"},
                "item": {"item_id": "i1"},
            }
        )
        provider = StrandsAssessmentProvider(
            model_id="test-model",
            agent_factory=lambda: flaky_agent,
            trace_events=trace_events,
        )

        result = provider.assess(request, ("m1",))

        self.assertEqual(result.likelihood, 0.55)
        self.assertEqual(calls, 2)
        self.assertEqual(trace_events[0]["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
