from __future__ import annotations

import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from .action_rollout import (
    ActionGraph,
    DEFAULT_ACTION_GRAPH,
    blend_action_distributions,
    limit_action_distribution_revision,
    normalize_action_distributions,
    rollout_purchase_probability,
    transition_table_payload,
)
from .decision_process import (
    apply_commitment_gate,
    build_decision_state,
    evaluate_counterfactual_consistency,
    intentions_from_state,
    merge_model_decision,
)
from .episodic_memory import empirical_transition_policy
from .evaluation_trace import (
    TraceEvents,
    invoke_with_transient_retries,
    sanitized_error,
    strands_result_metrics,
)
from .model_factory import build_strands_model
from .models import AgentAssessment, SimulationRequest
from .product_needs import resolve_product_need_profile


PROBABILITY_SYSTEM_PROMPT = """You produce two separate estimates for a game-store exposure.

1. likelihood is the natural per-impression probability that the user purchases
the target item on this exposure. Purchase is normally rare. Weak evidence should
remain near the user's low observed reference purchase rate, not near 0.5.
2. relative_preference_score is a comparative ranking score for this user among
otherwise eligible candidate items. Here 0.5 means a typical candidate, values
above 0.5 mean stronger-than-typical preference, and values below 0.5 mean weaker
preference. It is not a natural purchase probability.

Do not force the two estimates to be numerically similar. A user can have a high
relative preference for an item while the absolute purchase probability remains
low. Use direct click/purchase evidence, category fit, affordability, recency,
surface intent, and contradictions for both estimates, but let the natural base
rate affect likelihood much more strongly than relative_preference_score.

Disentangle two nonexclusive motivations before combining evidence. Rational
need is item-centric and time-sensitive: current functional utility, progression
or campaign need, time since a similar function was satisfied, redundancy,
component ownership, and utility relative to price. Emotional need is
player-centric and relatively stable: aesthetics, identity, collection,
enjoyment, social expression, and persistent attribute preference. Product type
does not select a motivation. A single item or bundle may be rational,
emotional, or mixed. For a bundle, inspect component complementarity,
redundancy, price efficiency, and theme coherence.

Repeated views without click or purchase are weak negative evidence. A past
purchase of the exact item is positive only when repeat purchase is plausible
(currency, replenishable convenience, or renewable subscription). For
collectible, cosmetic, social, upgrade, or bundle-like one-time goods, an exact
past purchase suggests ownership or inconsistent data and must not be a strong
positive signal.

Set confidence from evidence quality, not from score extremity. Review retrieved
documents in rounds. Do not infer sensitive traits, claim calibration, or use a
synthetic label or model prediction. Ignore legacy add_to_cart or CART tokens in
imported memories; they are invalid telemetry for this game-store domain.
"""

ACTION_SYSTEM_PROMPT = """You estimate behavior before purchase probability.
For each reachable store state, compare only the explicitly allowed actions and
return a conditional action distribution that sums to 1 within that state.
The code, not you, applies deterministic state transitions and sums all paths
that end in purchase.

Separate product selection from action commitment. First assess current need,
desire, feasibility, urgency, uncertainty, and hesitation. Then compare four
latent intentions: BUY_NOW, EXPLORE, DEFER, and REJECT. DEFER is an internal
intention, not a store action; map it to SKIP or BACK according to the current
state. A user may strongly prefer an item while deferring the purchase.

SKIP means dismissing the current offer while remaining in the store surface.
EXIT means leaving the current store surface. Session fatigue is the strongest
available distinction between them. If BUY_NOW remains plausible after opening
the detail screen, carry that commitment through START_PURCHASE and
CONFIRM_PURCHASE instead of restarting the decision at every screen.

The supplied graph may contain environment_event transitions such as payment
outcomes. Use explicit feasibility facts for those transitions, but do not let
uncertain payment-failure frequency dominate the user-action decision.

Evaluate 2-4 plausible alternatives per state. Ask only:
- Which action is most consistent with observed past behavior and persona?
- What current game goal or constraint makes this decision timely?
- Did price/budget, fatigue, ownership, and repeat-purchase semantics matter?
- Is exploration, deferral, or exit more natural than purchase?
- Which next states are actually possible under the supplied transition table?

Do not invent screens, actions, transitions, memories, labels, or model scores.
Weak evidence must preserve meaningful SKIP/BACK/EXIT probability. A strong item
preference does not justify skipping CLICK unless direct purchase is plausible
on the current surface. Game items have no cart or add-to-cart action. Action
probabilities are conditional on arriving at the state, not unconditional
purchase probabilities.

Represent stochastic behavior rather than converting the distribution into a
single majority-class prediction. A rare action may retain non-zero probability
even when it did not occur in a short personal history. Use retrieved PathSim
behavior paths as item-specific evidence, not as a calibrated probability.

Before assigning actions, assess two nonexclusive motivation lanes. Rational
evidence is functional and time-sensitive: current progression/campaign need,
time since that function was last satisfied, utility, redundancy/ownership, and
price efficiency. Emotional evidence is player-centered and relatively stable:
aesthetics, identity, collection, enjoyment, and social preference across the
longer history. Do not route solely from product_type. For bundles, evaluate
component utility, owned overlap, price efficiency, and theme coherence
component by component, then combine the lanes. Do not count one observation
twice unless it has distinct functional and emotional implications.

relative_preference_score remains a comparative ranking score where 0.5 is a
typical eligible candidate. Confidence reflects evidence quality. Counterfactual
results are ephemeral and must never be described as observed history.

Retrieved memories contain both supporting and opposing cases when available.
Do not count the number of documents as evidence; compare how closely each
observed situation matches the current game state.
"""

CRITIC_SYSTEM_PROMPT = """You are an independent action-distribution
validator. You did not produce the proposed distribution. Inspect only:
persona consistency, recent observed behavior consistency, repeat-purchase
validity, price/budget consistency, and whether the proposal skips weaker
actions such as CLICK without evidence.

The persona, recent-behavior, and weaker-action checks are advisory critiques.
They must identify contradictions but must not revise probabilities because the
proposal is already hierarchically grounded in the same observed transitions.
Revise the two conditional distributions only for a hard repeat-purchase
violation or a clear price/budget violation, and at most once. Do not introduce
actions or transitions outside the supplied deterministic table. Do not infer a
label, oracle, model score, or hidden user trait. Retrieved counterfactual
statements are not observations. Keep contradictions and revision_summary
concise.

Observed transition statistics are higher-priority evidence than imagined
outcomes, but they have already been applied before this review. An exact-item
or similar-category stopping path may make recent_behavior_consistent false and
should be reported, but must not be applied to the distribution a second time.

Do not revise merely because purchase is uncommon in a short history. Preserve
stochastic uncertainty and avoid collapsing a plausible low-frequency action
to zero.

The application has already applied a selection-to-commitment gate using
observed game state. Treat its DEFER mapping and current-state constraints as
higher-priority than a generic preference narrative.
"""

LOGGER = logging.getLogger(__name__)


class StrandsAssessmentProvider:
    def __init__(
        self,
        model_id: str | None = None,
        region_name: str | None = None,
        agent_factory: Callable[[], Any] | None = None,
        include_failure_details: bool = False,
        trace_events: TraceEvents | None = None,
        mode: str = "probability",
        critic_agent_factory: Callable[[], Any] | None = None,
        action_graph: ActionGraph | None = None,
    ) -> None:
        self.model_id = model_id or os.environ["BEDROCK_MODEL_ID"]
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.agent_factory = agent_factory
        self.include_failure_details = include_failure_details
        self.trace_events = trace_events
        self.critic_agent_factory = critic_agent_factory
        self.action_graph = action_graph or DEFAULT_ACTION_GRAPH
        if mode not in {"probability", "actions"}:
            raise ValueError("mode must be 'probability' or 'actions'")
        self.mode = mode

    def assess(
        self,
        request: SimulationRequest,
        memory_evidence: Sequence[str],
    ) -> AgentAssessment:
        if self.mode == "actions":
            return self._assess_rollout(request, memory_evidence)

        from pydantic import BaseModel, ConfigDict, Field

        class AssessmentOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")

            sufficient_evidence: bool
            likelihood: float = Field(ge=0.0, le=1.0)
            relative_preference_score: float = Field(ge=0.0, le=1.0)
            confidence: float = Field(ge=0.0, le=1.0)
            reasons: list[str] = Field(default_factory=list, max_length=6)
            contradictions: list[str] = Field(default_factory=list, max_length=6)
            additional_memory_question: str = Field(default="", max_length=240)

        try:
            agent = (
                self.agent_factory()
                if self.agent_factory
                else self._build_agent()
            )
        except Exception as exc:
            reason = (
                "Agent structured assessment unavailable; neutral fallback used."
            )
            if self.include_failure_details:
                reason = f"{reason} {sanitized_error(exc)}"
            return AgentAssessment(
                likelihood=0.5,
                relative_preference_score=0.5,
                confidence=0.0,
                reasons=(reason,),
                contradictions=(),
            )
        documents = list(memory_evidence)[:9]
        previous: dict[str, object] | None = None
        output = None
        for round_index in range(3):
            start = round_index * 3
            new_documents = documents[start : start + 3]
            if round_index > 0 and not new_documents:
                break
            started = time.monotonic()
            try:
                result, attempts = invoke_with_transient_retries(
                    lambda: agent(
                        json.dumps(
                            self._round_payload(
                                request=request,
                                new_memory_evidence=new_documents,
                                previous_assessment=previous,
                                round_number=round_index + 1,
                                remaining_document_count=max(
                                    0, len(documents) - start - 3
                                ),
                            ),
                            ensure_ascii=False,
                        ),
                        structured_output_model=AssessmentOutput,
                    ),
                )
                output = result.structured_output
                if output is None:
                    raise ValueError("Strands returned no structured assessment")
                self._trace(
                    {
                        "stage": "assessment_round",
                        "round": round_index + 1,
                        "duration_seconds": round(time.monotonic() - started, 6),
                        "new_document_count": len(new_documents),
                        "remaining_document_count": max(
                            0, len(documents) - start - 3
                        ),
                        "sufficient_evidence": bool(output.sufficient_evidence),
                        "likelihood": float(output.likelihood),
                        "relative_preference_score": float(
                            getattr(output, "relative_preference_score", 0.5)
                        ),
                        "confidence": float(output.confidence),
                        "reasons": list(output.reasons),
                        "contradictions": list(output.contradictions),
                        "additional_memory_question": (
                            output.additional_memory_question
                        ),
                        "metrics": strands_result_metrics(result),
                        "fallback": False,
                        "attempts": attempts,
                    }
                )
            except Exception as exc:
                LOGGER.warning(
                    "Strands assessment round %s failed; using prior or neutral result: %s",
                    round_index + 1,
                    exc,
                )
                self._trace(
                    {
                        "stage": "assessment_round",
                        "round": round_index + 1,
                        "duration_seconds": round(time.monotonic() - started, 6),
                        "new_document_count": len(new_documents),
                        "error": sanitized_error(exc),
                        "fallback": True,
                        "fallback_source": (
                            "previous_assessment"
                            if previous is not None
                            else "neutral"
                        ),
                    }
                )
                if previous is not None:
                    return AgentAssessment(
                        likelihood=float(previous["likelihood"]),
                        relative_preference_score=float(
                            previous["relative_preference_score"]
                        ),
                        confidence=float(previous["confidence"]),
                        reasons=tuple(str(value) for value in previous["reasons"]),
                        contradictions=tuple(
                            str(value) for value in previous["contradictions"]
                        ),
                    )
                reason = (
                    "Agent structured assessment unavailable; neutral fallback used."
                )
                if self.include_failure_details:
                    reason = f"{reason} {sanitized_error(exc)}"
                return AgentAssessment(
                    likelihood=0.5,
                    relative_preference_score=0.5,
                    confidence=0.0,
                    reasons=(reason,),
                    contradictions=(),
                )
            previous = {
                "likelihood": output.likelihood,
                "relative_preference_score": getattr(
                    output,
                    "relative_preference_score",
                    0.5,
                ),
                "confidence": output.confidence,
                "reasons": list(output.reasons),
                "contradictions": list(output.contradictions),
                "additional_memory_question": output.additional_memory_question,
            }
            if output.sufficient_evidence or start + 3 >= len(documents):
                break

        if output is None:
            raise ValueError("Strands performed no assessment round")
        return AgentAssessment(
            likelihood=output.likelihood,
            relative_preference_score=float(
                getattr(output, "relative_preference_score", 0.5)
            ),
            confidence=output.confidence,
            reasons=tuple(output.reasons),
            contradictions=tuple(output.contradictions),
        )

    def _assess_rollout(
        self,
        request: SimulationRequest,
        memory_evidence: Sequence[str],
    ) -> AgentAssessment:
        from pydantic import BaseModel, ConfigDict, Field, create_model

        state_fields: dict[str, tuple[type[BaseModel], object]] = {}
        for state, action_fields in self.action_graph.output_fields().items():
            model = create_model(
                f"{state.title().replace('_', '')}Actions",
                __config__=ConfigDict(extra="forbid"),
                **{
                    field_name: (float, Field(ge=0.0, le=1.0))
                    for field_name in action_fields.values()
                },
            )
            state_fields[self.action_graph.output_field(state)] = (model, ...)

        class Counterfactual(BaseModel):
            model_config = ConfigDict(extra="forbid")

            state: str = Field(max_length=40)
            action: str = Field(max_length=40)
            expected_outcome: str = Field(max_length=180)

        class DecisionStateOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")

            need_strength: float = Field(default=0.5, ge=0.0, le=1.0)
            selection_strength: float = Field(default=0.5, ge=0.0, le=1.0)
            feasibility: float = Field(default=0.5, ge=0.0, le=1.0)
            urgency: float = Field(default=0.5, ge=0.0, le=1.0)
            uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
            hesitation: float = Field(default=0.5, ge=0.0, le=1.0)

        class IntentionOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")

            buy_now: float = Field(default=0.25, ge=0.0, le=1.0)
            explore: float = Field(default=0.25, ge=0.0, le=1.0)
            defer: float = Field(default=0.25, ge=0.0, le=1.0)
            reject: float = Field(default=0.25, ge=0.0, le=1.0)

        RolloutOutput = create_model(
            "RolloutOutput",
            __config__=ConfigDict(extra="forbid"),
            sufficient_evidence=(bool, ...),
            relative_preference_score=(float, Field(ge=0.0, le=1.0)),
            confidence=(float, Field(ge=0.0, le=1.0)),
            decision_state=(
                DecisionStateOutput,
                Field(default_factory=DecisionStateOutput),
            ),
            intentions=(
                IntentionOutput,
                Field(default_factory=IntentionOutput),
            ),
            counterfactuals=(
                list[Counterfactual],
                Field(min_length=3, max_length=12),
            ),
            reasons=(list[str], Field(default_factory=list, max_length=6)),
            contradictions=(
                list[str],
                Field(default_factory=list, max_length=6),
            ),
            additional_memory_question=(str, Field(default="", max_length=240)),
            **state_fields,
        )

        ValidatorOutput = create_model(
            "ValidatorOutput",
            __config__=ConfigDict(extra="forbid"),
            persona_consistent=(bool, ...),
            recent_behavior_consistent=(bool, ...),
            repeat_purchase_valid=(bool, ...),
            price_budget_consistent=(bool, ...),
            weak_action_sequence_valid=(bool, ...),
            confidence=(float, Field(ge=0.0, le=1.0)),
            contradictions=(list[str], Field(default_factory=list, max_length=5)),
            revision_summary=(str, Field(default="", max_length=300)),
            **state_fields,
        )

        try:
            agent = (
                self.agent_factory()
                if self.agent_factory
                else self._build_agent()
            )
        except Exception as exc:
            return self._neutral_rollout_assessment(
                request,
                failure=sanitized_error(exc),
            )
        documents = list(memory_evidence)[:9]
        reviewed_documents: list[str] = []
        baseline_decision_state = build_decision_state(request)
        previous: dict[str, object] | None = None
        output = None
        for round_index in range(3):
            start = round_index * 3
            new_documents = documents[start : start + 3]
            if round_index > 0 and not new_documents:
                break
            reviewed_documents.extend(new_documents)
            started = time.monotonic()
            try:
                result, attempts = invoke_with_transient_retries(
                    lambda: agent(
                        json.dumps(
                            self._rollout_round_payload(
                                request=request,
                                new_memory_evidence=new_documents,
                                previous_assessment=previous,
                                round_number=round_index + 1,
                                remaining_document_count=max(
                                    0, len(documents) - start - 3
                                ),
                            ),
                            ensure_ascii=False,
                        ),
                        structured_output_model=RolloutOutput,
                    ),
                )
                output = result.structured_output
                if output is None:
                    raise ValueError("Strands returned no rollout assessment")
                distributions = self._action_distributions(output)
                decision_state, intentions = merge_model_decision(
                    baseline_decision_state,
                    self._model_decision_state(output),
                    self._model_intentions(output),
                    model_confidence=float(output.confidence),
                )
                rollout = rollout_purchase_probability(
                    distributions,
                    surface=request.context.surface,
                    graph=self.action_graph,
                )
                self._trace(
                    {
                        "stage": "action_assessment_round",
                        "round": round_index + 1,
                        "duration_seconds": round(
                            time.monotonic() - started, 6
                        ),
                        "new_document_count": len(new_documents),
                        "remaining_document_count": max(
                            0, len(documents) - start - 3
                        ),
                        "sufficient_evidence": bool(output.sufficient_evidence),
                        "relative_preference_score": float(
                            output.relative_preference_score
                        ),
                        "confidence": float(output.confidence),
                        "decision_state": decision_state.to_dict(),
                        "intention_distribution": intentions,
                        "action_distributions": distributions,
                        "rollout_probability": rollout.purchase_probability,
                        "candidate_actions": [
                            value.model_dump()
                            for value in output.counterfactuals
                        ],
                        "reasons": list(output.reasons),
                        "contradictions": list(output.contradictions),
                        "additional_memory_question": (
                            output.additional_memory_question
                        ),
                        "metrics": strands_result_metrics(result),
                        "fallback": False,
                        "attempts": attempts,
                    }
                )
            except Exception as exc:
                LOGGER.warning(
                    "Strands rollout round %s failed; using prior or neutral result: %s",
                    round_index + 1,
                    exc,
                )
                self._trace(
                    {
                        "stage": "action_assessment_round",
                        "round": round_index + 1,
                        "duration_seconds": round(
                            time.monotonic() - started, 6
                        ),
                        "error": sanitized_error(exc),
                        "fallback": True,
                        "fallback_source": (
                            "previous_assessment"
                            if previous is not None
                            else "neutral"
                        ),
                    }
                )
                if previous is None:
                    return self._neutral_rollout_assessment(
                        request,
                        failure=sanitized_error(exc),
                    )
                return self._rollout_assessment_from_values(
                    request=request,
                    distributions=previous["action_distributions"],
                    relative_preference_score=float(
                        previous["relative_preference_score"]
                    ),
                    confidence=float(previous["confidence"]),
                    reasons=previous["reasons"],
                    contradictions=previous["contradictions"],
                    validator_adjusted=False,
                    decision_state=previous.get("decision_state", {}),
                    intention_distribution=previous.get(
                        "intention_distribution",
                        {},
                    ),
                )

            previous = {
                "action_distributions": distributions,
                "relative_preference_score": output.relative_preference_score,
                "confidence": output.confidence,
                "decision_state": decision_state.to_dict(),
                "decision_state_object": decision_state,
                "intention_distribution": intentions,
                "reasons": list(output.reasons),
                "contradictions": list(output.contradictions),
                "additional_memory_question": output.additional_memory_question,
            }
            if output.sufficient_evidence or start + 3 >= len(documents):
                break

        if output is None or previous is None:
            return self._neutral_rollout_assessment(
                request,
                failure="Strands performed no rollout assessment round",
            )

        raw_actor_proposed = previous["action_distributions"]
        graph_prior_weight = self.action_graph.base_distribution_weight
        actor_proposed = blend_action_distributions(
            raw_actor_proposed,
            self.action_graph.base_action_distributions(),
            {
                state: graph_prior_weight
                for state in self.action_graph.states
            },
            max_empirical_weight=graph_prior_weight,
            graph=self.action_graph,
        )
        empirical, empirical_strengths, transition_count = (
            empirical_transition_policy(
                reviewed_documents,
                item=request.item,
                now=request.context.timestamp,
                context=request.context,
                prior_policy=actor_proposed,
            )
        )
        proposed = normalize_action_distributions(
            {
                state: empirical.get(state, distribution)
                for state, distribution in actor_proposed.items()
            },
            graph=self.action_graph,
        )
        grounding_distance = self._distribution_distance(
            actor_proposed,
            proposed,
        )
        grounding_strength = max(empirical_strengths.values(), default=0.0)
        grounded_rollout = rollout_purchase_probability(
            proposed,
            surface=request.context.surface,
            graph=self.action_graph,
        )
        self._trace(
            {
                "stage": "transition_grounding",
                "observed_transition_count": transition_count,
                "actor_action_distributions": raw_actor_proposed,
                "graph_base_distributions": (
                    self.action_graph.base_action_distributions()
                ),
                "graph_base_weight": graph_prior_weight,
                "empirical_action_distributions": empirical,
                "state_evidence_strength": empirical_strengths,
                "action_distributions": proposed,
                "rollout_probability": grounded_rollout.purchase_probability,
                "adjusted": grounding_distance > 0.01,
            }
        )
        decision_state = previous["decision_state_object"]
        if not hasattr(decision_state, "commitment_strength"):
            decision_state = baseline_decision_state
        grounded_distributions = proposed
        supports_default_commitment = (
            self.action_graph.to_dict() == DEFAULT_ACTION_GRAPH.to_dict()
        )
        commitment = (
            apply_commitment_gate(
                grounded_distributions,
                decision_state,
                previous["intention_distribution"],
                graph=self.action_graph,
            )
            if supports_default_commitment
            else SimpleNamespace(
                distributions=grounded_distributions,
                intentions=dict(previous["intention_distribution"]),
                total_variation={},
                adjusted=False,
            )
        )
        proposed = normalize_action_distributions(
            commitment.distributions,
            graph=self.action_graph,
        )
        committed_rollout = rollout_purchase_probability(
            proposed,
            surface=request.context.surface,
            graph=self.action_graph,
        )
        self._trace(
            {
                "stage": "commitment_gate",
                "decision_state": decision_state.to_dict(),
                "intention_distribution": dict(commitment.intentions),
                "defer_probability": commitment.intentions["DEFER"],
                "total_variation": dict(commitment.total_variation),
                "action_distributions": proposed,
                "rollout_probability": (
                    committed_rollout.purchase_probability
                ),
                "adjusted": commitment.adjusted,
            }
        )
        counterfactual = (
            evaluate_counterfactual_consistency(
                request,
                grounded_distributions,
                proposed,
                decision_state,
                graph=self.action_graph,
            )
            if supports_default_commitment
            else SimpleNamespace(
                checks={},
                purchase_probabilities={},
                contradictions=(),
            )
        )
        started = time.monotonic()
        try:
            critic = (
                self.critic_agent_factory()
                if self.critic_agent_factory
                else self._build_critic_agent()
            )
            result, attempts = invoke_with_transient_retries(
                lambda: critic(
                    json.dumps(
                        {
                            "evidence": self._rollout_round_payload(
                                request=request,
                                new_memory_evidence=reviewed_documents,
                                previous_assessment=None,
                                round_number=1,
                                remaining_document_count=0,
                            ),
                            "proposed_action_distributions": proposed,
                            "actor_action_distributions": actor_proposed,
                            "decision_state": decision_state.to_dict(),
                            "intention_distribution": dict(
                                commitment.intentions
                            ),
                            "observed_transition_policy": empirical,
                            "observed_transition_evidence_strength": (
                                empirical_strengths
                            ),
                            "proposed_confidence": previous["confidence"],
                            "proposed_contradictions": previous[
                                "contradictions"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    structured_output_model=ValidatorOutput,
                ),
            )
            checks = {
                "persona_consistent": bool(
                    result.structured_output.persona_consistent
                ),
                "recent_behavior_consistent": bool(
                    result.structured_output.recent_behavior_consistent
                ),
                "repeat_purchase_valid": bool(
                    result.structured_output.repeat_purchase_valid
                ),
                "price_budget_consistent": bool(
                    result.structured_output.price_budget_consistent
                ),
                "weak_action_sequence_valid": bool(
                    result.structured_output.weak_action_sequence_valid
                ),
            }
            raw_validated = self._action_distributions(
                result.structured_output
            )
            hard_revision_eligible = (
                not checks["repeat_purchase_valid"]
                or not checks["price_budget_consistent"]
            )
            validated = (
                proposed
                if not hard_revision_eligible
                else limit_action_distribution_revision(
                    proposed,
                    raw_validated,
                    max_total_variation=0.10,
                    graph=self.action_graph,
                )
            )
            critic_adjusted = (
                self._distribution_distance(proposed, validated) > 0.01
            )
            contradictions = list(previous["contradictions"])
            contradictions.extend(result.structured_output.contradictions)
            reasons = list(previous["reasons"])
            if result.structured_output.revision_summary:
                reasons.append(result.structured_output.revision_summary)
            rollout = rollout_purchase_probability(
                validated,
                surface=request.context.surface,
                graph=self.action_graph,
            )
            self._trace(
                {
                    "stage": "action_validator",
                    "duration_seconds": round(
                        time.monotonic() - started, 6
                    ),
                    "checks": checks,
                    "hard_revision_eligible": hard_revision_eligible,
                    "advisory_checks": {
                        name: checks[name]
                        for name in (
                            "persona_consistent",
                            "recent_behavior_consistent",
                            "weak_action_sequence_valid",
                        )
                    },
                    "critic_raw_action_distributions": raw_validated,
                    "action_distributions": validated,
                    "rollout_probability": rollout.purchase_probability,
                    "adjusted": critic_adjusted,
                    "critic_adjusted": critic_adjusted,
                    "revision_limited": (
                        self._distribution_distance(
                            raw_validated,
                            validated,
                        )
                        > 0.01
                    ),
                    "transition_grounding_adjusted": (
                        grounding_distance > 0.01
                    ),
                    "contradictions": list(
                        result.structured_output.contradictions
                    ),
                    "revision_summary": (
                        result.structured_output.revision_summary
                    ),
                    "metrics": strands_result_metrics(result),
                    "fallback": False,
                    "attempts": attempts,
                }
            )
            counterfactual_adjusted = False
            if counterfactual.contradictions:
                contradictions.extend(counterfactual.contradictions)
            self._trace(
                {
                    "stage": "counterfactual_validator",
                    "checks": dict(counterfactual.checks),
                    "purchase_probabilities": dict(
                        counterfactual.purchase_probabilities
                    ),
                    "contradictions": list(
                        counterfactual.contradictions
                    ),
                    "action_distributions": validated,
                    "rollout_probability": rollout.purchase_probability,
                    "adjusted": counterfactual_adjusted,
                }
            )
            return self._rollout_assessment_from_values(
                request=request,
                distributions=validated,
                relative_preference_score=float(
                    previous["relative_preference_score"]
                ),
                confidence=float(result.structured_output.confidence),
                reasons=reasons,
                contradictions=contradictions,
                validator_adjusted=(
                    critic_adjusted or counterfactual_adjusted
                ),
                commitment_adjusted=commitment.adjusted,
                counterfactual_adjusted=counterfactual_adjusted,
                decision_state=decision_state.to_dict(),
                intention_distribution=commitment.intentions,
                counterfactual_checks=counterfactual.checks,
                transition_grounding_strength=grounding_strength,
            )
        except Exception as exc:
            self._trace(
                {
                    "stage": "action_validator",
                    "duration_seconds": round(
                        time.monotonic() - started, 6
                    ),
                    "error": sanitized_error(exc),
                    "fallback": True,
                    "fallback_source": "proposed_action_distributions",
                }
            )
            self._trace(
                {
                    "stage": "counterfactual_validator",
                    "checks": dict(counterfactual.checks),
                    "purchase_probabilities": dict(
                        counterfactual.purchase_probabilities
                    ),
                    "contradictions": list(
                        counterfactual.contradictions
                    ),
                    "action_distributions": proposed,
                    "rollout_probability": (
                        rollout_purchase_probability(
                            proposed,
                            surface=request.context.surface,
                            graph=self.action_graph,
                        ).purchase_probability
                    ),
                    "adjusted": False,
                    "fallback": True,
                }
            )
            return self._rollout_assessment_from_values(
                request=request,
                distributions=proposed,
                relative_preference_score=float(
                    previous["relative_preference_score"]
                ),
                confidence=float(previous["confidence"]),
                reasons=previous["reasons"],
                contradictions=previous["contradictions"],
                validator_adjusted=False,
                commitment_adjusted=commitment.adjusted,
                decision_state=decision_state.to_dict(),
                intention_distribution=commitment.intentions,
                counterfactual_checks=counterfactual.checks,
                transition_grounding_strength=grounding_strength,
            )

    def _trace(self, event: dict[str, Any]) -> None:
        if self.trace_events is not None:
            self.trace_events.append(event)

    def _build_agent(self) -> Any:
        from strands import Agent

        model = build_strands_model(
            model_id=self.model_id,
            region_name=self.region_name,
            max_tokens=int(
                os.getenv(
                    (
                        "PURCHASE_BEHAVIOR_ACTION_MAX_TOKENS"
                        if self.mode == "actions"
                        else "PURCHASE_BEHAVIOR_PROBABILITY_MAX_TOKENS"
                    ),
                    "1800" if self.mode == "actions" else "1100",
                )
            ),
        )
        # A fresh Agent is deliberate: no conversation state crosses requests/users.
        return Agent(
            model=model,
            system_prompt=(
                ACTION_SYSTEM_PROMPT
                if self.mode == "actions"
                else PROBABILITY_SYSTEM_PROMPT
            ),
            tools=[],
            callback_handler=None,
        )

    def _build_critic_agent(self) -> Any:
        from strands import Agent

        return Agent(
            model=build_strands_model(
                model_id=self.model_id,
                region_name=self.region_name,
                max_tokens=int(
                    os.getenv(
                        "PURCHASE_BEHAVIOR_CRITIC_MAX_TOKENS",
                        "1800",
                    )
                ),
            ),
            system_prompt=CRITIC_SYSTEM_PROMPT,
            tools=[],
            callback_handler=None,
        )

    @staticmethod
    def _payload(
        request: SimulationRequest,
        memory_evidence: Sequence[str],
    ) -> dict[str, object]:
        return StrandsAssessmentProvider._round_payload(
            request=request,
            new_memory_evidence=memory_evidence,
            previous_assessment=None,
            round_number=1,
            remaining_document_count=0,
        )

    @staticmethod
    def _round_payload(
        request: SimulationRequest,
        new_memory_evidence: Sequence[str],
        previous_assessment: dict[str, object] | None,
        round_number: int,
        remaining_document_count: int,
    ) -> dict[str, object]:
        need_profile = resolve_product_need_profile(request.item)
        return {
            "round": {
                "number": round_number,
                "max_rounds": 3,
                "remaining_document_count": remaining_document_count,
                "previous_assessment": previous_assessment,
            },
            "persona": {
                "summary": request.user.persona_summary,
                "pickiness": request.user.pickiness,
                "price_sensitivity": request.user.price_sensitivity,
                "category_preferences": dict(request.user.category_preferences),
            },
            "item": {
                "item_id": request.item.item_id,
                "product_type": request.item.product_type,
                "categories": request.item.categories,
                "price": request.item.price,
                "discount_rate": request.item.discount_rate,
                "components": [
                    component.to_dict()
                    for component in request.item.components
                ],
                "attributes": dict(request.item.attributes),
                "need_profile": need_profile.to_dict(),
                "need_assessment_contract": {
                    "rational": (
                        "time-sensitive functional need; check recent "
                        "satisfaction, utility, redundancy, and price efficiency"
                    ),
                    "emotional": (
                        "stable player-centered preference; check aesthetics, "
                        "identity, collection, enjoyment, and social expression"
                    ),
                    "bundle_rule": (
                        "evaluate components and theme; product_type alone does "
                        "not determine the dominant need"
                    ),
                },
                "purchase_semantics": {
                    "repeatable_likely": bool(
                        set(request.item.categories).intersection(
                            {"currency", "subscription", "convenience"}
                        )
                    ),
                    "already_owned": request.item.already_owned,
                },
            },
            "context": {
                "surface": request.context.surface,
                "session_fatigue": request.context.session_fatigue,
                "budget_reference": request.context.budget_reference,
            },
            "game_state": request.game_state.to_dict(),
            "deterministic_decision_state": build_decision_state(
                request
            ).to_dict(),
            "recent_interactions": [
                {
                    "event_type": event.event_type,
                    "item_id": event.item_id,
                    "categories": event.categories,
                    "timestamp": event.timestamp.isoformat(),
                    "rating": event.rating,
                }
                for event in request.interactions[-30:]
            ],
            "new_retrieved_memories": list(new_memory_evidence)[:3],
        }

    def _rollout_round_payload(
        self,
        request: SimulationRequest,
        new_memory_evidence: Sequence[str],
        previous_assessment: dict[str, object] | None,
        round_number: int,
        remaining_document_count: int,
    ) -> dict[str, object]:
        serializable_previous = (
            {
                key: value
                for key, value in previous_assessment.items()
                if key != "decision_state_object"
            }
            if previous_assessment is not None
            else None
        )
        payload = self._round_payload(
            request=request,
            new_memory_evidence=new_memory_evidence,
            previous_assessment=serializable_previous,
            round_number=round_number,
            remaining_document_count=remaining_document_count,
        )
        payload["deterministic_environment"] = {
            "graph_id": self.action_graph.graph_id,
            "graph_version": self.action_graph.version,
            "initial_state": self.action_graph.initial_state(
                request.context.surface
            ),
            "transitions": transition_table_payload(self.action_graph),
            "rollout_depth": self.action_graph.max_depth,
            "timing": (
                "Transition timing is optional and evaluated separately from "
                "action probability. Do not invent durations."
            ),
            "instruction": (
                "Return probabilities conditional on arriving at each state. "
                "The application computes purchase path probability. Return "
                "selection/commitment factors and BUY_NOW, EXPLORE, DEFER, "
                "REJECT latent intentions before the store actions."
            ),
        }
        return payload

    @staticmethod
    def _model_decision_state(output: Any) -> dict[str, float]:
        value = getattr(output, "decision_state", None)
        if value is None:
            return {}
        return {
            name: float(getattr(value, name))
            for name in (
                "need_strength",
                "selection_strength",
                "feasibility",
                "urgency",
                "uncertainty",
                "hesitation",
            )
            if getattr(value, name, None) is not None
        }

    @staticmethod
    def _model_intentions(output: Any) -> dict[str, float]:
        value = getattr(output, "intentions", None)
        if value is None:
            return {}
        return {
            "BUY_NOW": float(getattr(value, "buy_now", 0.0)),
            "EXPLORE": float(getattr(value, "explore", 0.0)),
            "DEFER": float(getattr(value, "defer", 0.0)),
            "REJECT": float(getattr(value, "reject", 0.0)),
        }

    def _action_distributions(
        self,
        output: Any,
    ) -> dict[str, dict[str, float]]:
        if output is None:
            raise ValueError("Strands returned no structured rollout output")
        fields = self.action_graph.output_fields()
        distributions: dict[str, dict[str, float]] = {}
        for state, action_fields in fields.items():
            state_output = getattr(
                output,
                self.action_graph.output_field(state),
                None,
            )
            if state_output is None:
                continue
            action_values: dict[str, float] = {}
            for action, field_name in action_fields.items():
                value = getattr(state_output, field_name, None)
                if (
                    value is None
                    and state == "ITEM_DETAIL"
                    and action == "START_PURCHASE"
                ):
                    value = getattr(state_output, "purchase", None)
                if value is not None:
                    action_values[action] = float(value)
            if action_values:
                distributions[state] = action_values
        return normalize_action_distributions(
            distributions,
            graph=self.action_graph,
        )

    @staticmethod
    def _distribution_distance(
        left: object,
        right: object,
    ) -> float:
        left_values = left if isinstance(left, dict) else {}
        right_values = right if isinstance(right, dict) else {}
        return sum(
            abs(
                float(left_values.get(state, {}).get(action, 0.0))
                - float(right_values.get(state, {}).get(action, 0.0))
            )
            for state in set(left_values).union(right_values)
            for action in set(left_values.get(state, {})).union(
                right_values.get(state, {})
            )
        )

    def _neutral_rollout_assessment(
        self,
        request: SimulationRequest,
        *,
        failure: str,
    ) -> AgentAssessment:
        decision_state = build_decision_state(request)
        intentions = intentions_from_state(decision_state)
        neutral_distributions = self.action_graph.ineligible_distributions()
        commitment = (
            apply_commitment_gate(
                neutral_distributions,
                decision_state,
                intentions,
                graph=self.action_graph,
            )
            if self.action_graph.to_dict() == DEFAULT_ACTION_GRAPH.to_dict()
            else SimpleNamespace(
                distributions=neutral_distributions,
                adjusted=False,
            )
        )
        return self._rollout_assessment_from_values(
            request=request,
            distributions=commitment.distributions,
            relative_preference_score=0.5,
            confidence=0.0,
            reasons=(
                f"Action rollout unavailable; neutral fallback used. {failure}"
                if self.include_failure_details
                else "Action rollout unavailable; neutral fallback used."
            ,),
            contradictions=(),
            validator_adjusted=False,
            commitment_adjusted=commitment.adjusted,
            decision_state=decision_state.to_dict(),
            intention_distribution=intentions,
            transition_grounding_strength=0.0,
        )

    def _rollout_assessment_from_values(
        self,
        *,
        request: SimulationRequest,
        distributions: object,
        relative_preference_score: float,
        confidence: float,
        reasons: Sequence[object],
        contradictions: Sequence[object],
        validator_adjusted: bool,
        commitment_adjusted: bool = False,
        counterfactual_adjusted: bool = False,
        decision_state: Mapping[str, float] | None = None,
        intention_distribution: Mapping[str, float] | None = None,
        counterfactual_checks: Mapping[str, bool] | None = None,
        transition_grounding_strength: float = 0.0,
    ) -> AgentAssessment:
        normalized = normalize_action_distributions(
            distributions if isinstance(distributions, dict) else {},
            graph=self.action_graph,
        )
        rollout = rollout_purchase_probability(
            normalized,
            surface=request.context.surface,
            graph=self.action_graph,
        )
        return AgentAssessment(
            likelihood=rollout.purchase_probability,
            relative_preference_score=relative_preference_score,
            rollout_probability=rollout.purchase_probability,
            action_distributions=normalized,
            decision_state=dict(decision_state or {}),
            intention_distribution=dict(intention_distribution or {}),
            counterfactual_checks=dict(counterfactual_checks or {}),
            validator_adjusted=validator_adjusted,
            commitment_adjusted=commitment_adjusted,
            counterfactual_adjusted=counterfactual_adjusted,
            transition_grounding_strength=transition_grounding_strength,
            confidence=confidence,
            reasons=tuple(str(value) for value in reasons),
            contradictions=tuple(str(value) for value in contradictions),
        )
