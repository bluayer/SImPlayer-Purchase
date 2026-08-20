from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, Sequence

from .evaluation_trace import TraceEvents
from .models import AgentAssessment, SimulationRequest


class AssessmentProvider(Protocol):
    def assess(
        self,
        request: SimulationRequest,
        memory_evidence: Sequence[str],
    ) -> AgentAssessment: ...


class HybridAssessmentProvider:
    """Keep the established probability estimate and add an independent rollout."""

    def __init__(
        self,
        *,
        probability_provider: AssessmentProvider,
        rollout_provider: AssessmentProvider,
        trace_events: TraceEvents | None = None,
    ) -> None:
        self.probability_provider = probability_provider
        self.rollout_provider = rollout_provider
        self.trace_events = trace_events

    def assess(
        self,
        request: SimulationRequest,
        memory_evidence: Sequence[str],
    ) -> AgentAssessment:
        with ThreadPoolExecutor(max_workers=2) as executor:
            probability_future = executor.submit(
                self.probability_provider.assess,
                request,
                memory_evidence,
            )
            rollout_future = executor.submit(
                self.rollout_provider.assess,
                request,
                memory_evidence,
            )
            probability = probability_future.result()
            rollout = rollout_future.result()

        combined = AgentAssessment(
            likelihood=probability.likelihood,
            relative_preference_score=probability.relative_preference_score,
            rollout_probability=rollout.rollout_probability,
            action_distributions=rollout.action_distributions,
            decision_state=rollout.decision_state,
            intention_distribution=rollout.intention_distribution,
            counterfactual_checks=rollout.counterfactual_checks,
            validator_adjusted=rollout.validator_adjusted,
            commitment_adjusted=rollout.commitment_adjusted,
            counterfactual_adjusted=rollout.counterfactual_adjusted,
            transition_grounding_strength=(
                rollout.transition_grounding_strength
            ),
            confidence=probability.confidence,
            rollout_confidence=rollout.confidence,
            reasons=probability.reasons,
            contradictions=tuple(
                dict.fromkeys(
                    (
                        *probability.contradictions,
                        *rollout.contradictions,
                    )
                )
            ),
        )
        if self.trace_events is not None:
            self.trace_events.append(
                {
                    "stage": "hybrid_assessment",
                    "probability_likelihood": probability.likelihood,
                    "probability_confidence": probability.confidence,
                    "rollout_probability": rollout.rollout_probability,
                    "rollout_confidence": rollout.confidence,
                    "validator_adjusted": rollout.validator_adjusted,
                    "commitment_adjusted": rollout.commitment_adjusted,
                    "counterfactual_adjusted": (
                        rollout.counterfactual_adjusted
                    ),
                    "decision_state": dict(rollout.decision_state),
                    "intention_distribution": dict(
                        rollout.intention_distribution
                    ),
                }
            )
        return combined
