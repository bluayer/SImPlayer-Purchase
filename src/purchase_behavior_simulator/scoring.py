from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from .models import (
    AgentAssessment,
    BehaviorEvent,
    Item,
    KnowledgeGraphEvidence,
    SimulationResult,
    ExposureContext,
    UserProfile,
)


EPSILON = 1e-6


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def logit(probability: float) -> float:
    bounded = clamp(probability, EPSILON, 1.0 - EPSILON)
    return math.log(bounded / (1.0 - bounded))


@dataclass(frozen=True)
class PlattCalibration:
    slope: float = 1.0
    intercept: float = 0.0
    version: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.version is not None

    def apply(self, probability: float) -> float:
        return sigmoid(self.slope * logit(probability) + self.intercept)


@dataclass(frozen=True)
class ScoringConfig:
    event_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "impression": 0.0,
            "view": 0.08,
            "click": 0.20,
            "try_on": 0.35,
            "equip": 0.55,
            "use": 0.65,
            "purchase": 1.0,
            "dismiss": -0.25,
            "refund": -1.0,
        }
    )
    recency_half_life_days: float = 30.0
    item_match_multiplier: float = 1.0
    category_match_multiplier: float = 0.55
    kg_item_weight: float = 0.5
    episodic_logit_weight: float = 0.60
    kg_logit_weight: float = 0.35
    context_logit_weight: float = 0.45
    agent_logit_weight: float = 0.50
    agent_logit_reference: str = "neutral"
    rollout_logit_weight: float = 0.0
    rollout_logit_reference: str = "base"
    rollout_output_blend: float = 0.0
    fusion_logit_shrink: float = 1.0
    fusion_prior_anchor: float = 0.10
    fallback_prior_rate: float = 0.02
    fallback_prior_strength: float = 50.0
    model_version: str = "purchase-behavior-simulator-prototype"


def smoothed_purchase_rate(
    interactions: Sequence[BehaviorEvent],
    prior_rate: float,
    prior_strength: float,
) -> float:
    impressions = sum(event.event_type == "impression" for event in interactions)
    purchases = sum(event.event_type == "purchase" for event in interactions)
    denominator = impressions + prior_strength
    if denominator <= 0:
        return clamp(prior_rate, EPSILON, 1.0 - EPSILON)
    return clamp(
        (purchases + prior_rate * prior_strength) / denominator,
        EPSILON,
        1.0 - EPSILON,
    )


def combine_fused_and_rollout_probability(
    fused_probability: float,
    rollout_probability: float | None,
    *,
    rollout_output_blend: float,
    fusion_logit_shrink: float,
    fusion_prior_anchor: float,
) -> tuple[float, float]:
    blend = clamp(rollout_output_blend)
    fused = clamp(fused_probability, EPSILON, 1.0 - EPSILON)
    rollout = (
        clamp(rollout_probability, EPSILON, 1.0 - EPSILON)
        if rollout_probability is not None
        else None
    )
    raw_probability = (
        (1.0 - blend) * fused + blend * rollout
        if rollout is not None and blend > 0.0
        else fused
    )
    shrink = clamp(fusion_logit_shrink)
    anchor = clamp(
        fusion_prior_anchor,
        EPSILON,
        1.0 - EPSILON,
    )
    shrunk_probability = sigmoid(
        logit(anchor)
        + shrink * (logit(raw_probability) - logit(anchor))
    )
    return raw_probability, shrunk_probability


def episodic_affinity(
    interactions: Sequence[BehaviorEvent],
    item: Item,
    now: datetime,
    config: ScoringConfig,
) -> tuple[float, float]:
    weighted_sum = 0.0
    absolute_sum = 0.0
    item_categories = set(item.categories)

    for event in interactions:
        behavior_weight = config.event_weights.get(event.event_type, 0.0)
        if behavior_weight == 0.0:
            continue

        if event.item_id == item.item_id:
            relevance = config.item_match_multiplier
        elif item_categories.intersection(event.categories):
            overlap = len(item_categories.intersection(event.categories)) / max(
                1, len(item_categories.union(event.categories))
            )
            relevance = config.category_match_multiplier * overlap
        else:
            continue

        age_days = max(0.0, (now - event.timestamp).total_seconds() / 86400.0)
        decay = math.exp(
            -math.log(2.0) * age_days / max(config.recency_half_life_days, EPSILON)
        )
        signal = behavior_weight
        magnitude = abs(behavior_weight)
        if event.rating is not None:
            rating_signal = clamp((event.rating - 3.0) / 2.0, -1.0, 1.0)
            signal = 0.75 * behavior_weight + 0.25 * rating_signal
            magnitude = 0.75 * abs(behavior_weight) + 0.25 * abs(rating_signal)

        evidence = signal * relevance * decay
        weighted_sum += evidence
        absolute_sum += magnitude * relevance * decay

    if absolute_sum == 0.0:
        return 0.5, 0.0

    normalized = clamp(weighted_sum / absolute_sum, -1.0, 1.0)
    probability_like_score = 0.5 + 0.45 * normalized
    confidence = 1.0 - math.exp(-absolute_sum)
    return clamp(probability_like_score, 0.05, 0.95), clamp(confidence)


def pathsim(shared_paths: float, source_self_paths: float, target_self_paths: float) -> float:
    denominator = source_self_paths + target_self_paths
    if denominator <= 0.0:
        return 0.5
    return clamp(2.0 * shared_paths / denominator)


def knowledge_graph_affinity(
    evidence: KnowledgeGraphEvidence,
    item_weight: float,
) -> tuple[float, float]:
    if evidence.precomputed_affinity is not None:
        return clamp(evidence.precomputed_affinity), clamp(
            evidence.precomputed_confidence
        )

    has_item_evidence = evidence.item_source_self_paths + evidence.item_target_self_paths > 0
    has_user_evidence = evidence.user_self_paths + evidence.user_target_self_paths > 0
    if not has_item_evidence and not has_user_evidence:
        return 0.5, 0.0

    item_score = pathsim(
        evidence.item_shared_paths,
        evidence.item_source_self_paths,
        evidence.item_target_self_paths,
    )
    user_score = pathsim(
        evidence.user_shared_paths,
        evidence.user_self_paths,
        evidence.user_target_self_paths,
    )
    if not has_item_evidence:
        item_weight = 0.0
    elif not has_user_evidence:
        item_weight = 1.0
    score = item_weight * item_score + (1.0 - item_weight) * user_score
    confidence = clamp(
        math.log1p(
            evidence.item_shared_paths
            + evidence.user_shared_paths
            + evidence.item_source_self_paths
            + evidence.user_self_paths
        )
        / 5.0
    )
    return clamp(score), confidence


SURFACE_INTENT = {
    "store_home": 0.50,
    "character_screen": 0.55,
    "match_preparation": 0.58,
    "failure_recovery": 0.68,
    "event_popup": 0.57,
    "checkout": 0.78,
}


def context_affinity(
    user: UserProfile,
    item: Item,
    context: ExposureContext,
) -> tuple[float, float]:
    category_scores = [user.category_preferences.get(category, 0.5) for category in item.categories]
    category_preference = sum(category_scores) / len(category_scores) if category_scores else 0.5
    surface_intent = SURFACE_INTENT.get(context.surface, 0.5)

    if context.budget_reference and context.budget_reference > 0:
        price_ratio = item.price / context.budget_reference
        price_fit = sigmoid(2.5 * (1.0 - price_ratio) * (0.5 + user.price_sensitivity))
    else:
        price_fit = 0.5

    discount_signal = 0.5 + 0.5 * clamp(item.discount_rate)
    fatigue_signal = 1.0 - clamp(context.session_fatigue)
    pickiness_signal = 1.0 - 0.35 * clamp(user.pickiness)

    score = (
        0.35 * category_preference
        + 0.20 * surface_intent
        + 0.20 * price_fit
        + 0.10 * discount_signal
        + 0.10 * fatigue_signal
        + 0.05 * pickiness_signal
    )
    confidence = 0.75 if item.categories else 0.55
    return clamp(score), confidence


class BehaviorSimulationScorer:
    def __init__(
        self,
        config: ScoringConfig | None = None,
        calibration: PlattCalibration | None = None,
    ) -> None:
        self.config = config or ScoringConfig()
        self.calibration = calibration or PlattCalibration()

    def score(
        self,
        *,
        user: UserProfile,
        item: Item,
        context: ExposureContext,
        interactions: Sequence[BehaviorEvent],
        kg_evidence: KnowledgeGraphEvidence,
        base_model_probability: float | None,
        agent_assessment: AgentAssessment,
    ) -> SimulationResult:
        if not item.eligible or item.already_owned:
            reason = "상품이 판매 대상이 아님" if not item.eligible else "사용자가 이미 보유한 상품"
            return SimulationResult(
                probability=0.0,
                confidence=1.0,
                eligible=False,
                is_calibrated=self.calibration.is_fitted,
                components={"base": 0.0},
                reasons=(reason,),
                contradictions=(),
                model_version=self.config.model_version,
                calibration_version=self.calibration.version,
                scalar_purchase_probability=0.0,
            )

        base_probability = (
            clamp(base_model_probability, EPSILON, 1.0 - EPSILON)
            if base_model_probability is not None
            else smoothed_purchase_rate(
                interactions,
                self.config.fallback_prior_rate,
                self.config.fallback_prior_strength,
            )
        )
        episodic_score, episodic_confidence = episodic_affinity(
            interactions, item, context.timestamp, self.config
        )
        kg_score, kg_confidence = knowledge_graph_affinity(
            kg_evidence, self.config.kg_item_weight
        )
        context_score, context_confidence = context_affinity(user, item, context)
        agent_score = clamp(agent_assessment.likelihood, EPSILON, 1.0 - EPSILON)
        agent_ranking_score = clamp(
            (
                agent_assessment.relative_preference_score
                if agent_assessment.relative_preference_score is not None
                else agent_assessment.likelihood
            ),
            EPSILON,
            1.0 - EPSILON,
        )
        agent_confidence = clamp(agent_assessment.confidence)
        rollout_confidence = clamp(
            (
                agent_assessment.rollout_confidence
                if agent_assessment.rollout_confidence is not None
                else agent_assessment.confidence
            )
        )

        combined_logit = logit(base_probability)
        combined_logit += (
            self.config.episodic_logit_weight
            * episodic_confidence
            * logit(episodic_score)
        )
        combined_logit += self.config.kg_logit_weight * kg_confidence * logit(kg_score)
        combined_logit += (
            self.config.context_logit_weight
            * context_confidence
            * logit(context_score)
        )
        agent_logit = logit(agent_score)
        if self.config.agent_logit_reference == "base":
            agent_logit -= logit(base_probability)
        elif self.config.agent_logit_reference != "neutral":
            raise ValueError(
                "agent_logit_reference must be 'neutral' or 'base'"
            )
        combined_logit += (
            self.config.agent_logit_weight
            * agent_confidence
            * agent_logit
        )
        rollout_score = (
            clamp(
                agent_assessment.rollout_probability,
                EPSILON,
                1.0 - EPSILON,
            )
            if agent_assessment.rollout_probability is not None
            else None
        )
        if rollout_score is not None and self.config.rollout_logit_weight:
            rollout_logit = logit(rollout_score)
            if self.config.rollout_logit_reference == "base":
                rollout_logit -= logit(base_probability)
            elif self.config.rollout_logit_reference != "neutral":
                raise ValueError(
                    "rollout_logit_reference must be 'neutral' or 'base'"
                )
            combined_logit += (
                self.config.rollout_logit_weight
                * rollout_confidence
                * rollout_logit
            )

        fused_probability = sigmoid(combined_logit)
        rollout_output_blend = clamp(self.config.rollout_output_blend)
        raw_probability, shrunk_probability = (
            combine_fused_and_rollout_probability(
                fused_probability,
                rollout_score,
                rollout_output_blend=rollout_output_blend,
                fusion_logit_shrink=self.config.fusion_logit_shrink,
                fusion_prior_anchor=self.config.fusion_prior_anchor,
            )
        )
        fusion_logit_shrink = clamp(self.config.fusion_logit_shrink)
        fusion_prior_anchor = clamp(
            self.config.fusion_prior_anchor,
            EPSILON,
            1.0 - EPSILON,
        )
        final_probability = self.calibration.apply(shrunk_probability)
        evidence_confidence = clamp(
            0.45
            + 0.20 * episodic_confidence
            + 0.10 * kg_confidence
            + 0.10 * context_confidence
            + 0.15 * agent_confidence
        )
        reasons = tuple(agent_assessment.reasons) or (
            "결정론적 행동·문맥·그래프 신호로 계산됨",
        )

        return SimulationResult(
            probability=round(final_probability, 6),
            confidence=round(evidence_confidence, 6),
            eligible=True,
            is_calibrated=self.calibration.is_fitted,
            components={
                "base": round(base_probability, 6),
                "episodic": round(episodic_score, 6),
                "episodic_confidence": round(episodic_confidence, 6),
            "knowledge_graph": round(kg_score, 6),
            "knowledge_graph_confidence": round(kg_confidence, 6),
            "knowledge_graph_retrieval_quality": round(kg_confidence, 6),
            "knowledge_graph_retrieval_support": round(
                kg_evidence.retrieval_support, 6
            ),
            "knowledge_graph_retrieval_coverage": round(
                kg_evidence.retrieval_coverage, 6
            ),
            "knowledge_graph_retrieval_consistency": round(
                kg_evidence.retrieval_consistency, 6
            ),
                **{
                    f"kg_{name}": round(value, 6)
                    for name, value in kg_evidence.meta_path_scores.items()
                },
                "context": round(context_score, 6),
                "agent": round(agent_score, 6),
                "agent_ranking_score": round(agent_ranking_score, 6),
                "agent_confidence": round(agent_confidence, 6),
                "rollout_confidence": round(rollout_confidence, 6),
                **(
                    {"rollout": round(rollout_score, 6)}
                    if rollout_score is not None
                    else {}
                ),
                "rollout_validator_adjusted": float(
                    agent_assessment.validator_adjusted
                ),
                "rollout_commitment_adjusted": float(
                    agent_assessment.commitment_adjusted
                ),
                "rollout_counterfactual_adjusted": float(
                    agent_assessment.counterfactual_adjusted
                ),
                "rollout_transition_grounding_strength": round(
                    clamp(agent_assessment.transition_grounding_strength),
                    6,
                ),
                **{
                    f"decision_{name}": round(clamp(value), 6)
                    for name, value in agent_assessment.decision_state.items()
                },
                **{
                    f"intention_{name.lower()}": round(
                        clamp(value),
                        6,
                    )
                    for name, value in (
                        agent_assessment.intention_distribution.items()
                    )
                },
                "raw_fusion": round(fused_probability, 6),
                "rollout_output_blend": round(rollout_output_blend, 6),
                "pre_shrink": round(raw_probability, 6),
                "fusion_logit_shrink": round(fusion_logit_shrink, 6),
                "fusion_prior_anchor": round(fusion_prior_anchor, 6),
                "raw": round(shrunk_probability, 6),
            },
            reasons=reasons,
            contradictions=tuple(agent_assessment.contradictions),
            model_version=self.config.model_version,
            calibration_version=self.calibration.version,
            scalar_purchase_probability=round(final_probability, 6),
        )
