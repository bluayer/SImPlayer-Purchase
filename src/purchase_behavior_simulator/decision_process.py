from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from .action_rollout import (
    ActionGraph,
    DEFAULT_ACTION_GRAPH,
    limit_action_distribution_revision,
    normalize_action_distributions,
    rollout_purchase_probability,
)
from .models import BehaviorEvent, SimulationRequest
from .product_needs import resolve_product_need_profile


INTENTIONS = ("BUY_NOW", "EXPLORE", "DEFER", "REJECT")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    bounded = {key: max(0.0, float(values.get(key, 0.0))) for key in INTENTIONS}
    total = sum(bounded.values())
    if total <= 0.0:
        return {key: 1.0 / len(INTENTIONS) for key in INTENTIONS}
    return {key: value / total for key, value in bounded.items()}


@dataclass(frozen=True)
class DecisionState:
    need_strength: float
    selection_strength: float
    feasibility: float
    urgency: float
    uncertainty: float
    hesitation: float
    evidence_confidence: float
    state_confidence: float
    exit_pressure: float
    repeat_purchase_plausible: bool

    @property
    def commitment_strength(self) -> float:
        return _clamp(
            self.selection_strength
            * self.feasibility
            * (0.45 + 0.55 * self.urgency)
            * (1.0 - 0.60 * self.hesitation)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "commitment_strength": self.commitment_strength,
        }


@dataclass(frozen=True)
class CommitmentResult:
    distributions: Mapping[str, Mapping[str, float]]
    intentions: Mapping[str, float]
    adjusted: bool
    total_variation: Mapping[str, float]


@dataclass(frozen=True)
class CounterfactualResult:
    checks: Mapping[str, bool]
    purchase_probabilities: Mapping[str, float]
    contradictions: tuple[str, ...] = ()


def build_decision_state(request: SimulationRequest) -> DecisionState:
    need_profile = resolve_product_need_profile(request.item)
    game_state = request.game_state
    category_preferences = [
        request.user.category_preferences.get(category, 0.5)
        for category in request.item.categories
    ]
    category_preference = (
        sum(category_preferences) / len(category_preferences)
        if category_preferences
        else 0.5
    )
    rational_need = (
        need_profile.rational * game_state.progression_need
        + 0.35
        * need_profile.rational
        * game_state.recent_failure_intensity
    )
    emotional_need = need_profile.emotional * category_preference
    need_strength = _clamp(rational_need + emotional_need)

    history_score, history_confidence = _history_affinity(
        request.interactions,
        request,
    )
    selection_strength = _clamp(
        0.40 * category_preference
        + 0.35 * need_strength
        + 0.25 * history_score
    )

    capacities = [
        value
        for value in (
            request.context.budget_reference,
            game_state.currency_balance,
        )
        if value is not None and value > 0.0
    ]
    if request.item.price <= 0.0:
        affordability = 1.0
        price_pressure = 0.0
    elif capacities:
        capacity = min(capacities)
        price_ratio = request.item.price / capacity
        affordability = _clamp(1.25 - 0.65 * price_ratio)
        price_pressure = _clamp(price_ratio - 0.35)
    else:
        affordability = 0.55
        price_pressure = 0.5

    repeat_purchase_plausible = bool(
        set(request.item.categories).intersection(
            {"currency", "subscription", "convenience", "consumable"}
        )
        or request.item.attributes.get("repeatable")
    )
    ownership_pressure = game_state.inventory_overlap
    if request.item.item_id in game_state.owned_item_ids:
        ownership_pressure = 1.0
    ownership_factor = (
        1.0
        if repeat_purchase_plausible
        else 1.0 - 0.85 * ownership_pressure
    )
    feasibility = _clamp(
        affordability
        * ownership_factor
        * (1.0 - 0.85 * game_state.purchase_cooldown)
    )

    limited_time = bool(request.item.attributes.get("limited_time"))
    urgency = _clamp(
        max(
            game_state.event_urgency,
            need_profile.rational
            * game_state.recent_failure_intensity,
            0.55 if limited_time else 0.0,
        )
    )
    explicit_state_signals = sum(
        (
            game_state.currency_balance is not None,
            bool(game_state.current_goals),
            bool(game_state.owned_item_ids),
            bool(game_state.features),
            game_state.progression_need != 0.5,
            game_state.recent_failure_intensity > 0.0,
            game_state.event_urgency > 0.0,
            game_state.inventory_overlap > 0.0,
        )
    )
    evidence_confidence = _clamp(
        0.20
        + 0.45 * history_confidence
        + 0.05 * min(7, explicit_state_signals)
    )
    state_confidence = _clamp(
        len(game_state.provided_fields) / 4.0
    )
    uncertainty = _clamp(1.0 - evidence_confidence)
    hesitation = _clamp(
        0.25 * request.user.pickiness
        + 0.25 * request.user.price_sensitivity * price_pressure
        + 0.20 * uncertainty
        + 0.15 * (1.0 - urgency)
        + 0.15 * ownership_pressure
    )
    exit_pressure = _clamp(
        0.25 + 0.45 * request.context.session_fatigue
    )
    return DecisionState(
        need_strength=need_strength,
        selection_strength=selection_strength,
        feasibility=feasibility,
        urgency=urgency,
        uncertainty=uncertainty,
        hesitation=hesitation,
        evidence_confidence=evidence_confidence,
        state_confidence=state_confidence,
        exit_pressure=exit_pressure,
        repeat_purchase_plausible=repeat_purchase_plausible,
    )


def intentions_from_state(state: DecisionState) -> dict[str, float]:
    commitment = state.commitment_strength
    uncommitted_interest = state.selection_strength * (1.0 - commitment)
    return _normalize(
        {
            "BUY_NOW": commitment,
            "EXPLORE": uncommitted_interest
            * (1.0 - 0.55 * state.hesitation),
            "DEFER": uncommitted_interest
            * (0.35 + 0.65 * state.hesitation)
            * (1.0 - 0.45 * state.urgency),
            "REJECT": (1.0 - state.selection_strength)
            * (0.45 + 0.55 * (1.0 - state.feasibility)),
        }
    )


def merge_model_decision(
    baseline: DecisionState,
    model_state: Mapping[str, Any] | None,
    model_intentions: Mapping[str, Any] | None,
    *,
    model_confidence: float,
) -> tuple[DecisionState, dict[str, float]]:
    weight = min(0.35, 0.35 * _clamp(model_confidence))
    values = baseline.to_dict()
    for name in (
        "need_strength",
        "selection_strength",
        "feasibility",
        "urgency",
        "uncertainty",
        "hesitation",
    ):
        if model_state and model_state.get(name) is not None:
            values[name] = (
                (1.0 - weight) * float(values[name])
                + weight * _clamp(float(model_state[name]))
            )
    merged = replace(
        baseline,
        need_strength=_clamp(values["need_strength"]),
        selection_strength=_clamp(values["selection_strength"]),
        feasibility=_clamp(values["feasibility"]),
        urgency=_clamp(values["urgency"]),
        uncertainty=_clamp(values["uncertainty"]),
        hesitation=_clamp(values["hesitation"]),
    )
    baseline_intentions = intentions_from_state(merged)
    if not model_intentions:
        return merged, baseline_intentions
    normalized_model = _normalize(
        {
            name: float(model_intentions.get(name, 0.0))
            for name in INTENTIONS
        }
    )
    return merged, _normalize(
        {
            name: (
                (1.0 - weight) * baseline_intentions[name]
                + weight * normalized_model[name]
            )
            for name in INTENTIONS
        }
    )


def apply_commitment_gate(
    distributions: Mapping[str, Mapping[str, float]],
    state: DecisionState,
    intentions: Mapping[str, float] | None = None,
    *,
    max_total_variation: float = 0.08,
    graph: ActionGraph | None = None,
) -> CommitmentResult:
    graph = graph or DEFAULT_ACTION_GRAPH
    proposed = normalize_action_distributions(distributions, graph=graph)
    intent = _normalize(intentions or intentions_from_state(state))
    non_engagement = intent["DEFER"] + intent["REJECT"]
    exposure_target = {
        "CLICK": intent["EXPLORE"] + 0.15 * intent["DEFER"],
        "SKIP": non_engagement * (1.0 - state.exit_pressure),
        "EXIT": non_engagement * state.exit_pressure,
        "PURCHASE_NOW": intent["BUY_NOW"] * (0.25 + 0.75 * state.urgency),
    }
    leave_detail = intent["DEFER"] + intent["EXPLORE"] + intent["REJECT"]
    detail_target = {
        "START_PURCHASE": intent["BUY_NOW"],
        "BACK": leave_detail,
    }
    confirmation_target = {
        "CONFIRM_PURCHASE": intent["BUY_NOW"],
        "CANCEL": intent["EXPLORE"] + intent["DEFER"] + intent["REJECT"],
    }
    target = normalize_action_distributions(
        {
            "ITEM_EXPOSURE": exposure_target,
            "ITEM_DETAIL": detail_target,
            "PURCHASE_CONFIRMATION": confirmation_target,
        },
        graph=graph,
    )
    influence = min(
        0.18,
        0.18 * state.state_confidence * state.evidence_confidence,
    )
    adjusted_states = {
        "ITEM_EXPOSURE",
        "ITEM_DETAIL",
        "PURCHASE_CONFIRMATION",
    }
    requested = {}
    for state_name, action_values in proposed.items():
        if state_name not in adjusted_states:
            requested[state_name] = dict(action_values)
            continue
        requested[state_name] = {
            action: (
                (1.0 - influence) * probability
                + influence * target[state_name][action]
            )
            for action, probability in action_values.items()
        }
    revised = limit_action_distribution_revision(
        proposed,
        requested,
        max_total_variation=max_total_variation,
        graph=graph,
    )
    variation = {
        state_name: 0.5
        * sum(
            abs(revised[state_name][action] - proposed[state_name][action])
            for action in proposed[state_name]
        )
        for state_name in proposed
    }
    return CommitmentResult(
        distributions=revised,
        intentions=intent,
        adjusted=max(variation.values(), default=0.0) > 0.01,
        total_variation=variation,
    )


def evaluate_counterfactual_consistency(
    request: SimulationRequest,
    base_distributions: Mapping[str, Mapping[str, float]],
    current_distributions: Mapping[str, Mapping[str, float]],
    current_state: DecisionState,
    *,
    graph: ActionGraph | None = None,
) -> CounterfactualResult:
    graph = graph or DEFAULT_ACTION_GRAPH
    base = normalize_action_distributions(base_distributions, graph=graph)
    current = normalize_action_distributions(
        current_distributions,
        graph=graph,
    )
    current_probability = rollout_purchase_probability(
        current,
        surface=request.context.surface,
        graph=graph,
    ).purchase_probability
    scenarios: dict[str, DecisionState] = {
        "price_increase": replace(
            current_state,
            feasibility=_clamp(current_state.feasibility * 0.55),
            hesitation=_clamp(current_state.hesitation + 0.20),
        ),
        "need_resolved": replace(
            current_state,
            need_strength=_clamp(current_state.need_strength * 0.25),
            selection_strength=_clamp(
                current_state.selection_strength
                * (
                    0.45
                    if resolve_product_need_profile(request.item).rational
                    >= resolve_product_need_profile(request.item).emotional
                    else 0.80
                )
            ),
            urgency=_clamp(current_state.urgency * 0.50),
        ),
        "urgency_removed": replace(
            current_state,
            urgency=0.0,
            hesitation=_clamp(current_state.hesitation + 0.10),
        ),
    }
    if not current_state.repeat_purchase_plausible:
        scenarios["ownership_constraint"] = replace(
            current_state,
            feasibility=_clamp(current_state.feasibility * 0.10),
            selection_strength=_clamp(
                current_state.selection_strength * 0.40
            ),
            hesitation=_clamp(current_state.hesitation + 0.30),
        )

    probabilities = {"current": current_probability}
    checks: dict[str, bool] = {}
    contradictions: list[str] = []
    for name, scenario_state in scenarios.items():
        scenario_result = apply_commitment_gate(
            base,
            scenario_state,
            intentions_from_state(scenario_state),
            graph=graph,
        )
        probability = rollout_purchase_probability(
            scenario_result.distributions,
            surface=request.context.surface,
            graph=graph,
        ).purchase_probability
        probabilities[name] = probability
        checks[name] = probability <= current_probability + 1e-9
        if not checks[name]:
            contradictions.append(
                f"{name} counterfactual increased purchase probability"
            )
    return CounterfactualResult(
        checks=checks,
        purchase_probabilities=probabilities,
        contradictions=tuple(contradictions),
    )


def _history_affinity(
    interactions: Sequence[BehaviorEvent],
    request: SimulationRequest,
) -> tuple[float, float]:
    item_categories = set(request.item.categories)
    weighted = 0.0
    magnitude = 0.0
    event_weights = {
        "view": 0.10,
        "click": 0.35,
        "try_on": 0.45,
        "equip": 0.60,
        "use": 0.70,
        "purchase": 1.0,
        "dismiss": -0.35,
        "refund": -1.0,
        "exit": -0.25,
    }
    for event in interactions:
        weight = event_weights.get(event.event_type.lower(), 0.0)
        if weight == 0.0:
            continue
        if event.item_id == request.item.item_id:
            relevance = 1.0
        elif item_categories.intersection(event.categories):
            relevance = 0.65
        else:
            continue
        age_days = max(
            0.0,
            (
                request.context.timestamp - event.timestamp
            ).total_seconds()
            / 86400.0,
        )
        recency = math.exp(-math.log(2.0) * age_days / 45.0)
        weighted += weight * relevance * recency
        magnitude += abs(weight) * relevance * recency
    if magnitude <= 0.0:
        return 0.5, 0.0
    return _clamp(0.5 + 0.45 * weighted / magnitude), _clamp(
        1.0 - math.exp(-magnitude)
    )
