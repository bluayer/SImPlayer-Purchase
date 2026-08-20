from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping


class StoreState(str, Enum):
    ITEM_EXPOSURE = "ITEM_EXPOSURE"
    ITEM_DETAIL = "ITEM_DETAIL"
    PURCHASED = "PURCHASED"
    EXITED = "EXITED"


class UserAction(str, Enum):
    CLICK = "CLICK"
    SKIP = "SKIP"
    EXIT = "EXIT"
    PURCHASE_NOW = "PURCHASE_NOW"
    PURCHASE = "PURCHASE"
    BACK = "BACK"


def _value(value: str | Enum) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _field_name(value: str) -> str:
    field_name = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not field_name or field_name[0].isdigit():
        field_name = f"value_{field_name}"
    return field_name


@dataclass(frozen=True)
class TransitionTiming:
    """Optional timing metadata; no duration is inferred without a timing model."""

    expected_seconds: float | None = None
    timeout_seconds: float | None = None

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "TransitionTiming":
        if not payload:
            return cls()
        expected = payload.get("expected_seconds")
        timeout = payload.get("timeout_seconds")
        return cls(
            expected_seconds=float(expected) if expected is not None else None,
            timeout_seconds=float(timeout) if timeout is not None else None,
        )

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class ActionTransition:
    state: str
    action: str
    next_state: str
    timing: TransitionTiming = field(default_factory=TransitionTiming)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionTransition":
        return cls(
            state=str(payload["state"]),
            action=str(payload["action"]),
            next_state=str(payload["next_state"]),
            timing=TransitionTiming.from_dict(payload.get("timing")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "timing": self.timing.to_dict(),
        }


@dataclass(frozen=True)
class ActionGraph:
    """Declarative game action/event graph shared by actor and rollout."""

    graph_id: str
    version: str
    transitions: tuple[ActionTransition, ...]
    terminal_outcomes: Mapping[str, str]
    default_initial_state: str
    surface_initial_states: Mapping[str, str] = field(default_factory=dict)
    state_output_fields: Mapping[str, str] = field(default_factory=dict)
    ineligible_policy: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    max_depth: int = 4

    def __post_init__(self) -> None:
        if not self.graph_id.strip():
            raise ValueError("action graph requires graph_id")
        if not self.transitions:
            raise ValueError("action graph requires at least one transition")
        if self.max_depth < 1:
            raise ValueError("action graph max_depth must be positive")
        keys: set[tuple[str, str]] = set()
        known_states = {self.default_initial_state, *self.terminal_outcomes}
        for transition in self.transitions:
            key = (transition.state, transition.action)
            if key in keys:
                raise ValueError(
                    f"duplicate action graph transition: {transition.state}/{transition.action}"
                )
            keys.add(key)
            known_states.update((transition.state, transition.next_state))
        for surface, state in self.surface_initial_states.items():
            if state not in known_states:
                raise ValueError(
                    f"surface {surface!r} references unknown initial state {state!r}"
                )
        output_fields = [self.output_field(state) for state in self.states]
        if len(set(output_fields)) != len(output_fields):
            raise ValueError("action graph state output fields must be unique")
        for state, distribution in self.ineligible_policy.items():
            invalid = set(distribution).difference(self.valid_actions(state))
            if invalid:
                raise ValueError(
                    f"ineligible policy for {state!r} has invalid actions: {sorted(invalid)}"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionGraph":
        return cls(
            graph_id=str(payload["graph_id"]),
            version=str(payload.get("version", "1")),
            transitions=tuple(
                ActionTransition.from_dict(value)
                for value in payload.get("transitions", ())
            ),
            terminal_outcomes={
                str(state): str(outcome)
                for state, outcome in payload.get("terminal_outcomes", {}).items()
            },
            default_initial_state=str(payload["default_initial_state"]),
            surface_initial_states={
                str(surface): str(state)
                for surface, state in payload.get(
                    "surface_initial_states", {}
                ).items()
            },
            state_output_fields={
                str(state): str(field_name)
                for state, field_name in payload.get(
                    "state_output_fields", {}
                ).items()
            },
            ineligible_policy={
                str(state): {
                    str(action): float(probability)
                    for action, probability in distribution.items()
                }
                for state, distribution in payload.get(
                    "ineligible_policy", {}
                ).items()
            },
            max_depth=int(payload.get("max_depth", 4)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "version": self.version,
            "default_initial_state": self.default_initial_state,
            "surface_initial_states": dict(self.surface_initial_states),
            "state_output_fields": dict(self.state_output_fields),
            "ineligible_policy": {
                state: dict(distribution)
                for state, distribution in self.ineligible_policy.items()
            },
            "terminal_outcomes": dict(self.terminal_outcomes),
            "max_depth": self.max_depth,
            "transitions": [transition.to_dict() for transition in self.transitions],
        }

    @property
    def states(self) -> tuple[str, ...]:
        return tuple(sorted({transition.state for transition in self.transitions}))

    def initial_state(self, surface: str) -> str:
        return self.surface_initial_states.get(surface, self.default_initial_state)

    def valid_actions(self, state: str | Enum) -> tuple[str, ...]:
        state_value = _value(state)
        return tuple(
            transition.action
            for transition in self.transitions
            if transition.state == state_value
        )

    def transition(
        self,
        state: str | Enum,
        action: str | Enum,
    ) -> ActionTransition:
        state_value = _value(state)
        action_value = _value(action)
        for transition in self.transitions:
            if (
                transition.state == state_value
                and transition.action == action_value
            ):
                return transition
        raise ValueError(f"{action_value} is not valid from {state_value}")

    def is_terminal(self, state: str | Enum) -> bool:
        return _value(state) in self.terminal_outcomes

    def outcome(self, state: str | Enum) -> str | None:
        return self.terminal_outcomes.get(_value(state))

    def purchased(self, state: str | Enum) -> bool:
        return self.outcome(state) == "purchase"

    def output_fields(self) -> dict[str, dict[str, str]]:
        return {
            state: {
                action: _field_name(action)
                for action in self.valid_actions(state)
            }
            for state in self.states
        }

    def output_field(self, state: str) -> str:
        return self.state_output_fields.get(state, _field_name(state))

    def ineligible_distributions(self) -> dict[str, dict[str, float]]:
        distributions: dict[str, dict[str, float]] = {}
        for state in self.states:
            actions = self.valid_actions(state)
            configured = self.ineligible_policy.get(state)
            if configured is not None:
                raw = {
                    action: max(0.0, float(configured.get(action, 0.0)))
                    for action in actions
                }
                total = sum(raw.values())
                if total > 0.0:
                    distributions[state] = {
                        action: probability / total
                        for action, probability in raw.items()
                    }
                    continue
            selected = next(
                (
                    action
                    for action in actions
                    if not self.purchased(
                        self.transition(state, action).next_state
                    )
                ),
                actions[0],
            )
            distributions[state] = {
                action: 1.0 if action == selected else 0.0
                for action in actions
            }
        return distributions


DEFAULT_ACTION_GRAPH = ActionGraph(
    graph_id="game_store_purchase",
    version="1",
    default_initial_state=StoreState.ITEM_EXPOSURE.value,
    surface_initial_states={"checkout": StoreState.ITEM_DETAIL.value},
    state_output_fields={
        StoreState.ITEM_EXPOSURE.value: "exposure",
        StoreState.ITEM_DETAIL.value: "detail",
    },
    ineligible_policy={
        StoreState.ITEM_EXPOSURE.value: {
            UserAction.SKIP.value: 0.98,
            UserAction.EXIT.value: 0.02,
        },
        StoreState.ITEM_DETAIL.value: {
            UserAction.BACK.value: 0.98,
            UserAction.EXIT.value: 0.02,
        },
    },
    terminal_outcomes={
        StoreState.PURCHASED.value: "purchase",
        StoreState.EXITED.value: "exit",
    },
    max_depth=2,
    transitions=(
        ActionTransition("ITEM_EXPOSURE", "CLICK", "ITEM_DETAIL"),
        ActionTransition("ITEM_EXPOSURE", "SKIP", "EXITED"),
        ActionTransition("ITEM_EXPOSURE", "EXIT", "EXITED"),
        ActionTransition("ITEM_EXPOSURE", "PURCHASE_NOW", "PURCHASED"),
        ActionTransition("ITEM_DETAIL", "PURCHASE", "PURCHASED"),
        ActionTransition("ITEM_DETAIL", "BACK", "EXITED"),
        ActionTransition("ITEM_DETAIL", "EXIT", "EXITED"),
    ),
)


VALID_ACTIONS: Mapping[StoreState, tuple[UserAction, ...]] = {
    state: tuple(
        UserAction(action)
        for action in DEFAULT_ACTION_GRAPH.valid_actions(state)
    )
    for state in (StoreState.ITEM_EXPOSURE, StoreState.ITEM_DETAIL)
}


@dataclass(frozen=True)
class StateTransition:
    state: str
    action: str
    next_state: str
    timing: TransitionTiming = field(default_factory=TransitionTiming)
    terminal_outcome: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.terminal_outcome is not None

    @property
    def purchased(self) -> bool:
        return self.terminal_outcome == "purchase"


@dataclass(frozen=True)
class RolloutPath:
    probability: float
    states: tuple[str, ...]
    actions: tuple[str, ...]
    purchased: bool
    terminal_outcome: str | None = None
    transition_timings: tuple[TransitionTiming, ...] = ()

    @property
    def expected_duration_seconds(self) -> float | None:
        if not self.transition_timings or any(
            timing.expected_seconds is None
            for timing in self.transition_timings
        ):
            return None
        return sum(
            float(timing.expected_seconds)
            for timing in self.transition_timings
            if timing.expected_seconds is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "states": list(self.states),
            "actions": list(self.actions),
            "purchased": self.purchased,
            "terminal_outcome": self.terminal_outcome,
            "expected_duration_seconds": self.expected_duration_seconds,
            "transition_timings": [
                timing.to_dict() for timing in self.transition_timings
            ],
        }


@dataclass(frozen=True)
class RolloutResult:
    purchase_probability: float
    paths: tuple[RolloutPath, ...]
    initial_state: str
    max_depth: int
    graph_id: str
    graph_version: str


class DeterministicStoreEnvironment:
    def __init__(self, graph: ActionGraph | None = None) -> None:
        self.graph = graph or DEFAULT_ACTION_GRAPH

    def initial_state(self, surface: str) -> str | StoreState:
        state = self.graph.initial_state(surface)
        if self.graph is DEFAULT_ACTION_GRAPH:
            return StoreState(state)
        return state

    def valid_actions(
        self,
        state: str | Enum,
    ) -> tuple[str | UserAction, ...]:
        actions = self.graph.valid_actions(state)
        if self.graph is DEFAULT_ACTION_GRAPH:
            return tuple(UserAction(action) for action in actions)
        return actions

    def transition(
        self,
        state: str | Enum,
        action: str | Enum,
    ) -> StateTransition:
        transition = self.graph.transition(state, action)
        return StateTransition(
            state=transition.state,
            action=transition.action,
            next_state=transition.next_state,
            timing=transition.timing,
            terminal_outcome=self.graph.outcome(transition.next_state),
        )


def normalize_action_distributions(
    distributions: Mapping[str, Mapping[str, float]],
    *,
    graph: ActionGraph | None = None,
) -> dict[str, dict[str, float]]:
    graph = graph or DEFAULT_ACTION_GRAPH
    normalized: dict[str, dict[str, float]] = {}
    for state in graph.states:
        valid_actions = graph.valid_actions(state)
        raw = distributions.get(state, {})
        values = {
            action: max(0.0, float(raw.get(action, 0.0)))
            for action in valid_actions
        }
        total = sum(values.values())
        if total <= 0.0:
            uniform = 1.0 / len(valid_actions)
            normalized[state] = {action: uniform for action in valid_actions}
            continue
        normalized[state] = {
            action: value / total for action, value in values.items()
        }
    return normalized


def blend_action_distributions(
    proposed: Mapping[str, Mapping[str, float]],
    empirical: Mapping[str, Mapping[str, float]],
    evidence_strength: Mapping[str, float],
    *,
    max_empirical_weight: float = 0.65,
    graph: ActionGraph | None = None,
) -> dict[str, dict[str, float]]:
    graph = graph or DEFAULT_ACTION_GRAPH
    proposed_normalized = normalize_action_distributions(proposed, graph=graph)
    empirical_normalized = normalize_action_distributions(empirical, graph=graph)
    blended: dict[str, dict[str, float]] = {}
    for state, actions in proposed_normalized.items():
        weight = min(
            max_empirical_weight,
            max(0.0, float(evidence_strength.get(state, 0.0))),
        )
        if state not in empirical:
            weight = 0.0
        blended[state] = {
            action: (
                (1.0 - weight) * probability
                + weight * empirical_normalized[state][action]
            )
            for action, probability in actions.items()
        }
    return normalize_action_distributions(blended, graph=graph)


def limit_action_distribution_revision(
    proposed: Mapping[str, Mapping[str, float]],
    revised: Mapping[str, Mapping[str, float]],
    *,
    max_total_variation: float = 0.10,
    graph: ActionGraph | None = None,
) -> dict[str, dict[str, float]]:
    """Bound a critic revision independently within each conditional state."""
    graph = graph or DEFAULT_ACTION_GRAPH
    proposed_normalized = normalize_action_distributions(proposed, graph=graph)
    revised_normalized = normalize_action_distributions(revised, graph=graph)
    limited: dict[str, dict[str, float]] = {}
    bound = max(0.0, min(1.0, max_total_variation))
    for state, proposed_actions in proposed_normalized.items():
        revised_actions = revised_normalized[state]
        total_variation = 0.5 * sum(
            abs(revised_actions[action] - proposed_actions[action])
            for action in proposed_actions
        )
        scale = (
            1.0
            if total_variation <= bound or total_variation == 0.0
            else bound / total_variation
        )
        limited[state] = {
            action: (
                proposed_probability
                + scale * (revised_actions[action] - proposed_probability)
            )
            for action, proposed_probability in proposed_actions.items()
        }
    return normalize_action_distributions(limited, graph=graph)


def rollout_purchase_probability(
    distributions: Mapping[str, Mapping[str, float]],
    *,
    surface: str,
    max_depth: int | None = None,
    environment: DeterministicStoreEnvironment | None = None,
    graph: ActionGraph | None = None,
) -> RolloutResult:
    if environment is not None and graph is not None:
        raise ValueError("provide either environment or graph, not both")
    graph = graph or (environment.graph if environment else DEFAULT_ACTION_GRAPH)
    environment = environment or DeterministicStoreEnvironment(graph)
    effective_max_depth = max_depth if max_depth is not None else graph.max_depth
    if effective_max_depth < 1:
        raise ValueError("max_depth must be positive")
    normalized = normalize_action_distributions(distributions, graph=graph)
    initial_state = _value(environment.initial_state(surface))
    paths: list[RolloutPath] = []

    def visit(
        state: str,
        probability: float,
        states: tuple[str, ...],
        actions: tuple[str, ...],
        timings: tuple[TransitionTiming, ...],
        depth: int,
    ) -> None:
        if graph.is_terminal(state):
            paths.append(
                RolloutPath(
                    probability=probability,
                    states=states,
                    actions=actions,
                    purchased=graph.purchased(state),
                    terminal_outcome=graph.outcome(state),
                    transition_timings=timings,
                )
            )
            return
        if depth >= effective_max_depth:
            paths.append(
                RolloutPath(
                    probability=probability,
                    states=states,
                    actions=actions,
                    purchased=False,
                    transition_timings=timings,
                )
            )
            return
        for action in graph.valid_actions(state):
            action_probability = normalized[state][action]
            if action_probability <= 0.0:
                continue
            transition = environment.transition(state, action)
            visit(
                transition.next_state,
                probability * action_probability,
                (*states, transition.next_state),
                (*actions, transition.action),
                (*timings, transition.timing),
                depth + 1,
            )

    visit(initial_state, 1.0, (initial_state,), (), (), 0)
    purchase_probability = sum(path.probability for path in paths if path.purchased)
    return RolloutResult(
        purchase_probability=min(1.0, max(0.0, purchase_probability)),
        paths=tuple(paths),
        initial_state=initial_state,
        max_depth=effective_max_depth,
        graph_id=graph.graph_id,
        graph_version=graph.version,
    )


def transition_table_payload(
    graph: ActionGraph | None = None,
) -> dict[str, dict[str, str]]:
    graph = graph or DEFAULT_ACTION_GRAPH
    return {
        state: {
            action: graph.transition(state, action).next_state
            for action in graph.valid_actions(state)
        }
        for state in graph.states
    }


def load_action_graph(path: str | Path) -> ActionGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("action graph file must contain a JSON object")
    return ActionGraph.from_dict(payload)
