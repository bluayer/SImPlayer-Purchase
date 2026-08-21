from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    BehaviorEvent,
    Item,
    MemoryDocument,
    ObservationBatch,
    ObservedStateTransition,
    ExposureContext,
    parse_timestamp,
)
from .action_rollout import VALID_ACTIONS, normalize_action_distributions
from .scoring import clamp


OBSERVATION_OPEN = "<simuser-observation>"
OBSERVATION_CLOSE = "</simuser-observation>"
REFLECTION_OPEN = "<simuser-reflection>"
REFLECTION_CLOSE = "</simuser-reflection>"
TRANSITIONS_OPEN = "<simuser-observed-transitions>"
TRANSITIONS_CLOSE = "</simuser-observed-transitions>"
OBSERVED_SOURCES = frozenset(
    {
        "external_observation",
        "historical_import",
        "experiment_observation",
    }
)
_OBSERVATION_PATTERN = re.compile(
    re.escape(OBSERVATION_OPEN)
    + r"\s*(.*?)\s*"
    + re.escape(OBSERVATION_CLOSE),
    re.DOTALL,
)
_TRANSITIONS_PATTERN = re.compile(
    re.escape(TRANSITIONS_OPEN)
    + r"\s*(.*?)\s*"
    + re.escape(TRANSITIONS_CLOSE),
    re.DOTALL,
)


def serialize_observation(batch: ObservationBatch) -> str:
    payload = {
        "schema": "simuser.observation.v1",
        "source": batch.source,
        "page_id": batch.page_id,
        "recommended_item_ids": list(batch.recommended_item_ids),
        "feeling": batch.feeling,
        "review": batch.review,
        "events": [
            {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "item_id": event.item_id,
                "categories": list(event.categories),
                "rating": event.rating,
            }
            for event in batch.events
        ],
    }
    return (
        f"{OBSERVATION_OPEN}"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        f"{OBSERVATION_CLOSE}"
    )


def serialize_reflection(reflection: str) -> str:
    return f"{REFLECTION_OPEN}{reflection.strip()}{REFLECTION_CLOSE}"


def serialize_observed_transitions(
    transitions: Sequence[ObservedStateTransition],
    source: str = "external_observation",
) -> str:
    if source not in OBSERVED_SOURCES:
        raise ValueError("observed transitions require an external source")
    payload = {
        "schema": "simuser.observed-transitions.v2",
        "source": source,
        "transitions": [
            {
                "state": transition.state,
                "action": transition.action,
                "next_state": transition.next_state,
                "timestamp": transition.timestamp.isoformat(),
                "item_id": transition.item_id,
                "categories": list(transition.categories),
                "surface": transition.surface,
                "price_budget_ratio": transition.price_budget_ratio,
                "session_fatigue": transition.session_fatigue,
                "outcome": transition.outcome,
            }
            for transition in transitions
        ],
    }
    return (
        f"{TRANSITIONS_OPEN}"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        f"{TRANSITIONS_CLOSE}"
    )


def behavior_events_from_text(text: str) -> tuple[BehaviorEvent, ...]:
    events: list[BehaviorEvent] = []
    for match in _OBSERVATION_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except (TypeError, ValueError):
            continue
        if payload.get("schema") != "simuser.observation.v1":
            continue
        if payload.get("source") not in OBSERVED_SOURCES:
            continue
        for value in payload.get("events", ()):
            if isinstance(value, dict) and value.get("event_type"):
                events.append(BehaviorEvent.from_dict(value))
    return tuple(events)


def observed_transitions_from_text(
    text: str,
) -> tuple[ObservedStateTransition, ...]:
    transitions: list[ObservedStateTransition] = []
    for match in _TRANSITIONS_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except (TypeError, ValueError):
            continue
        if payload.get("schema") not in {
            "simuser.observed-transitions.v1",
            "simuser.observed-transitions.v2",
        }:
            continue
        if payload.get("source") not in OBSERVED_SOURCES:
            continue
        for value in payload.get("transitions", ()):
            if isinstance(value, dict):
                try:
                    transitions.append(ObservedStateTransition.from_dict(value))
                except (KeyError, TypeError, ValueError):
                    continue
    return tuple(transitions)


def transitions_from_observation(
    batch: ObservationBatch,
) -> tuple[ObservedStateTransition, ...]:
    if batch.transitions:
        return batch.transitions

    transitions: list[ObservedStateTransition] = []
    grouped: dict[tuple[str, str], list[BehaviorEvent]] = {}
    for event in batch.events:
        if not event.item_id:
            continue
        key = (event.timestamp.isoformat(), event.item_id)
        grouped.setdefault(key, []).append(event)

    for events in grouped.values():
        event_types = {event.event_type for event in events}
        representative = events[0]

        def append(state: str, action: str, next_state: str) -> None:
            transitions.append(
                ObservedStateTransition(
                    state=state,
                    action=action,
                    next_state=next_state,
                    timestamp=representative.timestamp,
                    item_id=representative.item_id,
                    categories=representative.categories,
                    surface=batch.page_id or "",
                    outcome=(
                        "purchase"
                        if next_state == "PURCHASED"
                        else "continued"
                        if next_state not in {"EXITED", "PURCHASED"}
                        else "no_purchase"
                    ),
                )
            )

        clicked = bool({"click", "clicked"} & event_types)
        purchased = bool(
            {"purchase", "purchased", "payment_success"} & event_types
        )
        exited = "exit" in event_types
        if purchased and not clicked:
            append(
                "ITEM_EXPOSURE",
                "PURCHASE_NOW",
                "PURCHASE_CONFIRMATION",
            )
            append(
                "PURCHASE_CONFIRMATION",
                "CONFIRM_PURCHASE",
                "PAYMENT_PROCESSING",
            )
            append(
                "PAYMENT_PROCESSING",
                "PAYMENT_SUCCESS",
                "PURCHASED",
            )
            continue
        if not clicked:
            append(
                "ITEM_EXPOSURE",
                "EXIT" if exited else "SKIP",
                "EXITED",
            )
            continue

        append("ITEM_EXPOSURE", "CLICK", "ITEM_DETAIL")
        if purchased:
            append(
                "ITEM_DETAIL",
                "START_PURCHASE",
                "PURCHASE_CONFIRMATION",
            )
            append(
                "PURCHASE_CONFIRMATION",
                "CONFIRM_PURCHASE",
                "PAYMENT_PROCESSING",
            )
            append(
                "PAYMENT_PROCESSING",
                "PAYMENT_SUCCESS",
                "PURCHASED",
            )
        else:
            append(
                "ITEM_DETAIL",
                "EXIT" if exited else "BACK",
                "EXITED" if exited else "ITEM_EXPOSURE",
            )
            if not exited:
                append("ITEM_EXPOSURE", "SKIP", "EXITED")
    return tuple(transitions)


def empirical_transition_policy(
    texts: Sequence[str],
    *,
    item: Item,
    now: datetime,
    context: ExposureContext,
    prior_policy: Mapping[str, Mapping[str, float]] | None = None,
    prior_strength: float = 4.0,
    max_observed_weight: float = 0.35,
) -> tuple[dict[str, dict[str, float]], dict[str, float], int]:
    transitions: list[ObservedStateTransition] = []
    seen: set[tuple[object, ...]] = set()
    for text in texts:
        for transition in observed_transitions_from_text(text):
            key = (
                transition.state,
                transition.action,
                transition.next_state,
                transition.timestamp.isoformat(),
                transition.item_id,
            )
            if key in seen:
                continue
            seen.add(key)
            valid_actions = VALID_ACTIONS.get(transition.state)
            if valid_actions is None:
                continue
            if transition.action not in {
                action.value for action in valid_actions
            }:
                continue
            transitions.append(transition)

    counts = {
        state.value: {action.value: 0.0 for action in actions}
        for state, actions in VALID_ACTIONS.items()
    }
    totals = {state.value: 0.0 for state in VALID_ACTIONS}
    item_categories = set(item.categories)
    for transition in transitions:
        if transition.state not in counts:
            continue
        if transition.action not in counts[transition.state]:
            continue
        transition_categories = set(transition.categories)
        if transition.item_id == item.item_id:
            relevance = 1.0
        elif item_categories and transition_categories:
            overlap = len(item_categories.intersection(transition_categories))
            union = len(item_categories.union(transition_categories))
            relevance = 0.65 * overlap / max(1, union)
        else:
            relevance = 0.05
        if relevance <= 0.0:
            continue
        age_days = max(
            0.0,
            (now - transition.timestamp).total_seconds() / 86400.0,
        )
        recency = math.exp(-math.log(2.0) * age_days / 60.0)
        surface = (
            1.0
            if transition.surface == context.surface
            else 0.8
            if not transition.surface
            or transition.surface == "simuser-half-split-history"
            else 0.5
        )
        ratio = 0.7
        if (
            context.budget_reference
            and transition.price_budget_ratio is not None
        ):
            current_ratio = item.price / context.budget_reference
            ratio = math.exp(
                -abs(current_ratio - transition.price_budget_ratio)
            )
        fatigue = 0.7
        if transition.session_fatigue is not None:
            fatigue = 1.0 - min(
                1.0,
                abs(context.session_fatigue - transition.session_fatigue),
            )
        weight = relevance * recency * surface * ratio * fatigue
        counts[transition.state][transition.action] += weight
        totals[transition.state] += weight

    distributions: dict[str, dict[str, float]] = {}
    strengths: dict[str, float] = {}
    normalized_prior = normalize_action_distributions(prior_policy or {})
    for state, action_counts in counts.items():
        total = totals[state]
        if total <= 0.0:
            continue
        raw_distribution = {
            action: count / total for action, count in action_counts.items()
        }
        observed_weight = min(
            max(0.0, max_observed_weight),
            total / (total + max(prior_strength, 1e-9)),
        )
        distributions[state] = {
            action: (
                (1.0 - observed_weight) * normalized_prior[state][action]
                + observed_weight * raw_distribution[action]
            )
            for action in action_counts
        }
        strengths[state] = observed_weight
    return distributions, strengths, len(transitions)


def observed_at_from_record(record: Any) -> datetime | None:
    for key in ("eventTimestamp", "createdAt", "updatedAt", "timestamp"):
        value = _get(record, key)
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return parse_timestamp(value)
            except ValueError:
                continue
    return None


def record_text(record: Any) -> str:
    if isinstance(record, str):
        return record
    content = _get(record, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if not content:
            return ""
        for key in ("text", "value", "summary"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    for key in ("text", "memory", "summary"):
        value = _get(record, key)
        if isinstance(value, str):
            return value
    return ""


def record_relevance(record: Any) -> float:
    for key in ("relevanceScore", "score", "similarityScore"):
        value = _get(record, key)
        if value is not None:
            try:
                return clamp(float(value))
            except (TypeError, ValueError):
                pass
    return 0.5


def record_namespace(record: Any) -> str:
    namespaces = _get(record, "namespaces")
    if isinstance(namespaces, (list, tuple)):
        return ",".join(str(value) for value in namespaces)
    for key in ("namespace", "namespacePath"):
        value = _get(record, key)
        if isinstance(value, str):
            return value
    return ""


def rerank_memory_documents(
    documents: Sequence[MemoryDocument],
    item: Item,
    now: datetime,
    context: ExposureContext | None = None,
    limit: int = 12,
) -> tuple[MemoryDocument, ...]:
    item_terms = {
        value.lower()
        for value in (item.item_id, *item.categories)
        if value and len(value) > 1
    }
    scored: list[tuple[float, int, MemoryDocument]] = []
    seen: set[str] = set()

    for index, document in enumerate(documents):
        normalized = " ".join(document.content.lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        lexical = (
            sum(term in normalized for term in item_terms) / len(item_terms)
            if item_terms
            else 0.0
        )
        observed_at = document.observed_at
        if observed_at is None:
            recency = 0.5
        else:
            age_days = max(0.0, (now - observed_at).total_seconds() / 86400.0)
            recency = math.exp(-math.log(2.0) * age_days / 45.0)

        event_relevance = 0.0
        for event in behavior_events_from_text(document.content):
            if event.item_id == item.item_id:
                event_relevance = 1.0
                break
            if set(event.categories).intersection(item.categories):
                event_relevance = max(event_relevance, 0.7)

        transition_relevance = 0.0
        for transition in observed_transitions_from_text(document.content):
            category_overlap = bool(
                set(transition.categories).intersection(item.categories)
            )
            if transition.item_id == item.item_id:
                item_relevance = 1.0
            elif category_overlap:
                item_relevance = 0.75
            else:
                item_relevance = 0.2
            surface_relevance = (
                1.0
                if context and transition.surface == context.surface
                else 0.5
                if not transition.surface
                else 0.2
            )
            fatigue_relevance = 0.5
            if (
                context is not None
                and transition.session_fatigue is not None
            ):
                fatigue_relevance = 1.0 - min(
                    1.0,
                    abs(
                        transition.session_fatigue
                        - context.session_fatigue
                    ),
                )
            ratio_relevance = 0.5
            if (
                context is not None
                and context.budget_reference
                and transition.price_budget_ratio is not None
            ):
                current_ratio = item.price / context.budget_reference
                ratio_relevance = math.exp(
                    -abs(current_ratio - transition.price_budget_ratio)
                )
            transition_relevance = max(
                transition_relevance,
                0.40 * item_relevance
                + 0.25 * surface_relevance
                + 0.20 * ratio_relevance
                + 0.15 * fatigue_relevance,
            )

        score = (
            0.40 * clamp(document.relevance)
            + 0.15 * lexical
            + 0.15 * recency
            + 0.15 * event_relevance
            + 0.15 * transition_relevance
        )
        scored.append((score, -index, document))

    scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
    return tuple(value[2] for value in scored[:limit])


def memory_document_outcome(document: MemoryDocument) -> str:
    event_types = {
        event.event_type.lower()
        for event in behavior_events_from_text(document.content)
    }
    transition_actions = {
        transition.action
        for transition in observed_transitions_from_text(document.content)
    }
    if "purchase" in event_types or transition_actions.intersection(
        {
            "PURCHASE",
            "PURCHASE_NOW",
            "START_PURCHASE",
            "CONFIRM_PURCHASE",
            "PAYMENT_SUCCESS",
        }
    ):
        return "purchase"
    if event_types.intersection({"dismiss", "refund", "exit"}) or (
        transition_actions.intersection({"SKIP", "BACK", "EXIT"})
    ):
        return "non_purchase"
    return "unknown"


def select_contrastive_memory_documents(
    documents: Sequence[MemoryDocument],
    *,
    limit: int = 9,
    per_outcome: int = 3,
) -> tuple[MemoryDocument, ...]:
    if limit <= 0:
        return ()
    indexed = list(enumerate(documents))
    selected: set[int] = {
        index
        for index, document in indexed
        if document.kind == "current_session"
    }
    for outcome in ("purchase", "non_purchase"):
        matches = [
            index
            for index, document in indexed
            if memory_document_outcome(document) == outcome
            and index not in selected
        ]
        selected.update(matches[: max(0, per_outcome)])
    for index, _ in indexed:
        if len(selected) >= limit:
            break
        selected.add(index)
    return tuple(
        document
        for index, document in indexed
        if index in selected
    )[:limit]


def merge_behavior_events(
    *event_groups: Iterable[BehaviorEvent],
) -> tuple[BehaviorEvent, ...]:
    merged: list[BehaviorEvent] = []
    seen: set[tuple[object, ...]] = set()
    for events in event_groups:
        for event in events:
            key = (
                event.event_type,
                event.timestamp.isoformat(),
                event.item_id,
                event.categories,
                event.rating,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(event)
    merged.sort(key=lambda event: event.timestamp)
    return tuple(merged)


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key)
    return getattr(value, key, None)
