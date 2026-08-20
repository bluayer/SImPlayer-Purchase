from __future__ import annotations

import argparse
import json
import math
import random
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    AgentAssessment,
    BehaviorEvent,
    Item,
    KnowledgeGraphEvidence,
    ExposureContext,
    UserProfile,
)
from .neptune_graph import NeptuneGraphConfig, NeptuneGraphEvidenceProvider
from .scoring import BehaviorSimulationScorer, clamp


RATIO_SPECS = {
    "1:1": (10, 10),
    "1:3": (5, 15),
    "1:9": (2, 18),
}
REPEATABLE_LIKE_CATEGORIES = frozenset(
    {"currency", "subscription", "convenience"}
)
EPSILON = 1e-8


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def history_events(
    rows: Sequence[Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> tuple[BehaviorEvent, ...]:
    events: list[BehaviorEvent] = []
    for row in rows:
        item = items[str(row["item_id"])]
        categories = tuple(str(value) for value in item.get("categories", ()))
        timestamp = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
        events.append(
            BehaviorEvent(
                event_type="view",
                timestamp=timestamp,
                item_id=str(row["item_id"]),
                categories=categories,
            )
        )
        for field, event_type in (
            ("clicked", "click"),
            ("purchased", "purchase"),
        ):
            if int(row.get(field, 0)):
                events.append(
                    BehaviorEvent(
                        event_type=event_type,
                        timestamp=timestamp,
                        item_id=str(row["item_id"]),
                        categories=categories,
                    )
                )
    return tuple(events)


def user_profile_payload(user: Mapping[str, Any], actor_id: str) -> dict[str, Any]:
    return {
        "user_id": actor_id,
        "persona_summary": user.get("persona_summary", ""),
        "pickiness": user.get("pickiness", 0.5),
        "price_sensitivity": user.get("price_sensitivity", 0.5),
        "category_preferences": user.get("category_preferences", {}),
        "engagement": user.get("engagement", 0.5),
        "variety": user.get("novelty_affinity", 0.5),
    }


def item_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item["item_id"],
        "product_type": item.get("product_type", "item"),
        "categories": item.get("categories", ()),
        "price": item.get("price", 0.0),
        "discount_rate": item.get("discount_rate", 0.0),
        "components": item.get("components", ()),
        "attributes": {
            key: value
            for key, value in item.items()
            if key
            not in {
                "item_id",
                "product_type",
                "categories",
                "price",
                "discount_rate",
                "components",
                "latent_vector",
                "random_effect",
            }
        },
    }


def repeat_purchase_is_plausible(item: Mapping[str, Any]) -> bool:
    return bool(
        set(str(value) for value in item.get("categories", ())).intersection(
            REPEATABLE_LIKE_CATEGORIES
        )
    )


def graph_evidence(
    *,
    target_item: Mapping[str, Any],
    events: Sequence[BehaviorEvent],
    items: Mapping[str, Mapping[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    provider = NeptuneGraphEvidenceProvider(
        config=NeptuneGraphConfig(endpoint_url="offline://holdout-snapshot"),
        client=object(),
    )
    target_rows: list[dict[str, str]] = [
        {"relationType": "IN_CATEGORY", "neighborId": str(category)}
        for category in target_item.get("categories", ())
    ]
    target_rows.extend(
        (
            {
                "relationType": "TARGETS",
                "neighborId": str(target_item.get("character", "")),
            },
            {
                "relationType": "AVAILABLE_IN",
                "neighborId": str(target_item.get("event_id", "")),
            },
        )
    )
    target_rows.extend(
        {
            "relationType": "CONTAINS",
            "neighborId": str(
                component.get("item_id", component.get("product_id", ""))
                if isinstance(component, Mapping)
                else component
            ),
        }
        for component in target_item.get("components", ())
    )
    event_names = {
        "view": "VIEWED",
        "click": "CLICKED",
        "purchase": "PURCHASED",
    }
    history_rows: list[dict[str, str]] = []
    for event in events:
        if not event.item_id or event.item_id not in items:
            continue
        source = items[event.item_id]
        common = {
            "sourceItemId": event.item_id,
            "interactionType": event_names.get(
                event.event_type, event.event_type.upper()
            ),
            "interactionTimestamp": event.timestamp.isoformat(),
        }
        for category in source.get("categories", ()):
            history_rows.append(
                {
                    **common,
                    "relationType": "IN_CATEGORY",
                    "neighborId": str(category),
                }
            )
        history_rows.extend(
            (
                {
                    **common,
                    "relationType": "TARGETS",
                    "neighborId": str(source.get("character", "")),
                },
                {
                    **common,
                    "relationType": "AVAILABLE_IN",
                    "neighborId": str(source.get("event_id", "")),
                },
            )
        )
        history_rows.extend(
            {
                **common,
                "relationType": "CONTAINS",
                "neighborId": str(
                    component.get("item_id", component.get("product_id", ""))
                    if isinstance(component, Mapping)
                    else component
                ),
            }
            for component in source.get("components", ())
        )
    return asdict(
        provider.calculate_evidence(
            target_rows=target_rows,
            history_rows=history_rows,
            now=now,
            target_item_id=str(target_item.get("item_id", "")),
        )
    )


@dataclass(frozen=True)
class HoldoutCase:
    case_id: str
    original_user_id: str
    actor_id: str
    session_id: str
    item_id: str
    label: int
    oracle_probability: float | None
    observed_initial_state: str
    observed_next_action: str
    observed_detail_action: str | None
    ratio_membership: tuple[str, ...]
    request: Mapping[str, Any]

    def blind_payload(self) -> dict[str, Any]:
        return {
            "operation": "evaluate_snapshot",
            "request": dict(self.request),
        }


@dataclass(frozen=True)
class HoldoutProtocol:
    run_id: str
    users: int
    history_fraction: float
    history_impressions_per_user: int
    cases: tuple[HoldoutCase, ...]
    bootstrap_payloads: tuple[Mapping[str, Any], ...]
    natural_metrics: Mapping[str, Any]


def observed_action_labels(
    row: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    """Map sampled behavior to the deterministic game-store action states."""
    purchased = bool(int(row.get("purchased", 0)))
    clicked = bool(int(row.get("clicked", 0)))
    if str(row.get("surface", "")) == "checkout":
        return (
            "ITEM_DETAIL",
            "PURCHASE" if purchased else "BACK",
            None,
        )
    if clicked:
        return (
            "ITEM_EXPOSURE",
            "CLICK",
            "PURCHASE" if purchased else "BACK",
        )
    return (
        "ITEM_EXPOSURE",
        "PURCHASE_NOW" if purchased else "SKIP",
        None,
    )


def unique_holdout_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    purchased_items = {
        str(row["item_id"]) for row in rows if int(row.get("purchased", 0))
    }
    positives: dict[str, Mapping[str, Any]] = {}
    negatives: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item_id = str(row["item_id"])
        if int(row.get("purchased", 0)):
            positives.setdefault(item_id, row)
        elif item_id not in purchased_items:
            negatives[item_id] = row
    return list(positives.values()), list(negatives.values())


def ratio_membership(
    positive_index: int | None,
    negative_index: int | None,
) -> tuple[str, ...]:
    memberships = []
    for name, (positive_count, negative_count) in RATIO_SPECS.items():
        if positive_index is not None and positive_index < positive_count:
            memberships.append(name)
        if negative_index is not None and negative_index < negative_count:
            memberships.append(name)
    return tuple(memberships)


def binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    count = max(1, len(labels))
    positive_probabilities = [
        probability
        for probability, label in zip(probabilities, labels)
        if label == 1
    ]
    negative_probabilities = [
        probability
        for probability, label in zip(probabilities, labels)
        if label == 0
    ]
    predictions = [int(value >= threshold) for value in probabilities]
    true_positive = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(labels, predictions)
    )
    false_positive = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(labels, predictions)
    )
    true_negative = sum(
        label == 0 and prediction == 0
        for label, prediction in zip(labels, predictions)
    )
    false_negative = sum(
        label == 1 and prediction == 0
        for label, prediction in zip(labels, predictions)
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    observed = sum(labels) / count
    mean_prediction = sum(probabilities) / count
    expected_positive_count = sum(probabilities)
    positive_mean_prediction = (
        sum(positive_probabilities) / len(positive_probabilities)
        if positive_probabilities
        else 0.0
    )
    negative_mean_prediction = (
        sum(negative_probabilities) / len(negative_probabilities)
        if negative_probabilities
        else 0.0
    )
    return {
        "count": len(labels),
        "positives": sum(labels),
        "observed_rate": round(observed, 8),
        "mean_prediction": round(mean_prediction, 8),
        "mean_bias_pp": round((mean_prediction - observed) * 100.0, 6),
        "expected_positive_count": round(expected_positive_count, 8),
        "positive_count_error": round(expected_positive_count - sum(labels), 8),
        "probability_mae": round(
            sum(
                abs(probability - label)
                for probability, label in zip(probabilities, labels)
            )
            / count,
            8,
        ),
        "positive_mean_prediction": round(positive_mean_prediction, 8),
        "negative_mean_prediction": round(negative_mean_prediction, 8),
        "mean_prediction_separation_pp": round(
            (positive_mean_prediction - negative_mean_prediction) * 100.0,
            6,
        ),
        "accuracy": round((true_positive + true_negative) / count, 8),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1": round(
            2.0 * precision * recall / max(EPSILON, precision + recall), 8
        ),
        "brier": round(
            sum(
                (probability - label) ** 2
                for probability, label in zip(probabilities, labels)
            )
            / count,
            8,
        ),
        "log_loss": round(
            -sum(
                label * math.log(clamp(probability, EPSILON, 1.0 - EPSILON))
                + (1 - label)
                * math.log(
                    1.0 - clamp(probability, EPSILON, 1.0 - EPSILON)
                )
                for probability, label in zip(probabilities, labels)
            )
            / count,
            8,
        ),
        "roc_auc": round(roc_auc(probabilities, labels), 8),
        "average_precision": round(average_precision(probabilities, labels), 8),
        "confusion_matrix": {
            "tp": true_positive,
            "fp": false_positive,
            "tn": true_negative,
            "fn": false_negative,
        },
    }


def roc_auc(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ordered = sorted(zip(probabilities, labels), key=lambda value: value[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def average_precision(
    probabilities: Sequence[float], labels: Sequence[int]
) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ordered = sorted(
        zip(probabilities, labels), key=lambda value: value[0], reverse=True
    )
    true_positive = 0
    total = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            true_positive += 1
            total += true_positive / rank
    return total / positives


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    groups: list[list[int]] = [[] for _ in range(bins)]
    for index, probability in enumerate(probabilities):
        groups[min(bins - 1, int(clamp(probability) * bins))].append(index)
    table: list[dict[str, Any]] = []
    ece = 0.0
    for group in groups:
        if not group:
            continue
        predicted = sum(probabilities[index] for index in group) / len(group)
        observed = sum(labels[index] for index in group) / len(group)
        ece += len(group) / max(1, len(labels)) * abs(predicted - observed)
        table.append(
            {
                "count": len(group),
                "mean_prediction": round(predicted, 8),
                "observed_rate": round(observed, 8),
                "gap_pp": round((predicted - observed) * 100.0, 6),
            }
        )
    return round(ece, 8), table


def deterministic_natural_evaluation(
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    users: Mapping[str, Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    *,
    history_fraction: float,
    history_limit: int,
) -> dict[str, Any]:
    scorer = BehaviorSimulationScorer()
    labels: list[int] = []
    predictions: list[float] = []
    oracle: list[float] = []
    for user_id, rows in grouped_rows.items():
        cutoff = max(1, int(len(rows) * history_fraction))
        history_rows = rows[max(0, cutoff - history_limit) : cutoff]
        heldout_rows = rows[cutoff:]
        events = history_events(history_rows, items)
        user = users[user_id]
        profile = UserProfile.from_dict(user)
        for row in heldout_rows:
            item = items[str(row["item_id"])]
            timestamp = datetime.fromisoformat(
                str(row["timestamp"]).replace("Z", "+00:00")
            )
            evidence = KnowledgeGraphEvidence.from_dict(
                graph_evidence(
                    target_item=item,
                    events=events,
                    items=items,
                    now=timestamp,
                )
            )
            result = scorer.score(
                user=profile,
                item=Item.from_dict(item),
                context=ExposureContext.from_dict(
                    {
                        "surface": row.get("surface", "store_home"),
                        "session_fatigue": row.get("session_fatigue", 0.0),
                        "budget_reference": user.get("spending_power"),
                        "timestamp": row["timestamp"],
                        "features": {
                            "progression_need": row.get("progression_need", 0.0)
                        },
                    }
                ),
                interactions=events,
                kg_evidence=evidence,
                base_model_probability=None,
                agent_assessment=AgentAssessment(),
            )
            labels.append(int(row["purchased"]))
            predictions.append(result.probability)
            oracle.append(float(row["ground_truth_purchase_probability"]))
    metrics = binary_metrics(labels, predictions)
    ece, calibration = expected_calibration_error(labels, predictions)
    oracle_metrics = binary_metrics(labels, oracle)
    oracle_ece, oracle_calibration = expected_calibration_error(labels, oracle)
    metrics["ece"] = ece
    metrics["calibration"] = calibration
    metrics["mae_to_oracle_probability"] = round(
        sum(abs(left - right) for left, right in zip(predictions, oracle))
        / max(1, len(labels)),
        8,
    )
    oracle_metrics["ece"] = oracle_ece
    oracle_metrics["calibration"] = oracle_calibration
    return {
        "deterministic_without_llm": metrics,
        "inference_isolated_oracle_reference": oracle_metrics,
    }


def prepare_protocol(
    data_dir: Path,
    *,
    selected_users: int = 10,
    history_fraction: float = 0.5,
    history_limit: int = 50,
    seed: int = 20260818,
    excluded_user_ids: Sequence[str] = (),
    compute_natural_metrics: bool = True,
) -> HoldoutProtocol:
    users = {
        str(row["user_id"]): row for row in read_jsonl(data_dir / "users.jsonl")
    }
    items = {
        str(row["item_id"]): row for row in read_jsonl(data_dir / "items.jsonl")
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(data_dir / "impressions.jsonl"):
        grouped[str(row["user_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row["timestamp"]))

    eligible: list[str] = []
    candidate_sets: dict[
        str, tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]
    ] = {}
    excluded = set(excluded_user_ids)
    for user_id, rows in grouped.items():
        if user_id in excluded:
            continue
        cutoff = max(1, int(len(rows) * history_fraction))
        history_purchases = {
            str(row["item_id"])
            for row in rows[:cutoff]
            if int(row.get("purchased", 0))
        }
        positives, negatives = unique_holdout_candidates(rows[cutoff:])
        positives = [
            row
            for row in positives
            if str(row["item_id"]) not in history_purchases
            or repeat_purchase_is_plausible(items[str(row["item_id"])])
        ]
        negatives = [
            row
            for row in negatives
            if str(row["item_id"]) not in history_purchases
            or repeat_purchase_is_plausible(items[str(row["item_id"])])
        ]
        if len(positives) >= 10 and len(negatives) >= 18:
            eligible.append(user_id)
            candidate_sets[user_id] = (positives, negatives)
    if len(eligible) < selected_users:
        raise ValueError(
            f"only {len(eligible)} users have enough holdout positives/negatives"
        )

    # Evaluation sampling must be reproducible; this is not security randomness.
    rng = random.Random(seed)  # nosec B311
    chosen_users = sorted(rng.sample(eligible, selected_users))
    run_id = f"half-{seed}-{uuid.uuid4().hex[:8]}"
    cases: list[HoldoutCase] = []
    bootstrap_payloads: list[Mapping[str, Any]] = []
    for user_id in chosen_users:
        actor_id = f"eval-{run_id}-{user_id}"
        session_id = f"history-{run_id}-{user_id}"
        rows = grouped[user_id]
        cutoff = max(1, int(len(rows) * history_fraction))
        selected_history = rows[max(0, cutoff - history_limit) : cutoff]
        events = history_events(selected_history, items)
        bootstrap_payloads.append(
            {
                "operation": "initialize_memory",
                "observation": {
                    "user_id": actor_id,
                    "session_id": session_id,
                    "source": "historical_import",
                    "page_id": "simuser-half-split-history",
                    "recommended_item_ids": [
                        str(row["item_id"]) for row in selected_history
                    ],
                    "events": [
                        {
                            "event_type": event.event_type,
                            "timestamp": event.timestamp.isoformat(),
                            "item_id": event.item_id,
                            "categories": list(event.categories),
                        }
                        for event in events
                    ],
                },
            }
        )
        positives, negatives = candidate_sets[user_id]
        local_rng = random.Random(  # nosec B311
            f"{seed}:{user_id}"
        )
        selected_positive = local_rng.sample(positives, 10)
        selected_negative = local_rng.sample(negatives, 18)
        user = users[user_id]
        for label, selected in ((1, selected_positive), (0, selected_negative)):
            for index, row in enumerate(selected):
                item = items[str(row["item_id"])]
                timestamp = datetime.fromisoformat(
                    str(row["timestamp"]).replace("Z", "+00:00")
                )
                memberships = ratio_membership(
                    index if label else None,
                    index if not label else None,
                )
                request = {
                    "user": user_profile_payload(user, actor_id),
                    "item": item_payload(item),
                    "context": {
                        "surface": row.get("surface", "store_home"),
                        "session_fatigue": row.get("session_fatigue", 0.0),
                        "budget_reference": user.get("spending_power"),
                        "timestamp": row["timestamp"],
                        "features": dict(
                            row.get(
                                "context_features",
                                {
                                    "progression_need": row.get(
                                        "progression_need",
                                        0.0,
                                    )
                                },
                            )
                        ),
                    },
                    "game_state": dict(row.get("game_state", {})),
                    "interactions": [],
                    "kg_evidence": graph_evidence(
                        target_item=item,
                        events=events,
                        items=items,
                        now=timestamp,
                    ),
                    "memory_session_id": session_id,
                    "request_id": (
                        f"holdout-{run_id}-{user_id}-{row['item_id']}"
                    ),
                }
                (
                    observed_initial_state,
                    observed_next_action,
                    observed_detail_action,
                ) = observed_action_labels(row)
                cases.append(
                    HoldoutCase(
                        case_id=f"{user_id}:{row['item_id']}",
                        original_user_id=user_id,
                        actor_id=actor_id,
                        session_id=session_id,
                        item_id=str(row["item_id"]),
                        label=label,
                        oracle_probability=float(
                            row["ground_truth_purchase_probability"]
                        ),
                        observed_initial_state=observed_initial_state,
                        observed_next_action=observed_next_action,
                        observed_detail_action=observed_detail_action,
                        ratio_membership=memberships,
                        request=request,
                    )
                )

    natural_metrics = (
        deterministic_natural_evaluation(
            grouped,
            users,
            items,
            history_fraction=history_fraction,
            history_limit=history_limit,
        )
        if compute_natural_metrics
        else {}
    )
    return HoldoutProtocol(
        run_id=run_id,
        users=selected_users,
        history_fraction=history_fraction,
        history_impressions_per_user=history_limit,
        cases=tuple(cases),
        bootstrap_payloads=tuple(bootstrap_payloads),
        natural_metrics=natural_metrics,
    )


class AgentCoreRuntimeClient:
    def __init__(self, runtime_arn: str, region_name: str = "us-east-1") -> None:
        import boto3

        self.runtime_arn = runtime_arn
        self.client = boto3.client("bedrock-agentcore", region_name=region_name)

    def invoke(
        self,
        payload: Mapping[str, Any],
        *,
        user_id: str,
        retries: int = 3,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                return self._invoke_once(payload, user_id=user_id)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
        if last_error is None:
            raise RuntimeError("runtime invocation failed without an exception")
        raise last_error

    def _invoke_once(
        self,
        payload: Mapping[str, Any],
        *,
        user_id: str,
    ) -> dict[str, Any]:
        response = self.client.invoke_agent_runtime(
            agentRuntimeArn=self.runtime_arn,
            runtimeSessionId=f"eval-{uuid.uuid4().hex}-{uuid.uuid4().hex[:8]}",
            runtimeUserId=user_id,
            contentType="application/json",
            accept="application/json",
            payload=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        chunks = []
        for event in response["response"]:
            if isinstance(event, (bytes, bytearray)):
                chunks.append(bytes(event))
            elif "chunk" in event:
                chunks.append(event["chunk"]["bytes"])
            elif "payloadPart" in event:
                chunks.append(event["payloadPart"]["bytes"])
            elif "internalServerException" in event:
                raise RuntimeError(str(event["internalServerException"]))
        if response.get("statusCode", 200) >= 300:
            raise RuntimeError(
                f"AgentCore returned HTTP {response['statusCode']}: "
                f"{b''.join(chunks)!r}"
            )
        return json.loads(b"".join(chunks).decode("utf-8"))


def save_protocol(protocol: HoldoutProtocol, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "bootstrap.jsonl", protocol.bootstrap_payloads)
    write_jsonl(
        output_dir / "blind_cases.jsonl",
        (
            {
                "case_id": case.case_id,
                "actor_id": case.actor_id,
                "payload": case.blind_payload(),
            }
            for case in protocol.cases
        ),
    )
    write_jsonl(
        output_dir / "answer_key.jsonl",
        (
            {
                "case_id": case.case_id,
                "original_user_id": case.original_user_id,
                "item_id": case.item_id,
                "label": case.label,
                "oracle_probability": case.oracle_probability,
                "observed_initial_state": case.observed_initial_state,
                "observed_next_action": case.observed_next_action,
                "observed_detail_action": case.observed_detail_action,
                "ratio_membership": case.ratio_membership,
            }
            for case in protocol.cases
        ),
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "run_id": protocol.run_id,
                "users": protocol.users,
                "history_fraction": protocol.history_fraction,
                "history_impressions_per_user": (
                    protocol.history_impressions_per_user
                ),
                "candidate_cases": len(protocol.cases),
                "ratio_specs": RATIO_SPECS,
                "natural_metrics": protocol.natural_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_live_evaluation(
    protocol: HoldoutProtocol,
    runtime_arn: str,
    output_dir: Path,
    *,
    workers: int = 6,
) -> dict[str, Any]:
    client = AgentCoreRuntimeClient(runtime_arn)
    bootstrap_results = []
    for payload in protocol.bootstrap_payloads:
        actor_id = str(payload["observation"]["user_id"])
        bootstrap_results.append(client.invoke(payload, user_id=actor_id))

    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                client.invoke,
                case.blind_payload(),
                user_id=case.actor_id,
            ): case
            for case in protocol.cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                results[case.case_id] = future.result()
            except Exception as exc:
                errors[case.case_id] = str(exc)

    scored_cases = [
        case for case in protocol.cases if case.case_id in results
    ]
    ratio_reports: dict[str, Any] = {}
    for ratio in RATIO_SPECS:
        selected = [
            case for case in scored_cases if ratio in case.ratio_membership
        ]
        labels = [case.label for case in selected]
        final_probability = [
            float(results[case.case_id]["probability"]) for case in selected
        ]
        agent_likelihood = [
            float(results[case.case_id]["components"]["agent"])
            for case in selected
        ]
        ratio_reports[ratio] = {
            "final_purchase_probability": binary_metrics(
                labels, final_probability
            ),
            "agent_likelihood": binary_metrics(labels, agent_likelihood),
        }

    all_labels = [case.label for case in scored_cases]
    all_final = [
        float(results[case.case_id]["probability"]) for case in scored_cases
    ]
    all_agent = [
        float(results[case.case_id]["components"]["agent"])
        for case in scored_cases
    ]
    report = {
        "protocol": {
            "run_id": protocol.run_id,
            "users": protocol.users,
            "history_fraction": protocol.history_fraction,
            "history_impressions_per_user": (
                protocol.history_impressions_per_user
            ),
            "candidate_cases": len(protocol.cases),
            "successful_cases": len(scored_cases),
            "failed_cases": len(errors),
            "base_model_probability": "omitted",
            "holdout_labels_sent_to_agent": False,
            "production_neptune_used": False,
            "memory_actor_prefix": f"eval-{protocol.run_id}",
        },
        "memory_bootstrap": bootstrap_results,
        "natural_half_holdout": protocol.natural_metrics,
        "simuser_ratio_evaluation": ratio_reports,
        "union_case_control_sample": {
            "final_purchase_probability": binary_metrics(all_labels, all_final),
            "agent_likelihood": binary_metrics(all_labels, all_agent),
        },
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_dir / "predictions.jsonl",
        (
            {
                "case_id": case.case_id,
                "label": case.label,
                "oracle_probability": case.oracle_probability,
                "ratio_membership": case.ratio_membership,
                "result": results.get(case.case_id),
                "error": errors.get(case.case_id),
            }
            for case in protocol.cases
        ),
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    natural = report["natural_half_holdout"]
    lines = [
        "# SimUSER-style half-holdout evaluation",
        "",
        "> 실제 값은 독립 labeling session이 생성한 synthetic purchase outcome이며, "
        "실제 넥슨 고객 구매가 아닙니다.",
        "",
        "## Protocol",
        "",
        f"- Users: {report['protocol']['users']}",
        f"- History/holdout: {report['protocol']['history_fraction']:.0%} / "
        f"{1.0 - report['protocol']['history_fraction']:.0%}",
        f"- Memory history: latest "
        f"{report['protocol']['history_impressions_per_user']} impressions per user",
        f"- Live Agent cases: {report['protocol']['successful_cases']} "
        f"(failed {report['protocol']['failed_cases']})",
        "- No learned CVR/base probability",
        "- Holdout answer key was never sent to Runtime",
        "",
        "## Natural half-holdout",
        "",
        "| 모델 | 실제 구매율 | 평균 예측 | 차이(pp) | Brier | LogLoss | AUC | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in natural.items():
        lines.append(
            f"| {name} | {metrics['observed_rate']:.4f} | "
            f"{metrics['mean_prediction']:.4f} | "
            f"{metrics['mean_bias_pp']:+.3f} | {metrics['brier']:.5f} | "
            f"{metrics['log_loss']:.5f} | {metrics['roc_auc']:.4f} | "
            f"{metrics['ece']:.5f} |"
        )
    lines.extend(
        (
            "",
            "## SimUSER ratio evaluation",
            "",
            "| 비율 | 신호 | Accuracy | Precision | Recall | F1 | AUC |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for ratio, signals in report["simuser_ratio_evaluation"].items():
        for name, metrics in signals.items():
            lines.append(
                f"| {ratio} | {name} | {metrics['accuracy']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |"
            )
    lines.extend(
        (
            "",
            "Case-control 비율 표의 평균 확률과 Brier는 자연 구매율 calibration으로 "
            "해석하면 안 됩니다. 자연분포 calibration은 위 half-holdout 표를 사용합니다.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimUSER-style half-history purchase evaluation."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-arn")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--history-fraction", type=float, default=0.5)
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--exclude-users", nargs="*", default=())
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.history_fraction < 1.0:
        raise ValueError("history-fraction must be between zero and one")
    protocol = prepare_protocol(
        args.data_dir,
        selected_users=args.users,
        history_fraction=args.history_fraction,
        history_limit=args.history_limit,
        seed=args.seed,
        excluded_user_ids=args.exclude_users,
    )
    save_protocol(protocol, args.output_dir)
    if args.prepare_only:
        print(json.dumps(protocol.natural_metrics, ensure_ascii=False, indent=2))
        return
    if not args.runtime_arn:
        raise ValueError("--runtime-arn is required unless --prepare-only is used")
    report = run_live_evaluation(
        protocol,
        args.runtime_arn,
        args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
