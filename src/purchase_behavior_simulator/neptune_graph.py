from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .models import KnowledgeGraphEvidence, SimulationRequest
from .scoring import clamp


TARGET_NEIGHBORS_QUERY = """
MATCH (target:Item {itemId: $itemId})-[relation]->(neighbor)
WHERE type(relation) IN ['IN_CATEGORY', 'TARGETS', 'AVAILABLE_IN', 'CONTAINS']
RETURN type(relation) AS relationType, neighbor.nodeId AS neighborId
"""


HISTORY_NEIGHBORS_QUERY = """
MATCH (user:User {userId: $userId})-[interaction]->(source:Item)
WHERE type(interaction) IN [
  'VIEWED', 'CLICK', 'CLICKED', 'START_PURCHASE', 'CONFIRM_PURCHASE',
  'PAYMENT_SUCCESS', 'PURCHASED', 'INSUFFICIENT_CURRENCY',
  'OPEN_TOP_UP', 'TOP_UP_SUCCESS', 'PAYMENT_FAILED', 'CANCEL',
  'CANCEL_TOP_UP', 'BACK', 'BACK_TO_ITEM', 'SKIP', 'DISMISSED',
  'EXIT', 'PURCHASE_NOW', 'TRIED_ON', 'EQUIPPED', 'USED', 'REFUNDED'
]
WITH source, interaction
ORDER BY interaction.timestamp DESC
LIMIT $historyLimit
OPTIONAL MATCH (source)-[relation]->(neighbor)
WHERE type(relation) IN ['IN_CATEGORY', 'TARGETS', 'AVAILABLE_IN', 'CONTAINS']
RETURN source.itemId AS sourceItemId,
       type(interaction) AS interactionType,
       interaction.timestamp AS interactionTimestamp,
       type(relation) AS relationType,
       neighbor.nodeId AS neighborId
"""


@dataclass(frozen=True)
class NeptuneGraphConfig:
    endpoint_url: str
    region_name: str = "us-east-1"
    history_limit: int = 100
    retrieval_limit: int = 5
    recency_half_life_days: float = 45.0
    relation_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "IN_CATEGORY": 0.35,
            "TARGETS": 0.30,
            "AVAILABLE_IN": 0.20,
            "CONTAINS": 0.15,
        }
    )
    interaction_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "VIEWED": 0.08,
            "CLICK": 0.20,
            "CLICKED": 0.20,
            "START_PURCHASE": 0.45,
            "CONFIRM_PURCHASE": 0.70,
            "PAYMENT_SUCCESS": 1.00,
            "TRIED_ON": 0.35,
            "EQUIPPED": 0.55,
            "USED": 0.65,
            "PURCHASED": 1.00,
            "INSUFFICIENT_CURRENCY": 0.35,
            "OPEN_TOP_UP": 0.55,
            "TOP_UP_SUCCESS": 0.80,
            "PAYMENT_FAILED": -0.45,
            "CANCEL": -0.35,
            "CANCEL_TOP_UP": -0.45,
            "BACK": -0.15,
            "BACK_TO_ITEM": -0.10,
            "SKIP": -0.25,
            "DISMISSED": -0.25,
            "EXIT": -0.30,
            "PURCHASE_NOW": 0.55,
            "REFUNDED": -1.00,
        }
    )


class NeptuneGraphEvidenceProvider:
    def __init__(
        self,
        config: NeptuneGraphConfig | None = None,
        client: Any | None = None,
    ) -> None:
        if config is None:
            endpoint = os.environ["NEPTUNE_ENDPOINT"]
            if not endpoint.startswith("http"):
                endpoint = f"https://{endpoint}:8182"
            config = NeptuneGraphConfig(
                endpoint_url=endpoint,
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                history_limit=int(os.getenv("NEPTUNE_HISTORY_LIMIT", "100")),
            )
        self.config = config
        if client is None:
            import boto3

            client = boto3.client(
                "neptunedata",
                endpoint_url=config.endpoint_url,
                region_name=config.region_name,
            )
        self.client = client

    def get_evidence(
        self, request: SimulationRequest
    ) -> KnowledgeGraphEvidence:
        stored_target_rows = self._execute(
            TARGET_NEIGHBORS_QUERY,
            {"itemId": request.item.item_id},
        )
        authoritative_relations = set()
        if request.item.attributes.get("scenario_categories_overridden"):
            authoritative_relations.add("IN_CATEGORY")
        if request.item.components:
            authoritative_relations.add("CONTAINS")
        filtered_stored_rows = tuple(
            row
            for row in stored_target_rows
            if self._text(row.get("relationType")) not in authoritative_relations
        )
        target_rows = self._merge_target_rows(
            filtered_stored_rows,
            self._request_target_rows(request),
        )
        history_rows = self._execute(
            HISTORY_NEIGHBORS_QUERY,
            {
                "userId": request.user.user_id,
                "historyLimit": self.config.history_limit,
            },
        )
        return self.calculate_evidence(
            target_rows=target_rows,
            history_rows=history_rows,
            now=request.context.timestamp,
            target_item_id=request.item.item_id,
        )

    @staticmethod
    def _request_target_rows(
        request: SimulationRequest,
    ) -> tuple[dict[str, str], ...]:
        rows = [
            {"relationType": "IN_CATEGORY", "neighborId": category}
            for category in request.item.categories
        ]
        character = request.item.attributes.get("character")
        event_id = request.item.attributes.get("event_id")
        if character:
            rows.append(
                {"relationType": "TARGETS", "neighborId": str(character)}
            )
        if event_id:
            rows.append(
                {"relationType": "AVAILABLE_IN", "neighborId": str(event_id)}
            )
        rows.extend(
            {
                "relationType": "CONTAINS",
                "neighborId": component.item_id,
            }
            for component in request.item.components
        )
        return tuple(rows)

    @classmethod
    def _merge_target_rows(
        cls,
        *row_groups: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        merged: list[Mapping[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for rows in row_groups:
            for row in rows:
                key = (
                    cls._text(row.get("relationType")),
                    cls._text(row.get("neighborId")),
                )
                if not all(key) or key in seen:
                    continue
                seen.add(key)
                merged.append(row)
        return tuple(merged)

    def _execute(
        self, query: str, parameters: Mapping[str, Any]
    ) -> Sequence[Mapping[str, Any]]:
        response = self.client.execute_open_cypher_query(
            openCypherQuery=query,
            parameters=json.dumps(dict(parameters), separators=(",", ":")),
        )
        return response.get("results", ())

    def calculate_evidence(
        self,
        *,
        target_rows: Sequence[Mapping[str, Any]],
        history_rows: Sequence[Mapping[str, Any]],
        now: datetime,
        target_item_id: str = "",
    ) -> KnowledgeGraphEvidence:
        target_neighbors: dict[str, set[str]] = defaultdict(set)
        for row in target_rows:
            relation = self._text(row.get("relationType"))
            neighbor = self._text(row.get("neighborId"))
            if relation and neighbor:
                target_neighbors[relation].add(neighbor)
        if target_item_id:
            target_neighbors["CONTAINS"].add(target_item_id)

        history: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for row in history_rows:
            source_id = self._text(row.get("sourceItemId"))
            interaction = self._text(row.get("interactionType"))
            timestamp = self._text(row.get("interactionTimestamp"))
            relation = self._text(row.get("relationType"))
            neighbor = self._text(row.get("neighborId"))
            if not source_id or not interaction:
                continue
            key = (source_id, interaction, timestamp)
            source_neighbors = history[key]
            if relation and neighbor:
                source_neighbors[relation].add(neighbor)
            source_neighbors["CONTAINS"].add(source_id)

        if not history or not target_neighbors:
            return KnowledgeGraphEvidence(
                precomputed_affinity=0.5,
                precomputed_confidence=0.0,
            )

        retrieved: list[
            tuple[
                float,
                float,
                str,
                Mapping[str, float],
                dict[str, Any],
            ]
        ] = []

        for (source_id, interaction, timestamp), source_neighbors in history.items():
            interaction_weight = self.config.interaction_weights.get(interaction, 0.0)
            if interaction_weight == 0.0:
                continue
            recency = self._recency(timestamp, now)

            source_score = 0.0
            source_relation_weight = 0.0
            relation_scores: dict[str, float] = {}
            shared_neighbors: dict[str, list[str]] = {}
            for relation, relation_weight in self.config.relation_weights.items():
                target_set = target_neighbors.get(relation, set())
                if not target_set:
                    continue
                source_set = source_neighbors.get(relation, set())
                intersection = target_set.intersection(source_set)
                score = self._pathsim_sets(source_set, target_set)
                source_score += relation_weight * score
                source_relation_weight += relation_weight
                if score > 0.0:
                    relation_scores[relation] = score
                    shared_neighbors[relation] = sorted(intersection)

            if source_relation_weight == 0.0:
                continue
            normalized_source_score = source_score / source_relation_weight
            if normalized_source_score <= 0.0:
                continue

            # SimUSER retrieves behavior on related items with PathSim and lets the
            # LLM interpret the observed interaction. Unrelated history must not
            # become high-confidence negative evidence merely because it is large.
            evidence_weight = (
                abs(interaction_weight) * recency * normalized_source_score
            )
            direction = 1.0 if interaction_weight > 0.0 else -1.0
            preference = clamp(
                0.5 + 0.5 * direction * normalized_source_score
            )

            retrieved.append(
                (
                    evidence_weight,
                    preference,
                    source_id,
                    relation_scores,
                    {
                        "schema": "simuser.kg-evidence.v1",
                        "target_item_id": target_item_id,
                        "source_item_id": source_id,
                        "observed_interaction": interaction,
                        "observed_at": timestamp or None,
                        "polarity": "positive" if interaction_weight > 0.0 else "negative",
                        "pathsim": round(normalized_source_score, 8),
                        "recency_weight": round(recency, 8),
                        "relation_pathsim": {
                            relation.lower(): round(score, 8)
                            for relation, score in relation_scores.items()
                        },
                        "shared_neighbors": {
                            relation.lower(): neighbors
                            for relation, neighbors in shared_neighbors.items()
                        },
                    },
                )
            )

        if not retrieved:
            return KnowledgeGraphEvidence(
                precomputed_affinity=0.5,
                precomputed_confidence=0.0,
            )

        retrieved.sort(key=lambda value: value[0], reverse=True)
        selected = retrieved[: self.config.retrieval_limit]
        total_weight = sum(value[0] for value in selected)
        total_signal = sum(value[0] * value[1] for value in selected)
        affinity = clamp(total_signal / total_weight)
        relevant_items = {value[2] for value in selected}
        covered_relations = {
            relation
            for _, _, _, relation_scores, _ in selected
            for relation in relation_scores
        }
        available_relation_weight = sum(
            weight
            for relation, weight in self.config.relation_weights.items()
            if target_neighbors.get(relation)
        )
        covered_relation_weight = sum(
            self.config.relation_weights[relation]
            for relation in covered_relations
        )
        coverage = clamp(
            covered_relation_weight / max(available_relation_weight, 1e-9)
        )
        support = clamp(total_weight / max(1, len(selected)))
        diversity = clamp(len(relevant_items) / max(1, len(selected)))
        consistency = clamp(abs(2.0 * affinity - 1.0))
        retrieval_quality = clamp(
            support
            * (0.5 + 0.5 * coverage)
            * (0.5 + 0.5 * diversity)
            * (0.5 + 0.5 * consistency)
        )
        relation_signal: dict[str, float] = defaultdict(float)
        relation_total: dict[str, float] = defaultdict(float)
        for event_weight, _, _, relation_scores, payload in selected:
            direction = 1.0 if payload["polarity"] == "positive" else -1.0
            for relation, score in relation_scores.items():
                relation_weight = event_weight * self.config.relation_weights[relation]
                relation_preference = clamp(0.5 + 0.5 * direction * score)
                relation_signal[relation] += relation_weight * relation_preference
                relation_total[relation] += relation_weight
        meta_path_scores = {
            relation.lower(): clamp(relation_signal[relation] / relation_total[relation])
            for relation in relation_total
            if relation_total[relation] > 0.0
        }
        for relation in target_neighbors:
            meta_path_scores.setdefault(relation.lower(), 0.0)
        evidence_documents = tuple(
            "<simuser-kg-evidence>"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "</simuser-kg-evidence>"
            for _, _, _, _, payload in selected
        )
        return KnowledgeGraphEvidence(
            precomputed_affinity=affinity,
            precomputed_confidence=retrieval_quality,
            meta_path_scores=meta_path_scores,
            retrieved_evidence=evidence_documents,
            retrieval_support=support,
            retrieval_coverage=coverage,
            retrieval_consistency=consistency,
        )

    def _recency(self, timestamp: str, now: datetime) -> float:
        if not timestamp:
            return 1.0
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
            return math.exp(
                -math.log(2.0)
                * age_days
                / max(self.config.recency_half_life_days, 1e-6)
            )
        except ValueError:
            return 1.0

    @staticmethod
    def _pathsim_sets(source: set[str], target: set[str]) -> float:
        denominator = len(source) + len(target)
        if denominator == 0:
            return 0.5
        return clamp(2.0 * len(source.intersection(target)) / denominator)

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            for key in ("value", "@value"):
                if key in value:
                    return str(value[key])
        return str(value)
