from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Sequence

from .episodic_memory import (
    behavior_events_from_text,
    merge_behavior_events,
    observed_transitions_from_text,
    observed_at_from_record,
    record_namespace,
    record_relevance,
    record_text,
    rerank_memory_documents,
    serialize_observation,
    serialize_observed_transitions,
    serialize_reflection,
    transitions_from_observation,
)
from .models import (
    EpisodicMemoryEvidence,
    Item,
    MemoryDocument,
    ObservationBatch,
    ObservationReceipt,
    SimulationRequest,
    SimulationResult,
    ExposureContext,
)


LOGGER = logging.getLogger(__name__)


class AgentCoreMemoryProvider:
    def __init__(
        self,
        memory_id: str | None = None,
        region_name: str | None = None,
        episode_namespace_path: str = "/episodes/{actorId}",
        session_manager: Any | None = None,
        data_plane_client: Any | None = None,
        control_plane_client: Any | None = None,
        episodic_strategy_id: str | None = None,
        transition_strategy_id: str | None = None,
        transition_namespace_path: str = "/users/{actorId}/observed-transitions",
    ) -> None:
        self.memory_id = (
            memory_id
            or os.getenv("AGENTCORE_MEMORY_ID")
            or os.environ["MEMORY_PURCHASEBEHAVIORSIMULATORMEMORY_ID"]
        )
        if session_manager is None:
            from bedrock_agentcore.memory import MemorySessionManager

            session_manager = MemorySessionManager(
                memory_id=self.memory_id,
                region_name=region_name or os.getenv("AWS_REGION", "us-east-1"),
            )
        if data_plane_client is None:
            import boto3

            data_plane_client = boto3.client(
                "bedrock-agentcore",
                region_name=region_name or os.getenv("AWS_REGION", "us-east-1"),
            )
        self.session_manager = session_manager
        self.data_plane_client = data_plane_client
        self.control_plane_client = control_plane_client
        self.episode_namespace_path = episode_namespace_path
        self.episodic_strategy_id = episodic_strategy_id or os.getenv(
            "PURCHASE_BEHAVIOR_EPISODIC_STRATEGY_ID"
        )
        self.transition_strategy_id = transition_strategy_id or os.getenv(
            "PURCHASE_BEHAVIOR_TRANSITION_STRATEGY_ID"
        )
        self.transition_namespace_path = transition_namespace_path

    def retrieve(
        self,
        user_id: str,
        queries: Sequence[str],
        item: Item,
        now: datetime,
        context: ExposureContext | None = None,
        session_id: str | None = None,
    ) -> EpisodicMemoryEvidence:
        documents: list[MemoryDocument] = []
        interactions = []
        transitions = []
        retrieval_session = self.session_manager.create_memory_session(
            actor_id=user_id,
            session_id=session_id or "purchase-behavior-retrieval",
        )

        if session_id:
            current_documents = self._current_session_documents(retrieval_session)
            documents.extend(current_documents)
            for document in current_documents:
                interactions.extend(behavior_events_from_text(document.content))
                transitions.extend(
                    observed_transitions_from_text(document.content)
                )

        episode_path = self.episode_namespace_path.format(actorId=user_id)
        try:
            recent_records = retrieval_session.list_long_term_memory_records(
                namespace_path=episode_path,
                max_results=20,
            )
        except Exception as exc:
            LOGGER.warning(
                "AgentCore recent episodic listing failed for %s: %s",
                episode_path,
                exc,
            )
            recent_records = ()
        for record in recent_records:
            document = self._document_from_record(
                record,
                source_query="recent actor episodes",
                kind="long_term_recent",
            )
            if document is not None:
                documents.append(document)
                interactions.extend(
                    behavior_events_from_text(document.content)
                )
                transitions.extend(
                    observed_transitions_from_text(document.content)
                )

        for query in queries:
            try:
                records = retrieval_session.search_long_term_memories(
                    query=query,
                    namespace_path=episode_path,
                    top_k=2,
                    max_results=6,
                )
            except Exception as exc:
                LOGGER.warning(
                    "AgentCore episodic search failed for %s: %s", episode_path, exc
                )
                records = ()
            for record in records:
                document = self._document_from_record(
                    record,
                    source_query=query,
                    kind="long_term_episode",
                )
                if document is None:
                    continue
                documents.append(document)
                interactions.extend(behavior_events_from_text(document.content))
                transitions.extend(
                    observed_transitions_from_text(document.content)
                )

        transition_path = self.transition_namespace_path.format(actorId=user_id)
        transition_query = self._transition_query(item=item, context=context)
        try:
            transition_records = retrieval_session.search_long_term_memories(
                query=transition_query,
                namespace_path=transition_path,
                top_k=3,
                max_results=8,
            )
        except Exception as exc:
            LOGGER.warning(
                "AgentCore observed transition search failed for %s: %s",
                transition_path,
                exc,
            )
            transition_records = ()
        for record in transition_records:
            document = self._document_from_record(
                record,
                source_query=transition_query,
                kind="observed_transition",
            )
            if document is None:
                continue
            documents.append(document)
            transitions.extend(observed_transitions_from_text(document.content))

        ranked = rerank_memory_documents(
            documents,
            item=item,
            now=now,
            context=context,
        )
        ranked_interactions = [
            event
            for document in ranked
            for event in behavior_events_from_text(document.content)
        ]
        return EpisodicMemoryEvidence(
            queries=tuple(queries),
            documents=ranked,
            interactions=merge_behavior_events(interactions, ranked_interactions),
            transitions=tuple(transitions),
        )

    def record_observations(
        self,
        batch: ObservationBatch,
        reflection: str,
    ) -> ObservationReceipt:
        from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

        session = self.session_manager.create_memory_session(
            actor_id=batch.user_id,
            session_id=batch.session_id,
        )
        messages = [
            ConversationalMessage(
                serialize_observation(batch),
                MessageRole.USER,
            )
        ]
        if reflection.strip():
            messages.append(
                ConversationalMessage(
                    serialize_reflection(reflection),
                    MessageRole.ASSISTANT,
                )
            )
        session.add_turns(
            messages=messages,
            event_timestamp=max(event.timestamp for event in batch.events),
        )
        long_term_record_count = self._write_long_term_records(batch, reflection)
        return ObservationReceipt(
            user_id=batch.user_id,
            session_id=batch.session_id,
            event_count=len(batch.events),
            long_term_record_count=long_term_record_count,
            source=batch.source,
            reflection=reflection,
        )

    def record_prediction(
        self,
        request: SimulationRequest,
        result: SimulationResult,
    ) -> None:
        # Deliberately inert: model outputs must never become evidence for later calls.
        return None

    @staticmethod
    def _current_session_documents(session: Any) -> tuple[MemoryDocument, ...]:
        try:
            events = session.list_events(max_results=100, include_payload=True)
        except Exception as exc:
            LOGGER.warning("AgentCore short-term memory search failed: %s", exc)
            return ()

        documents: list[MemoryDocument] = []
        for event in events:
            timestamp = observed_at_from_record(event)
            payload = _get(event, "payload") or ()
            for value in payload:
                conversational = (
                    value.get("conversational") if isinstance(value, dict) else None
                )
                if not isinstance(conversational, dict):
                    continue
                content = conversational.get("content")
                text = (
                    content.get("text")
                    if isinstance(content, dict)
                    else content or conversational.get("text")
                )
                if not isinstance(text, str) or not text.strip():
                    continue
                documents.append(
                    MemoryDocument(
                        content=text,
                        relevance=1.0,
                        namespace="short-term",
                        source_query="current session",
                        observed_at=timestamp,
                        kind="current_session",
                    )
                )
        return tuple(documents)

    @staticmethod
    def _document_from_record(
        record: Any,
        *,
        source_query: str,
        kind: str,
    ) -> MemoryDocument | None:
        text = record_text(record)
        if not text:
            return None
        return MemoryDocument(
            content=text,
            relevance=record_relevance(record),
            namespace=record_namespace(record),
            source_query=source_query,
            observed_at=observed_at_from_record(record),
            kind=kind,
        )

    def _write_long_term_records(
        self,
        batch: ObservationBatch,
        reflection: str,
    ) -> int:
        self._ensure_strategy_ids()
        timestamp = max(event.timestamp for event in batch.events)
        observation = serialize_observation(batch)
        records = [
            self._long_term_record(
                content=observation,
                namespace=f"/episodes/{batch.user_id}/{batch.session_id}",
                timestamp=timestamp,
                prefix="observation",
            )
        ]
        transitions = transitions_from_observation(batch)
        if transitions:
            records.append(
                self._long_term_record(
                    content=serialize_observed_transitions(
                        transitions,
                        source=batch.source,
                    ),
                    namespace=self.transition_namespace_path.format(
                        actorId=batch.user_id
                    ),
                    timestamp=timestamp,
                    prefix="observed-transition",
                    memory_strategy_id=self.transition_strategy_id,
                )
            )
        if reflection.strip():
            records.append(
                self._long_term_record(
                    content=serialize_reflection(reflection),
                    namespace=f"/episodes/{batch.user_id}",
                    timestamp=timestamp,
                    prefix="reflection",
                    memory_strategy_id=self.episodic_strategy_id,
                )
            )
        response = self.data_plane_client.batch_create_memory_records(
            memoryId=self.memory_id,
            records=records,
            clientToken=self._digest(
                f"{batch.user_id}:{batch.session_id}:{observation}:{reflection}"
            ),
        )
        failed = response.get("failedRecords", ())
        if failed:
            raise RuntimeError(f"failed to create long-term episodic records: {failed}")
        return len(response.get("successfulRecords", ()))

    def _ensure_strategy_ids(self) -> None:
        if self.episodic_strategy_id and self.transition_strategy_id:
            return
        if self.control_plane_client is None:
            raise RuntimeError(
                "Memory strategy IDs are not configured. Deploy with "
                "scripts/deploy_agentcore.py so the generated strategy IDs "
                "are injected into the Runtime environment."
            )
        response = self.control_plane_client.get_memory(
            memoryId=self.memory_id,
        )
        strategies = response.get("memory", {}).get("strategies", ())
        for strategy in strategies:
            strategy_type = str(strategy.get("type", "")).upper()
            if strategy_type == "EPISODIC":
                self.episodic_strategy_id = str(strategy["strategyId"])
            elif strategy_type == "SEMANTIC":
                self.transition_strategy_id = str(strategy["strategyId"])
        if not self.episodic_strategy_id or not self.transition_strategy_id:
            raise RuntimeError(
                "Memory must expose active EPISODIC and SEMANTIC strategies"
            )

    def _long_term_record(
        self,
        *,
        content: str,
        namespace: str,
        timestamp: datetime,
        prefix: str,
        memory_strategy_id: str | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "requestIdentifier": f"{prefix}-{self._digest(f'{namespace}:{content}')[:24]}",
            "namespaces": [namespace],
            "content": {"text": content},
            "timestamp": timestamp,
        }
        if memory_strategy_id is None and namespace.startswith("/episodes/"):
            memory_strategy_id = self.episodic_strategy_id
        if memory_strategy_id:
            record["memoryStrategyId"] = memory_strategy_id
        return record

    @staticmethod
    def _transition_query(
        *,
        item: Item,
        context: ExposureContext | None,
    ) -> str:
        categories = ", ".join(item.categories) or "unknown"
        surface = context.surface if context else "unknown"
        fatigue = (
            f"{context.session_fatigue:.2f}" if context is not None else "unknown"
        )
        price_ratio = "unknown"
        if context and context.budget_reference:
            price_ratio = f"{item.price / context.budget_reference:.2f}"
        return (
            f"observed transition surface={surface}, categories={categories}, "
            f"price_budget_ratio={price_ratio}, fatigue={fatigue}, "
            "actual action and next outcome"
        )

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key)
    return getattr(value, key, None)
