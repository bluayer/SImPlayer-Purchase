from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import inspect
from typing import Mapping, Protocol, Sequence

from .action_rollout import (
    ActionGraph,
    DEFAULT_ACTION_GRAPH,
    normalize_action_distributions,
    rollout_purchase_probability,
)
from .evaluation_trace import TraceEvents
from .episodic_memory import (
    memory_document_outcome,
    merge_behavior_events,
    select_contrastive_memory_documents,
)
from .episodic_reasoning import (
    DeterministicReflectionProvider,
    DeterministicSelfAskQueryPlanner,
)
from .models import (
    AgentAssessment,
    BehaviorEvent,
    EpisodicMemoryEvidence,
    Item,
    ObservationBatch,
    ObservationReceipt,
    SimulationRequest,
    SimulationResult,
    KnowledgeGraphEvidence,
    ExposureContext,
)
from .scoring import BehaviorSimulationScorer


class AssessmentProvider(Protocol):
    def assess(
        self,
        request: SimulationRequest,
        memory_evidence: Sequence[str],
    ) -> AgentAssessment: ...


class MemoryProvider(Protocol):
    def retrieve(
        self,
        user_id: str,
        queries: Sequence[str],
        item: Item,
        now: datetime,
        context: ExposureContext | None = None,
        session_id: str | None = None,
    ) -> EpisodicMemoryEvidence: ...

    def record_observations(
        self,
        batch: ObservationBatch,
        reflection: str,
    ) -> ObservationReceipt: ...


class SelfAskQueryPlanner(Protocol):
    def plan(
        self,
        request: SimulationRequest,
        initial_query: str,
    ) -> Sequence[str]: ...


class ReflectionProvider(Protocol):
    def reflect(self, batch: ObservationBatch) -> str: ...


class GraphEvidenceProvider(Protocol):
    def get_evidence(
        self, request: SimulationRequest
    ) -> KnowledgeGraphEvidence: ...


class NeutralAssessmentProvider:
    def assess(
        self,
        request: SimulationRequest,
        memory_evidence: Sequence[str],
    ) -> AgentAssessment:
        return request.agent_assessment or AgentAssessment()


class NoopMemoryProvider:
    def retrieve(
        self,
        user_id: str,
        queries: Sequence[str],
        item: Item,
        now: datetime,
        context: ExposureContext | None = None,
        session_id: str | None = None,
    ) -> EpisodicMemoryEvidence:
        return EpisodicMemoryEvidence(queries=tuple(queries))

    def record_observations(
        self,
        batch: ObservationBatch,
        reflection: str,
    ) -> ObservationReceipt:
        raise RuntimeError("AgentCore Memory is not configured")


class RequestGraphEvidenceProvider:
    def get_evidence(
        self, request: SimulationRequest
    ) -> KnowledgeGraphEvidence:
        return request.kg_evidence


class BehaviorSimulationService:
    def __init__(
        self,
        scorer: BehaviorSimulationScorer | None = None,
        assessment_provider: AssessmentProvider | None = None,
        memory_provider: MemoryProvider | None = None,
        graph_provider: GraphEvidenceProvider | None = None,
        query_planner: SelfAskQueryPlanner | None = None,
        reflection_provider: ReflectionProvider | None = None,
        trace_events: TraceEvents | None = None,
        action_graph: ActionGraph | None = None,
    ) -> None:
        self.scorer = scorer or BehaviorSimulationScorer()
        self.assessment_provider = assessment_provider or NeutralAssessmentProvider()
        self.memory_provider = memory_provider or NoopMemoryProvider()
        self.graph_provider = graph_provider or RequestGraphEvidenceProvider()
        self.query_planner = query_planner or DeterministicSelfAskQueryPlanner()
        self.reflection_provider = (
            reflection_provider or DeterministicReflectionProvider()
        )
        self.trace_events = trace_events
        self.action_graph = action_graph or DEFAULT_ACTION_GRAPH

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        return self._simulate(request, self.graph_provider)

    def evaluate_snapshot(
        self, request: SimulationRequest
    ) -> SimulationResult:
        """Score against caller-supplied, label-free holdout snapshot evidence."""
        return self._simulate(request, RequestGraphEvidenceProvider())

    def _simulate(
        self,
        request: SimulationRequest,
        graph_provider: GraphEvidenceProvider,
    ) -> SimulationResult:
        if not request.item.eligible or request.item.already_owned:
            result = self.scorer.score(
                user=request.user,
                item=request.item,
                context=request.context,
                interactions=request.interactions,
                kg_evidence=request.kg_evidence,
                base_model_probability=request.base_model_probability,
                agent_assessment=AgentAssessment(),
            )
            action_distributions = self.action_graph.ineligible_distributions()
            result = self._attach_behavior_outputs(
                result,
                request=request,
                action_distributions=action_distributions,
                decision_state={},
                intention_distribution={"REJECT": 1.0},
            )
            self._trace(
                {
                    "stage": "eligibility_short_circuit",
                    "reason": result.reasons[0] if result.reasons else "",
                    "action_distributions": action_distributions,
                    "rollout_probability": result.trajectory_purchase_probability,
                    "adjusted": True,
                }
            )
            return result

        initial_query = self._memory_query(request)
        follow_up_queries = self.query_planner.plan(request, initial_query)
        queries = self._dedupe_queries((initial_query, *follow_up_queries))
        self._trace(
            {
                "stage": "retrieval_plan",
                "queries": list(queries),
            }
        )
        memory_evidence = self._retrieve_memory(request, queries)
        assessment_documents = select_contrastive_memory_documents(
            memory_evidence.documents,
            limit=9,
        )
        merged_interactions = self._merge_memory_events(
            request.interactions,
            memory_evidence.interactions,
        )
        assessment_request = replace(
            request,
            interactions=tuple(merged_interactions),
        )
        self._trace(
            {
                "stage": "memory_retrieval",
                "documents": [
                    {
                        "rank": index + 1,
                        "document_id": hashlib.sha256(
                            document.content.encode("utf-8")
                        ).hexdigest()[:16],
                        "relevance": round(float(document.relevance), 8),
                        "namespace": document.namespace,
                        "source_query": document.source_query,
                        "kind": document.kind,
                        "observed_at": (
                            document.observed_at.isoformat()
                            if document.observed_at
                            else None
                        ),
                        "content_preview": " ".join(
                            document.content.split()
                        )[:400],
                        "outcome": memory_document_outcome(document),
                        "selected_for_assessment": (
                            document in assessment_documents
                        ),
                    }
                    for index, document in enumerate(memory_evidence.documents)
                ],
                "interaction_count": len(memory_evidence.interactions),
                "transition_count": len(memory_evidence.transitions),
                "interactions": [
                    {
                        "event_type": event.event_type,
                        "item_id": event.item_id,
                        "categories": list(event.categories),
                        "timestamp": event.timestamp.isoformat(),
                        "rating": event.rating,
                    }
                    for event in memory_evidence.interactions[:30]
                ],
                "transitions": [
                    {
                        "state": transition.state,
                        "action": transition.action,
                        "next_state": transition.next_state,
                        "surface": transition.surface,
                        "price_budget_ratio": transition.price_budget_ratio,
                        "session_fatigue": transition.session_fatigue,
                        "outcome": transition.outcome,
                    }
                    for transition in memory_evidence.transitions[:20]
                ],
            }
        )
        kg_evidence = graph_provider.get_evidence(request)
        self._trace(
            {
                "stage": "kg_retrieval",
                "retrieved_evidence_count": len(kg_evidence.retrieved_evidence),
                "retrieval_quality": kg_evidence.precomputed_confidence,
                "retrieval_support": kg_evidence.retrieval_support,
                "retrieval_coverage": kg_evidence.retrieval_coverage,
                "retrieval_consistency": kg_evidence.retrieval_consistency,
                "meta_path_scores": dict(kg_evidence.meta_path_scores),
            }
        )
        assessment_documents = (
            *(
                document.content
                for document in assessment_documents
            ),
            *kg_evidence.retrieved_evidence[:3],
        )
        assessment = (
            request.agent_assessment
            or self.assessment_provider.assess(
                assessment_request,
                assessment_documents,
            )
        )
        result = self.scorer.score(
            user=request.user,
            item=request.item,
            context=request.context,
            interactions=merged_interactions,
            kg_evidence=kg_evidence,
            base_model_probability=request.base_model_probability,
            agent_assessment=assessment,
        )
        self._trace(
            {
                "stage": "scoring",
                "agent_likelihood": assessment.likelihood,
                "agent_ranking_score": (
                    assessment.relative_preference_score
                    if assessment.relative_preference_score is not None
                    else assessment.likelihood
                ),
                "agent_confidence": assessment.confidence,
                "rollout_confidence": assessment.rollout_confidence,
                "rollout_probability": assessment.rollout_probability,
                "decision_state": dict(assessment.decision_state),
                "intention_distribution": dict(
                    assessment.intention_distribution
                ),
                "counterfactual_checks": dict(
                    assessment.counterfactual_checks
                ),
                "action_distributions": {
                    state: dict(distribution)
                    for state, distribution in assessment.action_distributions.items()
                },
                "validator_adjusted": assessment.validator_adjusted,
                "components": dict(result.components),
                "final_probability": result.probability,
                "final_confidence": result.confidence,
            }
        )
        result = replace(
            result,
            components={
                **result.components,
                "episodic_memory_queries": float(len(memory_evidence.queries)),
                "episodic_memory_records": float(len(memory_evidence.documents)),
                "episodic_memory_events": float(len(memory_evidence.interactions)),
                "observed_memory_transitions": float(
                    len(memory_evidence.transitions)
                ),
            },
        )
        return self._attach_behavior_outputs(
            result,
            request=request,
            action_distributions=assessment.action_distributions,
            decision_state=assessment.decision_state,
            intention_distribution=assessment.intention_distribution,
        )

    def _attach_behavior_outputs(
        self,
        result: SimulationResult,
        *,
        request: SimulationRequest,
        action_distributions: Mapping[str, Mapping[str, float]],
        decision_state: Mapping[str, float],
        intention_distribution: Mapping[str, float],
    ) -> SimulationResult:
        if not action_distributions:
            return replace(
                result,
                scalar_purchase_probability=result.probability,
                action_graph_id=self.action_graph.graph_id,
                action_graph_version=self.action_graph.version,
                decision_state=dict(decision_state),
                intention_distribution=dict(intention_distribution),
            )
        normalized = normalize_action_distributions(
            action_distributions,
            graph=self.action_graph,
        )
        rollout = rollout_purchase_probability(
            normalized,
            surface=request.context.surface,
            graph=self.action_graph,
        )
        likely_trajectories = tuple(
            path.to_dict()
            for path in sorted(
                rollout.paths,
                key=lambda path: path.probability,
                reverse=True,
            )[:5]
        )
        return replace(
            result,
            scalar_purchase_probability=result.probability,
            trajectory_purchase_probability=round(
                rollout.purchase_probability,
                6,
            ),
            action_distributions=normalized,
            likely_trajectories=likely_trajectories,
            decision_state=dict(decision_state),
            intention_distribution=dict(intention_distribution),
            action_graph_id=rollout.graph_id,
            action_graph_version=rollout.graph_version,
        )

    def _trace(self, event: dict[str, object]) -> None:
        if self.trace_events is not None:
            self.trace_events.append(event)

    def _retrieve_memory(
        self,
        request: SimulationRequest,
        queries: Sequence[str],
    ) -> EpisodicMemoryEvidence:
        retrieve = self.memory_provider.retrieve
        parameters = inspect.signature(retrieve).parameters
        arguments = {
            "user_id": request.user.user_id,
            "queries": queries,
            "item": request.item,
            "now": request.context.timestamp,
            "session_id": request.memory_session_id,
        }
        if "context" in parameters:
            arguments["context"] = request.context
        return retrieve(**arguments)

    def record_observations(self, batch: ObservationBatch) -> ObservationReceipt:
        reflection = self.reflection_provider.reflect(batch)
        return self.memory_provider.record_observations(batch, reflection)

    @staticmethod
    def _memory_query(request: SimulationRequest) -> str:
        categories = ", ".join(request.item.categories) or "unknown"
        price_ratio = (
            request.item.price / request.context.budget_reference
            if request.context.budget_reference
            else None
        )
        ratio_text = f"{price_ratio:.2f}" if price_ratio is not None else "unknown"
        return (
            f"상품 {request.item.item_id}, 카테고리 {categories}, "
            f"surface {request.context.surface}, price/budget {ratio_text}, "
            f"fatigue {request.context.session_fatigue:.2f}에서 관측된 "
            "행동 전이와 실제 다음 행동"
        )

    @staticmethod
    def _merge_memory_events(
        interactions: Sequence[BehaviorEvent],
        memory_interactions: Sequence[BehaviorEvent],
    ) -> Sequence[BehaviorEvent]:
        return merge_behavior_events(interactions, memory_interactions)

    @staticmethod
    def _dedupe_queries(queries: Sequence[str]) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for value in queries:
            query = " ".join(str(value).split()).strip()
            normalized = query.lower()
            if not query or normalized in seen:
                continue
            seen.add(normalized)
            result.append(query)
        return tuple(result[:4])
