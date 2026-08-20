from __future__ import annotations

import os
from pathlib import Path

from .action_rollout import DEFAULT_ACTION_GRAPH, load_action_graph
from .agentcore_memory import AgentCoreMemoryProvider
from .episodic_reasoning import (
    DeterministicReflectionProvider,
    DeterministicSelfAskQueryPlanner,
    StrandsReflectionProvider,
    StrandsSelfAskQueryPlanner,
)
from .hybrid_assessment import HybridAssessmentProvider
from .neptune_graph import NeptuneGraphEvidenceProvider
from .scoring import PlattCalibration, BehaviorSimulationScorer, ScoringConfig
from .service import NeutralAssessmentProvider, NoopMemoryProvider, BehaviorSimulationService
from .strands_assessment import StrandsAssessmentProvider


def build_reflection_provider():
    mode = os.getenv(
        "PURCHASE_BEHAVIOR_REFLECTION_MODE",
        "deterministic",
    ).strip().lower()
    if mode == "deterministic":
        return DeterministicReflectionProvider()
    if mode == "llm":
        if not os.getenv("BEDROCK_MODEL_ID"):
            raise RuntimeError(
                "BEDROCK_MODEL_ID is required when LLM reflection is enabled"
            )
        return StrandsReflectionProvider()
    raise ValueError(
        "PURCHASE_BEHAVIOR_REFLECTION_MODE must be deterministic or llm"
    )


def build_service() -> BehaviorSimulationService:
    action_graph_path = os.getenv("PURCHASE_BEHAVIOR_ACTION_GRAPH")
    if action_graph_path:
        configured_path = Path(action_graph_path)
        if not configured_path.exists():
            configured_path = (
                Path(__file__).parent
                / "action_graphs"
                / action_graph_path
            )
        action_graph_path = str(configured_path)
    action_graph = (
        load_action_graph(action_graph_path)
        if action_graph_path
        else DEFAULT_ACTION_GRAPH
    )
    scoring_config = ScoringConfig(
        agent_logit_weight=0.50,
        agent_logit_reference="base",
        kg_logit_weight=float(os.getenv("KG_LOGIT_WEIGHT", "0.0")),
        rollout_logit_weight=float(os.getenv("ROLLOUT_LOGIT_WEIGHT", "0.1")),
        rollout_logit_reference="base",
        rollout_output_blend=float(
            os.getenv("ROLLOUT_OUTPUT_BLEND", "0.35")
        ),
        fusion_logit_shrink=float(os.getenv("FUSION_LOGIT_SHRINK", "0.5")),
        fusion_prior_anchor=float(os.getenv("FUSION_PRIOR_ANCHOR", "0.12")),
        model_version="purchase-behavior-simulator-prototype",
    )
    calibration_version = os.getenv("CALIBRATION_VERSION")
    scorer = BehaviorSimulationScorer(
        config=scoring_config,
        calibration=PlattCalibration(
            slope=float(os.getenv("CALIBRATION_SLOPE", "1.0")),
            intercept=float(os.getenv("CALIBRATION_INTERCEPT", "0.0")),
            version=calibration_version,
        )
    )
    if not os.getenv("BEDROCK_MODEL_ID"):
        assessment_provider = NeutralAssessmentProvider()
    else:
        assessment_provider = HybridAssessmentProvider(
            probability_provider=StrandsAssessmentProvider(
                mode="probability"
            ),
            rollout_provider=StrandsAssessmentProvider(
                mode="actions",
                action_graph=action_graph,
            ),
        )
    query_planner = (
        StrandsSelfAskQueryPlanner()
        if os.getenv("BEDROCK_MODEL_ID")
        else DeterministicSelfAskQueryPlanner()
    )
    reflection_provider = build_reflection_provider()
    memory_id = (
        os.getenv("AGENTCORE_MEMORY_ID")
        or os.getenv("MEMORY_PURCHASEBEHAVIORSIMULATORMEMORY_ID")
    )
    memory_provider = AgentCoreMemoryProvider(memory_id=memory_id) if memory_id else NoopMemoryProvider()
    graph_provider = (
        NeptuneGraphEvidenceProvider() if os.getenv("NEPTUNE_ENDPOINT") else None
    )
    return BehaviorSimulationService(
        scorer=scorer,
        assessment_provider=assessment_provider,
        memory_provider=memory_provider,
        graph_provider=graph_provider,
        query_planner=query_planner,
        reflection_provider=reflection_provider,
        action_graph=action_graph,
    )
