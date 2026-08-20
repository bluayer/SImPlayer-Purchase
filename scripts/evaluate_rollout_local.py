from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from purchase_behavior_simulator.episodic_memory import (
    rerank_memory_documents,
    serialize_observation,
    serialize_observed_transitions,
    serialize_reflection,
    transitions_from_observation,
)
from purchase_behavior_simulator.episodic_reasoning import (
    DeterministicReflectionProvider,
    DeterministicSelfAskQueryPlanner,
)
from purchase_behavior_simulator.evaluation import (
    expected_calibration_error,
    read_jsonl,
)
from purchase_behavior_simulator.hybrid_assessment import (
    HybridAssessmentProvider,
)
from purchase_behavior_simulator.evaluation_summary import (
    summarize_simulation_run,
)
from purchase_behavior_simulator.models import (
    EpisodicMemoryEvidence,
    Item,
    MemoryDocument,
    ObservationBatch,
    ObservationReceipt,
    SimulationRequest,
)
from purchase_behavior_simulator.scoring import (
    BehaviorSimulationScorer,
    ScoringConfig,
    combine_fused_and_rollout_probability,
)
from purchase_behavior_simulator.service import BehaviorSimulationService
from purchase_behavior_simulator.strands_assessment import (
    StrandsAssessmentProvider,
)


ROLLOUT_OUTPUT_BLEND = 0.35
FUSION_LOGIT_SHRINK = 0.5
FUSION_PRIOR_ANCHOR = 0.12


def rescore_result(result: Mapping[str, Any]) -> dict[str, Any]:
    rescored_result = dict(result)
    components = dict(result.get("components", {}))
    if (
        components.get("raw_fusion") is None
        or components.get("rollout") is None
    ):
        return rescored_result
    pre_shrink, rescored = combine_fused_and_rollout_probability(
        float(components["raw_fusion"]),
        float(components["rollout"]),
        rollout_output_blend=ROLLOUT_OUTPUT_BLEND,
        fusion_logit_shrink=FUSION_LOGIT_SHRINK,
        fusion_prior_anchor=FUSION_PRIOR_ANCHOR,
    )
    rescored_result["probability"] = round(rescored, 6)
    rescored_result["components"] = {
        **components,
        "rollout_output_blend": ROLLOUT_OUTPUT_BLEND,
        "pre_shrink": round(pre_shrink, 6),
        "raw": round(rescored, 6),
    }
    return rescored_result


class SnapshotMemoryProvider:
    def __init__(self, batches: Mapping[str, ObservationBatch]) -> None:
        self.batches = batches
        self.reflection_provider = DeterministicReflectionProvider()

    def retrieve(
        self,
        user_id,
        queries,
        item,
        now,
        context=None,
        session_id=None,
    ):
        batch = self.batches[user_id]
        transitions = transitions_from_observation(batch)
        reflection = self.reflection_provider.reflect(batch)
        documents = [
            MemoryDocument(
                content=serialize_observation(batch),
                relevance=1.0,
                namespace="snapshot-session",
                source_query="current session",
                observed_at=max(event.timestamp for event in batch.events),
                kind="current_session",
            ),
            MemoryDocument(
                content=serialize_observed_transitions(
                    transitions,
                    source=batch.source,
                ),
                relevance=1.0,
                namespace=f"/users/{user_id}/observed-transitions",
                source_query="observed transition snapshot",
                observed_at=max(
                    transition.timestamp for transition in transitions
                ),
                kind="observed_transition",
            ),
            MemoryDocument(
                content=serialize_reflection(reflection),
                relevance=0.5,
                namespace=f"/episodes/{user_id}",
                source_query="deterministic observation reflection",
                observed_at=max(event.timestamp for event in batch.events),
                kind="reflection",
            ),
        ]
        ranked = rerank_memory_documents(
            documents,
            item=item,
            now=now,
            context=context,
        )
        return EpisodicMemoryEvidence(
            queries=tuple(queries),
            documents=ranked,
            interactions=batch.events,
            transitions=transitions,
        )

    def record_observations(
        self,
        batch: ObservationBatch,
        reflection: str,
    ) -> ObservationReceipt:
        raise RuntimeError("snapshot evaluator is read-only")


def read_case_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def read_resume_rows(output_dir: Path) -> list[dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in (
        output_dir / "predictions.jsonl",
        output_dir / "predictions.partial.jsonl",
    ):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            indexed[str(row["case_id"])] = dict(row)
    return list(indexed.values())


def sanitize_legacy_observation(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    sanitized = dict(payload)
    sanitized["events"] = [
        dict(event)
        for event in payload.get("events", ())
        if str(event.get("event_type", "")).lower() != "add_to_cart"
    ]
    return sanitized


def oracle_calibration(
    predictions: Sequence[float],
    oracle: Sequence[float],
    bins: int = 10,
) -> dict[str, Any]:
    if len(predictions) != len(oracle):
        raise ValueError("predictions and oracle must have the same length")
    ordered = sorted(
        range(len(predictions)),
        key=lambda index: (predictions[index], index),
    )
    groups = [
        ordered[start : start + max(1, len(ordered) // bins)]
        for start in range(0, len(ordered), max(1, len(ordered) // bins))
    ]
    if len(groups) > bins:
        groups[-2].extend(groups[-1])
        groups.pop()
    rows = []
    weighted_error = 0.0
    for group in groups:
        mean_prediction = statistics.fmean(predictions[index] for index in group)
        mean_oracle = statistics.fmean(oracle[index] for index in group)
        gap = mean_prediction - mean_oracle
        weighted_error += len(group) / max(1, len(predictions)) * abs(gap)
        rows.append(
            {
                "count": len(group),
                "mean_prediction": round(mean_prediction, 8),
                "mean_oracle": round(mean_oracle, 8),
                "gap_pp": round(gap * 100.0, 6),
            }
        )
    return {
        "mae": round(
            statistics.fmean(
                abs(prediction - target)
                for prediction, target in zip(predictions, oracle)
            ),
            8,
        ),
        "decile_calibration_error": round(weighted_error, 8),
        "mean_bias_pp": round(
            (
                statistics.fmean(predictions)
                - statistics.fmean(oracle)
            )
            * 100.0,
            6,
        ),
        "top_decile_bias_pp": rows[-1]["gap_pp"] if rows else 0.0,
        "deciles": rows,
    }


def build_service(
    *,
    model_id: str,
    memory_provider: SnapshotMemoryProvider,
    trace_events: list[dict[str, Any]],
) -> BehaviorSimulationService:
    assessment_provider = HybridAssessmentProvider(
        probability_provider=StrandsAssessmentProvider(
            model_id=model_id,
            include_failure_details=True,
            trace_events=trace_events,
            mode="probability",
        ),
        rollout_provider=StrandsAssessmentProvider(
            model_id=model_id,
            include_failure_details=True,
            trace_events=trace_events,
            mode="actions",
        ),
        trace_events=trace_events,
    )
    return BehaviorSimulationService(
        scorer=BehaviorSimulationScorer(
            config=ScoringConfig(
                agent_logit_weight=0.5,
                agent_logit_reference="base",
                kg_logit_weight=0.0,
                rollout_logit_weight=0.1,
                rollout_logit_reference="base",
                rollout_output_blend=ROLLOUT_OUTPUT_BLEND,
                fusion_logit_shrink=FUSION_LOGIT_SHRINK,
                fusion_prior_anchor=FUSION_PRIOR_ANCHOR,
                model_version="purchase-behavior-simulator-prototype",
            )
        ),
        assessment_provider=assessment_provider,
        memory_provider=memory_provider,
        query_planner=DeterministicSelfAskQueryPlanner(),
        reflection_provider=DeterministicReflectionProvider(),
        trace_events=trace_events,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-id", default="openai.gpt-5.6-sol")
    args = parser.parse_args(argv)

    selected_ids = read_case_ids(args.case_ids_file)
    blind_rows = list(read_jsonl(args.protocol_dir / "blind_cases.jsonl"))
    if selected_ids is not None:
        blind_rows = [
            row for row in blind_rows if str(row["case_id"]) in selected_ids
        ]
    if args.limit is not None:
        blind_rows = blind_rows[: args.limit]
    if not blind_rows:
        raise ValueError("no evaluation cases selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "predictions.partial.jsonl"
    checkpoint_rows = read_resume_rows(args.output_dir)
    indexed_by_case = {
        str(row["case_id"]): row for row in checkpoint_rows
    }
    selected_case_ids = {str(row["case_id"]) for row in blind_rows}
    unexpected_checkpoint_ids = (
        set(indexed_by_case).difference(selected_case_ids)
    )
    if unexpected_checkpoint_ids:
        raise ValueError(
            "checkpoint contains cases outside the selected protocol: "
            + ", ".join(sorted(unexpected_checkpoint_ids)[:5])
        )
    pending_blind_rows = [
        row
        for row in blind_rows
        if str(row["case_id"]) not in indexed_by_case
    ]
    if indexed_by_case:
        print(
            json.dumps(
                {
                    "resumed": len(indexed_by_case),
                    "remaining": len(pending_blind_rows),
                    "total": len(blind_rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    batches = {
        str(row["observation"]["user_id"]): ObservationBatch.from_dict(
            sanitize_legacy_observation(row["observation"])
        )
        for row in read_jsonl(args.protocol_dir / "bootstrap.jsonl")
    }
    memory_provider = SnapshotMemoryProvider(batches)
    prediction_rows: list[dict[str, Any]] = []

    def evaluate(blind: Mapping[str, Any]) -> dict[str, Any]:
        trace_events: list[dict[str, Any]] = []
        started = time.monotonic()
        result = None
        error = None
        try:
            request = SimulationRequest.from_dict(
                blind["payload"]["request"]
            )
            result = build_service(
                model_id=args.model_id,
                memory_provider=memory_provider,
                trace_events=trace_events,
            ).evaluate_snapshot(request).to_dict()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "case_id": str(blind["case_id"]),
            "actor_id": str(blind["actor_id"]),
            "result": result,
            "error": error,
            "latency_seconds": round(time.monotonic() - started, 6),
            "trace": {
                "trace_kind": "observable_structured",
                "raw_chain_of_thought_captured": False,
                "model_id": args.model_id,
                "simulator": "purchase-behavior-simulator-prototype",
                "events": trace_events,
            },
        }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(evaluate, blind): str(blind["case_id"])
            for blind in pending_blind_rows
        }
        for future in as_completed(futures):
            row = future.result()
            indexed_by_case[futures[future]] = row
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(
                json.dumps(
                    {
                        "completed": len(indexed_by_case),
                        "total": len(blind_rows),
                        "case_id": row["case_id"],
                        "error": row["error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        prediction_rows = [
            indexed_by_case[str(blind["case_id"])]
            for blind in blind_rows
        ]

    answers = {
        str(row["case_id"]): row
        for row in read_jsonl(args.protocol_dir / "answer_key.jsonl")
    }
    joined = []
    for row in prediction_rows:
        answer = answers[row["case_id"]]
        result = row.get("result")
        if result is not None:
            result = rescore_result(result)
            row = {**row, "result": result}
            for event in row.get("trace", {}).get("events", ()):
                if event.get("stage") == "scoring":
                    event["components"] = dict(result["components"])
                    event["final_probability"] = result["probability"]
        joined.append(
            {
                **row,
                "model_id": args.model_id,
                "original_user_id": answer["original_user_id"],
                "item_id": answer["item_id"],
                "label": answer["label"],
                "oracle_probability": answer.get("oracle_probability"),
                "ratio_membership": answer["ratio_membership"],
            }
        )

    successful = [row for row in joined if row["result"] is not None]
    final = [float(row["result"]["probability"]) for row in successful]
    rollout = [
        float(
            row["result"]["components"].get(
                "rollout",
                row["result"]["probability"],
            )
        )
        for row in successful
    ]
    oracle = [
        float(row["oracle_probability"])
        for row in successful
        if row.get("oracle_probability") is not None
    ]
    labels = [int(row["label"]) for row in successful]
    label_ece, label_calibration = expected_calibration_error(labels, final)
    report = {
        "protocol": {
            "model_id": args.model_id,
            "simulator": "purchase-behavior-simulator-prototype",
            "requested_cases": len(joined),
            "successful_cases": len(successful),
            "answer_key_sent_to_model": False,
            "counterfactuals_written_to_memory": False,
            "rollout_weight": 0.1,
            "rollout_output_blend": ROLLOUT_OUTPUT_BLEND,
            "checkpoint_results_rescored": True,
        },
        "summary": summarize_simulation_run(joined),
        "oracle_calibration": (
            {
                "available": True,
                "rollout": oracle_calibration(rollout, oracle),
                "final": oracle_calibration(final, oracle),
            }
            if len(oracle) == len(successful) and oracle
            else {
                "available": False,
                "reason": (
                    "answer key does not contain inference-isolated "
                    "oracle probabilities"
                ),
            }
        ),
        "label_calibration": {
            "ece": label_ece,
            "bins": label_calibration,
        },
    }
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in joined
        ),
        encoding="utf-8",
    )
    checkpoint_path.unlink(missing_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
