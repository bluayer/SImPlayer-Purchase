from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

from .evaluation import binary_metrics


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(
        range(len(values)),
        key=lambda index: (float(values[index]), index),
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start
        while (
            end + 1 < len(ordered)
            and float(values[ordered[end + 1]])
            == float(values[ordered[start]])
        ):
            end += 1
        average_rank = (start + end + 2) / 2.0
        for offset in range(start, end + 1):
            ranks[ordered[offset]] = average_rank
        start = end + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("values must have the same length")
    if not left:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_variance = sum(
        (value - left_mean) ** 2 for value in left
    )
    right_variance = sum(
        (value - right_mean) ** 2 for value in right
    )
    denominator = (left_variance * right_variance) ** 0.5
    return numerator / denominator if denominator > 0.0 else 0.0


def continuous_target_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    top_fraction: float = 0.10,
) -> dict[str, Any]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        return {
            "count": 0,
            "mae": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "top_fraction": top_fraction,
            "top_count": 0,
            "top_overlap": 0,
            "random_expected_overlap": 0.0,
        }
    count = len(predictions)
    top_count = max(1, round(count * top_fraction))
    predicted_top = set(
        sorted(
            range(count),
            key=lambda index: (-float(predictions[index]), index),
        )[:top_count]
    )
    target_top = set(
        sorted(
            range(count),
            key=lambda index: (-float(targets[index]), index),
        )[:top_count]
    )
    return {
        "count": count,
        "mae": round(
            statistics.fmean(
                abs(float(prediction) - float(target))
                for prediction, target in zip(predictions, targets)
            ),
            8,
        ),
        "pearson": round(_pearson(predictions, targets), 8),
        "spearman": round(
            _pearson(
                _average_ranks(predictions),
                _average_ranks(targets),
            ),
            8,
        ),
        "top_fraction": top_fraction,
        "top_count": top_count,
        "top_overlap": len(predicted_top.intersection(target_top)),
        "random_expected_overlap": round(
            top_count * top_count / count,
            4,
        ),
    }


def is_neutral_fallback(result: Mapping[str, Any]) -> bool:
    components = result.get("components", {})
    reasons = [str(value).lower() for value in result.get("reasons", [])]
    return (
        float(components.get("agent_confidence", 1.0)) == 0.0
        and any(
            "fallback" in reason or "unavailable" in reason
            for reason in reasons
        )
    )


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            round((len(ordered) - 1) * quantile),
        ),
    )
    return ordered[index]


def prevalence_matched_top_k_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
) -> dict[str, Any]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positive_count = sum(int(label) for label in labels)
    ranked = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )
    selected = set(ranked[:positive_count])
    predictions = [
        1 if index in selected else 0 for index in range(len(labels))
    ]
    tp = sum(
        prediction == 1 and int(label) == 1
        for prediction, label in zip(predictions, labels)
    )
    fp = sum(
        prediction == 1 and int(label) == 0
        for prediction, label in zip(predictions, labels)
    )
    tn = sum(
        prediction == 0 and int(label) == 0
        for prediction, label in zip(predictions, labels)
    )
    fn = sum(
        prediction == 0 and int(label) == 1
        for prediction, label in zip(predictions, labels)
    )
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "predicted_positive_count": positive_count,
        "accuracy": round((tp + tn) / max(1, len(labels)), 8),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1": round(f1, 8),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }


def summarize_simulation_run(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("result") is not None]
    labels = [int(row["label"]) for row in successful]
    final_probabilities = [
        float(row["result"]["probability"]) for row in successful
    ]
    agent_likelihoods = [
        float(
            row["result"]["components"].get(
                "agent",
                row["result"]["probability"],
            )
        )
        for row in successful
    ]
    agent_ranking_scores = [
        float(
            row["result"]["components"].get(
                "agent_ranking_score",
                row["result"]["components"].get(
                    "agent",
                    row["result"]["probability"],
                ),
            )
        )
        for row in successful
    ]
    rollout_probabilities = [
        float(
            row["result"]["components"].get(
                "rollout",
                row["result"]["probability"],
            )
        )
        for row in successful
    ]
    commitment_strengths = [
        float(
            row["result"]["components"].get(
                "decision_commitment_strength",
                0.0,
            )
        )
        for row in successful
    ]
    oracle_probabilities = [
        float(row["oracle_probability"])
        for row in successful
        if row.get("oracle_probability") is not None
    ]
    latencies = [
        float(row["latency_seconds"])
        for row in successful
        if row.get("latency_seconds") is not None
    ]
    fallbacks = sum(
        is_neutral_fallback(row["result"]) for row in successful
    )
    trace_events = [
        event
        for row in rows
        for event in (row.get("trace") or {}).get("events", [])
    ]
    usage = [
        event.get("metrics", {}).get("usage", {})
        for event in trace_events
        if event.get("metrics")
    ]
    case_durations = [
        float(row["trace"]["duration_seconds"])
        for row in rows
        if (row.get("trace") or {}).get("duration_seconds") is not None
    ]
    summary = {
        "requested_cases": len(rows),
        "successful_cases": len(successful),
        "failed_cases": len(rows) - len(successful),
        "neutral_fallbacks": fallbacks,
        "schema_success_rate": round(
            (len(successful) - fallbacks) / max(1, len(rows)),
            8,
        ),
        "latency_seconds": {
            "mean": (
                round(statistics.fmean(latencies), 4)
                if latencies
                else None
            ),
            "p50": (
                round(percentile(latencies, 0.5), 4)
                if latencies
                else None
            ),
            "p95": (
                round(percentile(latencies, 0.95), 4)
                if latencies
                else None
            ),
            "max": round(max(latencies), 4) if latencies else None,
        },
        "final_purchase_probability": binary_metrics(
            labels,
            final_probabilities,
        ),
        "agent_likelihood": binary_metrics(labels, agent_likelihoods),
        "agent_ranking_score": binary_metrics(
            labels,
            agent_ranking_scores,
        ),
        "prevalence_matched_top_k": {
            "final_purchase_probability": prevalence_matched_top_k_metrics(
                labels,
                final_probabilities,
            ),
            "agent_likelihood": prevalence_matched_top_k_metrics(
                labels,
                agent_likelihoods,
            ),
            "agent_ranking_score": prevalence_matched_top_k_metrics(
                labels,
                agent_ranking_scores,
            ),
        },
        "observable_trace": {
            "raw_chain_of_thought_captured": False,
            "case_trace_count": sum(bool(row.get("trace")) for row in rows),
            "self_ask_calls": sum(
                event.get("stage") == "self_ask" for event in trace_events
            ),
            "self_ask_fallbacks": sum(
                event.get("stage") == "self_ask"
                and bool(event.get("fallback"))
                for event in trace_events
            ),
            "assessment_rounds": sum(
                event.get("stage")
                in {"assessment_round", "action_assessment_round"}
                for event in trace_events
            ),
            "assessment_fallbacks": sum(
                event.get("stage")
                in {"assessment_round", "action_assessment_round"}
                and bool(event.get("fallback"))
                for event in trace_events
            ),
            "action_validator_calls": sum(
                event.get("stage") == "action_validator"
                for event in trace_events
            ),
            "action_validator_fallbacks": sum(
                event.get("stage") == "action_validator"
                and bool(event.get("fallback"))
                for event in trace_events
            ),
            "action_validator_adjustments": sum(
                event.get("stage") == "action_validator"
                and bool(event.get("adjusted"))
                for event in trace_events
            ),
            "eligibility_short_circuits": sum(
                event.get("stage") == "eligibility_short_circuit"
                for event in trace_events
            ),
            "input_tokens": sum(
                int(value.get("inputTokens", 0)) for value in usage
            ),
            "output_tokens": sum(
                int(value.get("outputTokens", 0)) for value in usage
            ),
            "case_duration_seconds": {
                "mean": (
                    round(statistics.fmean(case_durations), 4)
                    if case_durations
                    else None
                ),
                "p95": (
                    round(percentile(case_durations, 0.95), 4)
                    if case_durations
                    else None
                ),
            },
        },
    }
    if successful and len(oracle_probabilities) == len(successful):
        summary["inference_isolated_oracle_reference"] = {
            "mean_probability": round(
                statistics.fmean(oracle_probabilities),
                8,
            ),
            "agent_likelihood_mae": round(
                statistics.fmean(
                    abs(prediction - oracle)
                    for prediction, oracle in zip(
                        agent_likelihoods,
                        oracle_probabilities,
                    )
                ),
                8,
            ),
            "final_probability_mae": round(
                statistics.fmean(
                    abs(prediction - oracle)
                    for prediction, oracle in zip(
                        final_probabilities,
                        oracle_probabilities,
                    )
                ),
                8,
            ),
            "signals": {
                "final_purchase_probability": continuous_target_metrics(
                    final_probabilities,
                    oracle_probabilities,
                ),
                "rollout_purchase_probability": continuous_target_metrics(
                    rollout_probabilities,
                    oracle_probabilities,
                ),
                "agent_likelihood": continuous_target_metrics(
                    agent_likelihoods,
                    oracle_probabilities,
                ),
                "commitment_strength": continuous_target_metrics(
                    commitment_strengths,
                    oracle_probabilities,
                ),
            },
        }
    return summary
