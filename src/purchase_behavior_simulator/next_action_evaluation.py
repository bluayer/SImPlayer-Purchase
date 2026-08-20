from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence


EPSILON = 1e-9
STATE_ACTIONS: Mapping[str, tuple[str, ...]] = {
    "ITEM_EXPOSURE": ("CLICK", "SKIP", "EXIT", "PURCHASE_NOW"),
    "ITEM_DETAIL": ("PURCHASE", "BACK", "EXIT"),
}


def top_action(distribution: Mapping[str, float]) -> str:
    if not distribution:
        raise ValueError("action distribution is empty")
    return max(distribution, key=lambda action: float(distribution[action]))


def multiclass_metrics(
    labels: Sequence[str],
    distributions: Sequence[Mapping[str, float]],
    *,
    actions: Sequence[str],
) -> dict[str, Any]:
    if len(labels) != len(distributions):
        raise ValueError("labels and distributions must have the same length")
    if not labels:
        return {
            "count": 0,
            "accuracy": 0.0,
            "macro_f1_supported": 0.0,
            "macro_f1_all_actions": 0.0,
            "weighted_f1": 0.0,
            "log_loss": 0.0,
            "multiclass_brier": 0.0,
            "per_action": {},
        }

    predictions = [top_action(distribution) for distribution in distributions]
    per_action: dict[str, dict[str, float | int]] = {}
    supported_f1: list[float] = []
    all_f1: list[float] = []
    weighted_f1 = 0.0
    counts = Counter(labels)
    for action in actions:
        true_positive = sum(
            label == action and prediction == action
            for label, prediction in zip(labels, predictions)
        )
        false_positive = sum(
            label != action and prediction == action
            for label, prediction in zip(labels, predictions)
        )
        false_negative = sum(
            label == action and prediction != action
            for label, prediction in zip(labels, predictions)
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = (
            2.0 * precision * recall / max(EPSILON, precision + recall)
            if true_positive + false_positive + false_negative
            else 0.0
        )
        support = counts[action]
        all_f1.append(f1)
        if support:
            supported_f1.append(f1)
            weighted_f1 += support / len(labels) * f1
        per_action[action] = {
            "support": support,
            "predicted": predictions.count(action),
            "precision": round(precision, 8),
            "recall": round(recall, 8),
            "f1": round(f1, 8),
        }

    log_loss = -sum(
        math.log(
            max(
                EPSILON,
                min(1.0, float(distribution.get(label, 0.0))),
            )
        )
        for label, distribution in zip(labels, distributions)
    ) / len(labels)
    brier = sum(
        sum(
            (
                float(distribution.get(action, 0.0))
                - float(label == action)
            )
            ** 2
            for action in actions
        )
        / len(actions)
        for label, distribution in zip(labels, distributions)
    ) / len(labels)
    return {
        "count": len(labels),
        "accuracy": round(
            sum(label == prediction for label, prediction in zip(labels, predictions))
            / len(labels),
            8,
        ),
        "macro_f1_supported": round(
            sum(supported_f1) / max(1, len(supported_f1)), 8
        ),
        "macro_f1_all_actions": round(sum(all_f1) / max(1, len(all_f1)), 8),
        "weighted_f1": round(weighted_f1, 8),
        "log_loss": round(log_loss, 8),
        "multiclass_brier": round(brier, 8),
        "per_action": per_action,
    }


def expected_multiclass_metrics(
    labels: Sequence[str],
    distributions: Sequence[Mapping[str, float]],
    *,
    actions: Sequence[str],
) -> dict[str, Any]:
    if len(labels) != len(distributions):
        raise ValueError("labels and distributions must have the same length")
    if not labels:
        return {
            "count": 0,
            "expected_accuracy": 0.0,
            "macro_f1_supported": 0.0,
            "weighted_f1": 0.0,
            "per_action": {},
        }

    counts = Counter(labels)
    supported_f1: list[float] = []
    weighted_f1 = 0.0
    per_action: dict[str, dict[str, float | int]] = {}
    for action in actions:
        true_positive = sum(
            float(distribution.get(action, 0.0))
            for label, distribution in zip(labels, distributions)
            if label == action
        )
        false_positive = sum(
            float(distribution.get(action, 0.0))
            for label, distribution in zip(labels, distributions)
            if label != action
        )
        false_negative = sum(
            1.0 - float(distribution.get(action, 0.0))
            for label, distribution in zip(labels, distributions)
            if label == action
        )
        precision = true_positive / max(EPSILON, true_positive + false_positive)
        recall = true_positive / max(EPSILON, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(EPSILON, precision + recall)
        support = counts[action]
        if support:
            supported_f1.append(f1)
            weighted_f1 += support / len(labels) * f1
        per_action[action] = {
            "support": support,
            "expected_predicted": round(true_positive + false_positive, 8),
            "observed_count": support,
            "expected_count": round(true_positive + false_positive, 8),
            "count_gap": round(
                true_positive + false_positive - support,
                8,
            ),
            "observed_rate": round(support / len(labels), 8),
            "expected_rate": round(
                (true_positive + false_positive) / len(labels),
                8,
            ),
            "rate_gap_pp": round(
                (
                    (true_positive + false_positive) / len(labels)
                    - support / len(labels)
                )
                * 100.0,
                6,
            ),
            "expected_true_positive": round(true_positive, 8),
            "precision": round(precision, 8),
            "recall": round(recall, 8),
            "f1": round(f1, 8),
        }
    return {
        "count": len(labels),
        "expected_accuracy": round(
            sum(
                float(distribution.get(label, 0.0))
                for label, distribution in zip(labels, distributions)
            )
            / len(labels),
            8,
        ),
        "macro_f1_supported": round(
            sum(supported_f1) / max(1, len(supported_f1)),
            8,
        ),
        "weighted_f1": round(weighted_f1, 8),
        "per_action": per_action,
    }


def expected_binary_metrics(
    labels: Sequence[bool],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    true_positive = sum(
        probability
        for label, probability in zip(labels, probabilities)
        if label
    )
    false_positive = sum(
        probability
        for label, probability in zip(labels, probabilities)
        if not label
    )
    false_negative = sum(
        1.0 - probability
        for label, probability in zip(labels, probabilities)
        if label
    )
    true_negative = sum(
        1.0 - probability
        for label, probability in zip(labels, probabilities)
        if not label
    )
    precision = true_positive / max(EPSILON, true_positive + false_positive)
    recall = true_positive / max(EPSILON, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(EPSILON, precision + recall)
    observed_count = sum(labels)
    expected_count = sum(probabilities)
    brier = sum(
        (probability - float(label)) ** 2
        for label, probability in zip(labels, probabilities)
    ) / max(1, len(labels))
    log_loss = -sum(
        (
            math.log(max(EPSILON, min(1.0, probability)))
            if label
            else math.log(max(EPSILON, min(1.0, 1.0 - probability)))
        )
        for label, probability in zip(labels, probabilities)
    ) / max(1, len(labels))
    return {
        "count": len(labels),
        "positives": observed_count,
        "observed_count": observed_count,
        "expected_count": round(expected_count, 8),
        "count_gap": round(expected_count - observed_count, 8),
        "observed_rate": round(
            observed_count / max(1, len(labels)),
            8,
        ),
        "expected_rate": round(
            expected_count / max(1, len(labels)),
            8,
        ),
        "rate_gap_pp": round(
            (
                expected_count / max(1, len(labels))
                - observed_count / max(1, len(labels))
            )
            * 100.0,
            6,
        ),
        "expected_accuracy": round(
            (true_positive + true_negative) / max(1, len(labels)),
            8,
        ),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1": round(f1, 8),
        "brier": round(brier, 8),
        "log_loss": round(log_loss, 8),
        "expected_confusion_matrix": {
            "tp": round(true_positive, 8),
            "fp": round(false_positive, 8),
            "tn": round(true_negative, 8),
            "fn": round(false_negative, 8),
        },
    }


def monte_carlo_binary_f1(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    *,
    simulations: int = 1000,
    seed: int = 20260819,
) -> dict[str, Any]:
    if simulations < 1:
        raise ValueError("simulations must be positive")
    # Monte Carlo metrics require deterministic replay, not secure randomness.
    rng = random.Random(seed)  # nosec B311
    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    for _ in range(simulations):
        predictions = [rng.random() < probability for probability in probabilities]
        true_positive = sum(
            label and prediction
            for label, prediction in zip(labels, predictions)
        )
        false_positive = sum(
            not label and prediction
            for label, prediction in zip(labels, predictions)
        )
        false_negative = sum(
            label and not prediction
            for label, prediction in zip(labels, predictions)
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(EPSILON, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    def interval(values: Sequence[float]) -> dict[str, float]:
        ordered = sorted(values)
        low = ordered[int(0.025 * (len(ordered) - 1))]
        high = ordered[int(0.975 * (len(ordered) - 1))]
        return {
            "mean": round(statistics.fmean(values), 8),
            "p2_5": round(low, 8),
            "p97_5": round(high, 8),
        }

    return {
        "simulations": simulations,
        "seed": seed,
        "precision": interval(precision_values),
        "recall": interval(recall_values),
        "f1": interval(f1_values),
    }


def evaluate_next_actions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exposure_labels: list[str] = []
    exposure_distributions: list[Mapping[str, float]] = []
    detail_labels: list[str] = []
    detail_distributions: list[Mapping[str, float]] = []
    trajectory_matches: list[bool] = []
    purchase_labels: list[bool] = []
    purchase_predictions: list[bool] = []
    purchase_probabilities: list[float] = []
    trajectory_probabilities: list[float] = []

    for row in rows:
        initial_state = str(row["observed_initial_state"])
        next_action = str(row["observed_next_action"])
        detail_action = row.get("observed_detail_action")
        distributions = row["action_distributions"]

        initial_distribution = distributions.get(initial_state, {})
        initial_prediction = top_action(initial_distribution)
        trajectory_match = initial_prediction == next_action
        trajectory_probability = float(
            initial_distribution.get(next_action, 0.0)
        )
        actual_purchase = (
            next_action in {"PURCHASE", "PURCHASE_NOW"}
            or detail_action == "PURCHASE"
        )
        predicted_purchase = initial_prediction in {"PURCHASE", "PURCHASE_NOW"}
        predicted_purchase_probability = float(
            initial_distribution.get("PURCHASE", 0.0)
            + initial_distribution.get("PURCHASE_NOW", 0.0)
        )

        if initial_state == "ITEM_EXPOSURE":
            exposure_labels.append(next_action)
            exposure_distributions.append(initial_distribution)
            if detail_action is not None:
                detail_distribution = distributions.get("ITEM_DETAIL", {})
                detail_prediction = top_action(detail_distribution)
                trajectory_probability *= float(
                    detail_distribution.get(str(detail_action), 0.0)
                )
                detail_labels.append(str(detail_action))
                detail_distributions.append(detail_distribution)
                trajectory_match = (
                    trajectory_match and detail_prediction == detail_action
                )
                predicted_purchase = (
                    initial_prediction == "CLICK"
                    and detail_prediction == "PURCHASE"
                )
            predicted_purchase_probability = float(
                initial_distribution.get("PURCHASE_NOW", 0.0)
                + initial_distribution.get("CLICK", 0.0)
                * distributions.get("ITEM_DETAIL", {}).get("PURCHASE", 0.0)
            )
        else:
            detail_labels.append(next_action)
            detail_distributions.append(initial_distribution)

        trajectory_matches.append(trajectory_match)
        purchase_labels.append(actual_purchase)
        purchase_predictions.append(predicted_purchase)
        purchase_probabilities.append(predicted_purchase_probability)
        trajectory_probabilities.append(trajectory_probability)

    exposure_metrics = multiclass_metrics(
        exposure_labels,
        exposure_distributions,
        actions=STATE_ACTIONS["ITEM_EXPOSURE"],
    )
    detail_metrics = multiclass_metrics(
        detail_labels,
        detail_distributions,
        actions=STATE_ACTIONS["ITEM_DETAIL"],
    )
    exposure_expected = expected_multiclass_metrics(
        exposure_labels,
        exposure_distributions,
        actions=STATE_ACTIONS["ITEM_EXPOSURE"],
    )
    detail_expected = expected_multiclass_metrics(
        detail_labels,
        detail_distributions,
        actions=STATE_ACTIONS["ITEM_DETAIL"],
    )
    true_positive = sum(
        label and prediction
        for label, prediction in zip(purchase_labels, purchase_predictions)
    )
    false_positive = sum(
        not label and prediction
        for label, prediction in zip(purchase_labels, purchase_predictions)
    )
    false_negative = sum(
        label and not prediction
        for label, prediction in zip(purchase_labels, purchase_predictions)
    )
    true_negative = sum(
        not label and not prediction
        for label, prediction in zip(purchase_labels, purchase_predictions)
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    purchase_f1 = 2.0 * precision * recall / max(EPSILON, precision + recall)
    purchase_expected = expected_binary_metrics(
        purchase_labels,
        purchase_probabilities,
    )
    trajectory_nll = -sum(
        math.log(max(EPSILON, probability))
        for probability in trajectory_probabilities
    ) / max(1, len(trajectory_probabilities))
    return {
        "cases": len(rows),
        "primary_metrics": {
            "exposure_expected_macro_f1": exposure_expected[
                "macro_f1_supported"
            ],
            "detail_expected_macro_f1": detail_expected[
                "macro_f1_supported"
            ],
            "terminal_purchase_expected_f1": purchase_expected["f1"],
            "trajectory_expected_exact_match": round(
                sum(trajectory_probabilities)
                / max(1, len(trajectory_probabilities)),
                8,
            ),
        },
        "proper_scoring": {
            "exposure_log_loss": exposure_metrics["log_loss"],
            "exposure_multiclass_brier": exposure_metrics[
                "multiclass_brier"
            ],
            "detail_log_loss": detail_metrics["log_loss"],
            "detail_multiclass_brier": detail_metrics[
                "multiclass_brier"
            ],
            "terminal_purchase_log_loss": purchase_expected["log_loss"],
            "terminal_purchase_brier": purchase_expected["brier"],
            "trajectory_negative_log_likelihood": round(
                trajectory_nll, 8
            ),
        },
        "stochastic_expected": {
            "exposure": exposure_expected,
            "detail": detail_expected,
            "terminal_purchase": purchase_expected,
            "terminal_purchase_monte_carlo": monte_carlo_binary_f1(
                purchase_labels,
                purchase_probabilities,
            ),
            "trajectory": {
                "count": len(trajectory_probabilities),
                "expected_exact_match": round(
                    sum(trajectory_probabilities)
                    / max(1, len(trajectory_probabilities)),
                    8,
                ),
            },
        },
        "argmax_diagnostic": {
            "exposure": exposure_metrics,
            "detail": detail_metrics,
            "terminal_purchase": {
                "count": len(purchase_labels),
                "positives": sum(purchase_labels),
                "accuracy": round(
                    (true_positive + true_negative)
                    / max(1, len(purchase_labels)),
                    8,
                ),
                "precision": round(precision, 8),
                "recall": round(recall, 8),
                "f1": round(purchase_f1, 8),
                "confusion_matrix": {
                    "tp": true_positive,
                    "fp": false_positive,
                    "tn": true_negative,
                    "fn": false_negative,
                },
            },
            "trajectory": {
                "count": len(trajectory_matches),
                "exact_match": round(
                    sum(trajectory_matches) / max(1, len(trajectory_matches)),
                    8,
                ),
            },
        },
        "exposure": exposure_metrics,
        "detail": detail_metrics,
        "terminal_purchase": {
            "count": len(purchase_labels),
            "positives": sum(purchase_labels),
            "accuracy": round(
                (true_positive + true_negative) / max(1, len(purchase_labels)),
                8,
            ),
            "precision": round(precision, 8),
            "recall": round(recall, 8),
            "f1": round(purchase_f1, 8),
            "confusion_matrix": {
                "tp": true_positive,
                "fp": false_positive,
                "tn": true_negative,
                "fn": false_negative,
            },
        },
        "trajectory": {
            "count": len(trajectory_matches),
            "exact_match": round(
                sum(trajectory_matches) / max(1, len(trajectory_matches)), 8
            ),
        },
        "limitations": [
            "Synthetic logs do not distinguish EXIT from SKIP/BACK.",
            "ITEM_DETAIL metrics use only trajectories that actually reached detail, plus checkout-start cases.",
            "Oracle probabilities are not used by these action metrics.",
        ],
    }


def evaluate_next_actions_by_slice(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for field in ("motivation_segment", "product_type"):
        values = sorted({str(row.get(field, "unknown")) for row in rows})
        reports[field] = {
            value: evaluate_next_actions(
                tuple(
                    row
                    for row in rows
                    if str(row.get(field, "unknown")) == value
                )
            )
            for value in values
        }
    return reports
