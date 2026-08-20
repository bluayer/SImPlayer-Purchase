from __future__ import annotations

import time
from typing import Any


TraceEvents = list[dict[str, Any]]
TRANSIENT_MODEL_ERROR_MARKERS = (
    "failed to mint bedrock mantle bearer token",
    "throttlingexception",
    "too many requests",
    "rate limit",
    "server had an error while processing your request",
)


def strands_result_metrics(result: Any) -> dict[str, Any]:
    """Return only observable usage/timing metrics, never model message content."""
    metrics = getattr(result, "metrics", None)
    if metrics is None:
        return {}
    usage = dict(getattr(metrics, "accumulated_usage", {}) or {})
    model_metrics = dict(getattr(metrics, "accumulated_metrics", {}) or {})
    return {
        "stop_reason": getattr(result, "stop_reason", None),
        "cycles": int(getattr(metrics, "cycle_count", 0)),
        "usage": {
            key: int(value)
            for key, value in usage.items()
            if key
            in {
                "inputTokens",
                "outputTokens",
                "totalTokens",
                "cacheReadInputTokens",
                "cacheWriteInputTokens",
            }
        },
        "model_latency_ms": int(model_metrics.get("latencyMs", 0)),
    }


def sanitized_error(exc: Exception, limit: int = 500) -> str:
    detail = " ".join(str(exc).split())[:limit]
    return f"{type(exc).__name__}: {detail}"


def invoke_with_transient_retries(
    operation: Any,
    *,
    max_attempts: int = 3,
) -> tuple[Any, int]:
    for attempt in range(1, max_attempts + 1):
        try:
            return operation(), attempt
        except Exception as exc:
            normalized = str(exc).lower()
            retryable = any(
                marker in normalized
                for marker in TRANSIENT_MODEL_ERROR_MARKERS
            )
            if not retryable or attempt >= max_attempts:
                raise
            time.sleep(0.75 * attempt)
    raise RuntimeError("transient retry loop ended unexpectedly")
