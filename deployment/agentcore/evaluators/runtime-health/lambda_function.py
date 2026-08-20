from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
    EvaluatorOutput,
    custom_code_based_evaluator,
)


_ERROR_STATUS_CODES = {"2", "ERROR", "STATUS_CODE_ERROR"}
_ERROR_SEVERITIES = {"ERROR", "FATAL"}
_HTTP_STATUS_KEYS = {
    "http.response.status_code",
    "http.status_code",
    "http.status_code.value",
}
_INPUT_TOKEN_KEYS = {
    "gen_ai.usage.input_tokens",
    "input_tokens",
    "inputTokens",
}
_OUTPUT_TOKEN_KEYS = {
    "gen_ai.usage.output_tokens",
    "output_tokens",
    "outputTokens",
}


def _unwrap(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    for key in (
        "stringValue",
        "boolValue",
        "intValue",
        "doubleValue",
        "value",
    ):
        if key in value:
            return _unwrap(value[key])
    return value


def _attributes(span: Mapping[str, Any]) -> dict[str, Any]:
    raw = span.get("attributes") or {}
    if isinstance(raw, Mapping):
        return {str(key): _unwrap(value) for key, value in raw.items()}
    if isinstance(raw, list):
        result: dict[str, Any] = {}
        for entry in raw:
            if isinstance(entry, Mapping) and "key" in entry:
                result[str(entry["key"])] = _unwrap(entry.get("value"))
        return result
    return {}


def _is_true(value: Any) -> bool:
    value = _unwrap(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_int(value: Any) -> int:
    try:
        return max(0, int(_unwrap(value)))
    except (TypeError, ValueError):
        return 0


def _status_code(span: Mapping[str, Any]) -> str:
    status = span.get("status") or {}
    if isinstance(status, Mapping):
        return str(_unwrap(status.get("code", ""))).upper()
    return str(_unwrap(status)).upper()


def _span_issues(span: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    name = str(span.get("name") or span.get("scope", {}).get("name") or "span")
    status = _status_code(span)
    if status in _ERROR_STATUS_CODES:
        issues.append(f"{name}:error-status")

    severity = str(span.get("severityText") or "").upper()
    if severity in _ERROR_SEVERITIES:
        issues.append(f"{name}:{severity.lower()}-log")

    attributes = _attributes(span)
    event_name = str(attributes.get("event.name") or "").lower()
    if event_name == "exception" or "exception.type" in attributes:
        issues.append(f"{name}:exception")

    for key in _HTTP_STATUS_KEYS:
        if _as_int(attributes.get(key)) >= 500:
            issues.append(f"{name}:http-5xx")
            break

    for key, value in attributes.items():
        normalized = key.lower().replace("-", "_")
        if "fallback" in normalized and _is_true(value):
            issues.append(f"{name}:fallback")
            break

    return issues


def _token_totals(spans: list[dict[str, Any]]) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for span in spans:
        attributes = _attributes(span)
        input_tokens += max(
            (_as_int(attributes.get(key)) for key in _INPUT_TOKEN_KEYS),
            default=0,
        )
        output_tokens += max(
            (_as_int(attributes.get(key)) for key in _OUTPUT_TOKEN_KEYS),
            default=0,
        )
    return input_tokens, output_tokens


def evaluate_runtime_quality(input: EvaluatorInput) -> EvaluatorOutput:
    spans = input.session_spans
    if not spans:
        return EvaluatorOutput(
            errorCode="NO_SPANS",
            errorMessage="No session spans were supplied to the evaluator.",
        )

    issues = sorted(
        {
            issue
            for span in spans
            if isinstance(span, Mapping)
            for issue in _span_issues(span)
        }
    )
    input_tokens, output_tokens = _token_totals(spans)
    explanation = (
        f"spans={len(spans)}, input_tokens={input_tokens}, "
        f"output_tokens={output_tokens}"
    )
    if issues:
        return EvaluatorOutput(
            value=0.0,
            label="Fail",
            explanation=f"{explanation}, issues={';'.join(issues[:10])}",
        )
    return EvaluatorOutput(
        value=1.0,
        label="Pass",
        explanation=f"{explanation}, issues=none",
    )


@custom_code_based_evaluator()
def handler(input: EvaluatorInput, context) -> EvaluatorOutput:
    return evaluate_runtime_quality(input)
