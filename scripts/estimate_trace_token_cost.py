from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def _usage_rows(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for event in row.get("trace", {}).get("events", []):
        usage = event.get("metrics", {}).get("usage")
        if isinstance(usage, dict):
            yield usage


def summarize_tokens(path: Path) -> dict[str, int]:
    cases = 0
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            cases += 1
            for usage in _usage_rows(row):
                input_tokens += int(usage.get("inputTokens", 0))
                cached_input_tokens += int(
                    usage.get("cacheReadInputTokens", 0)
                )
                output_tokens += int(usage.get("outputTokens", 0))
    if cached_input_tokens > input_tokens:
        raise ValueError("cached input tokens exceed total input tokens")
    return {
        "cases": cases,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": input_tokens - cached_input_tokens,
        "output_tokens": output_tokens,
    }


def estimate_cost(
    tokens: dict[str, int],
    *,
    input_per_million: float,
    cached_input_per_million: float,
    output_per_million: float,
) -> dict[str, float]:
    uncached_cost = (
        tokens["uncached_input_tokens"] / 1_000_000 * input_per_million
    )
    cached_cost = (
        tokens["cached_input_tokens"]
        / 1_000_000
        * cached_input_per_million
    )
    output_cost = tokens["output_tokens"] / 1_000_000 * output_per_million
    total = uncached_cost + cached_cost + output_cost
    cases = tokens["cases"]
    return {
        "uncached_input_usd": round(uncached_cost, 6),
        "cached_input_usd": round(cached_cost, 6),
        "output_usd": round(output_cost, 6),
        "total_usd": round(total, 6),
        "per_case_usd": round(total / cases, 6) if cases else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate model token cost from observable evaluation traces."
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--input-per-million", type=float, required=True)
    parser.add_argument("--cached-input-per-million", type=float, required=True)
    parser.add_argument("--output-per-million", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokens = summarize_tokens(args.predictions)
    costs = estimate_cost(
        tokens,
        input_per_million=args.input_per_million,
        cached_input_per_million=args.cached_input_per_million,
        output_per_million=args.output_per_million,
    )
    print(
        json.dumps(
            {
                "predictions": str(args.predictions),
                "tokens": tokens,
                "rates_usd_per_million": {
                    "input": args.input_per_million,
                    "cached_input": args.cached_input_per_million,
                    "output": args.output_per_million,
                },
                "cost": costs,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
