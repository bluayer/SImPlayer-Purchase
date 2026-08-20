from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.estimate_trace_token_cost import estimate_cost, summarize_tokens


class EstimateTraceTokenCostTest(unittest.TestCase):
    def test_separates_cached_input_and_calculates_case_cost(self) -> None:
        rows = [
            {
                "trace": {
                    "events": [
                        {
                            "metrics": {
                                "usage": {
                                    "inputTokens": 100,
                                    "cacheReadInputTokens": 40,
                                    "outputTokens": 10,
                                }
                            }
                        }
                    ]
                }
            },
            {
                "trace": {
                    "events": [
                        {
                            "metrics": {
                                "usage": {
                                    "inputTokens": 200,
                                    "cacheReadInputTokens": 50,
                                    "outputTokens": 20,
                                }
                            }
                        }
                    ]
                }
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            predictions = Path(temporary) / "predictions.jsonl"
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            tokens = summarize_tokens(predictions)

        cost = estimate_cost(
            tokens,
            input_per_million=5.0,
            cached_input_per_million=0.5,
            output_per_million=30.0,
        )

        self.assertEqual(tokens["cases"], 2)
        self.assertEqual(tokens["uncached_input_tokens"], 210)
        self.assertEqual(tokens["cached_input_tokens"], 90)
        self.assertEqual(tokens["output_tokens"], 30)
        self.assertEqual(cost["total_usd"], 0.001995)
        self.assertEqual(cost["per_case_usd"], 0.000998)


if __name__ == "__main__":
    unittest.main()
