from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_action_stages import (
    stage_distribution,
    trace_uses_fallback,
)
from scripts.evaluate_next_actions import action_distributions, main


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts" / "dataset" / "protocol"


def read_first_jsonl(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


class EvaluateNextActionsScriptTest(unittest.TestCase):
    def test_stage_report_uses_public_distribution_for_fallback_case(
        self,
    ) -> None:
        expected = {
            "ITEM_EXPOSURE": {
                "CLICK": 0.2,
                "SKIP": 0.7,
                "EXIT": 0.05,
                "PURCHASE_NOW": 0.05,
            }
        }
        row = {
            "case_id": "fallback-case",
            "result": {"action_distributions": expected},
            "trace": {
                "events": [
                    {
                        "stage": "action_assessment_round",
                        "fallback": True,
                    }
                ]
            },
        }

        self.assertEqual(
            stage_distribution(row, "action_assessment_round"),
            expected,
        )
        self.assertTrue(trace_uses_fallback(row))

    def test_uses_eligibility_short_circuit_distribution(self) -> None:
        expected = {
            "ITEM_EXPOSURE": {
                "CLICK": 0.0,
                "SKIP": 0.98,
                "EXIT": 0.02,
                "PURCHASE_NOW": 0.0,
            },
            "ITEM_DETAIL": {
                "PURCHASE": 0.0,
                "BACK": 0.98,
                "EXIT": 0.02,
            },
        }

        self.assertEqual(
            action_distributions(
                {
                    "case_id": "owned-item",
                    "trace": {
                        "events": [
                            {
                                "stage": "eligibility_short_circuit",
                                "action_distributions": expected,
                            }
                        ]
                    },
                }
            ),
            expected,
        )

    def test_partial_prediction_run_joins_only_present_cases(self) -> None:
        answer = read_first_jsonl(PROTOCOL / "answer_key.jsonl")
        blind = read_first_jsonl(PROTOCOL / "blind_cases.jsonl")
        case_id = str(answer["case_id"])
        prediction = {
            "case_id": case_id,
            "trace": {
                "events": [
                    {
                        "stage": "scoring",
                        "action_distributions": {
                            "ITEM_EXPOSURE": {
                                "CLICK": 0.2,
                                "SKIP": 0.7,
                                "EXIT": 0.05,
                                "PURCHASE_NOW": 0.05,
                            },
                            "ITEM_DETAIL": {
                                "PURCHASE": 0.2,
                                "BACK": 0.7,
                                "EXIT": 0.1,
                            },
                        },
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol"
            output = root / "output"
            protocol.mkdir()
            (protocol / "answer_key.jsonl").write_text(
                json.dumps(answer, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (protocol / "blind_cases.jsonl").write_text(
                json.dumps(blind, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(prediction, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "--protocol-dir",
                        str(protocol),
                        "--predictions",
                        str(predictions),
                        "--output-dir",
                        str(output),
                    ]
                )

            report = json.loads(
                (output / "report.json").read_text(encoding="utf-8")
            )
            cases = (output / "cases.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(report["cases"], 1)
        self.assertEqual(len(cases), 1)


if __name__ == "__main__":
    unittest.main()
