from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_full_suite import (
    build_parser,
    main,
    replace_prediction_rows,
    write_retry_case_ids,
)


class RunFullSuiteTest(unittest.TestCase):
    def test_writes_and_clears_retry_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            retry_cases = root / "retry-case-ids.txt"
            predictions.write_text(
                json.dumps(
                    {
                        "case_id": "case-1",
                        "trace": {
                            "events": [
                                {
                                    "stage": "assessment_round",
                                    "fallback": True,
                                }
                            ]
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                write_retry_case_ids(predictions, retry_cases),
                ("case-1",),
            )
            self.assertEqual(
                retry_cases.read_text(encoding="utf-8"),
                "case-1\n",
            )

            predictions.write_text(
                json.dumps(
                    {
                        "case_id": "case-1",
                        "result": {"probability": 0.1},
                        "trace": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                write_retry_case_ids(predictions, retry_cases),
                (),
            )
            self.assertFalse(retry_cases.exists())

    def test_failed_case_is_written_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            retry_cases = root / "retry-case-ids.txt"
            predictions.write_text(
                json.dumps(
                    {
                        "case_id": "case-1",
                        "result": None,
                        "error": "TimeoutError",
                        "trace": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                write_retry_case_ids(predictions, retry_cases),
                ("case-1",),
            )

    def test_replaces_retry_rows_by_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            replacements = root / "replacements.jsonl"
            predictions.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {"case_id": "case-1", "result": {"probability": 0.1}},
                        {"case_id": "case-2", "result": None},
                    )
                ),
                encoding="utf-8",
            )
            replacements.write_text(
                json.dumps(
                    {
                        "case_id": "case-2",
                        "result": {"probability": 0.2},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            replace_prediction_rows(predictions, replacements)

            rows = [
                json.loads(line)
                for line in predictions.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["result"]["probability"], 0.1)
            self.assertEqual(rows[1]["result"]["probability"], 0.2)

    def test_default_worker_count_is_one(self) -> None:
        args = build_parser().parse_args(
            [
                "--protocol-dir",
                "protocol",
                "--output-dir",
                "output",
                "--model-id",
                "test-model",
            ]
        )

        self.assertEqual(args.workers, 1)

    def test_unbounded_run_requires_explicit_cost_confirmation(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "--confirm-model-cost",
        ):
            main(
                [
                    "--protocol-dir",
                    "artifacts/dataset/protocol",
                    "--output-dir",
                    "artifacts/evaluation/runs/test",
                    "--model-id",
                    "test-model",
                ]
            )

    @patch("scripts.run_full_suite.evaluate_stages")
    @patch("scripts.run_full_suite.evaluate_next_actions")
    @patch("scripts.run_full_suite.evaluate_simulations")
    def test_complete_checkpoint_can_be_reaggregated_without_cost_confirmation(
        self,
        evaluate_simulations,
        evaluate_next_actions,
        evaluate_stages,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol"
            simulation = root / "output" / "simulation"
            protocol.mkdir()
            simulation.mkdir(parents=True)
            (protocol / "blind_cases.jsonl").write_text(
                json.dumps({"case_id": "case-1"}) + "\n",
                encoding="utf-8",
            )
            (simulation / "predictions.jsonl").write_text(
                json.dumps({"case_id": "case-1"}) + "\n",
                encoding="utf-8",
            )

            main(
                [
                    "--protocol-dir",
                    str(protocol),
                    "--output-dir",
                    str(root / "output"),
                    "--model-id",
                    "test-model",
                ]
            )

        evaluate_simulations.assert_called_once()
        evaluate_next_actions.assert_called_once()
        evaluate_stages.assert_called_once()

    @patch("scripts.run_full_suite.evaluate_stages")
    @patch("scripts.run_full_suite.evaluate_next_actions")
    @patch("scripts.run_full_suite.evaluate_simulations")
    @patch("scripts.run_full_suite.write_retry_case_ids", return_value=())
    def test_limited_smoke_runs_all_evaluation_stages(
        self,
        write_retry_case_ids,
        evaluate_simulations,
        evaluate_next_actions,
        evaluate_stages,
    ) -> None:
        main(
            [
                "--protocol-dir",
                "artifacts/dataset/protocol",
                "--output-dir",
                "artifacts/evaluation/runs/smoke",
                "--model-id",
                "test-model",
                "--limit",
                "1",
            ]
        )

        evaluate_simulations.assert_called_once()
        evaluate_next_actions.assert_called_once()
        write_retry_case_ids.assert_called_once()
        evaluate_stages.assert_called_once()
        self.assertTrue(
            evaluate_stages.call_args.args[0][-1].endswith(
                "stage-report-advisory.json"
            )
        )

    @patch("scripts.run_full_suite.evaluate_stages")
    @patch("scripts.run_full_suite.evaluate_next_actions")
    @patch("scripts.run_full_suite.evaluate_simulations")
    def test_optional_fallback_retry_replaces_main_prediction(
        self,
        evaluate_simulations,
        evaluate_next_actions,
        evaluate_stages,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            protocol = root / "protocol"
            protocol.mkdir()

            def fake_evaluate(argv):
                output_dir = Path(argv[argv.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                predictions = output_dir / "predictions.jsonl"
                if "attempt-1" in str(output_dir):
                    row = {
                        "case_id": "case-1",
                        "result": {"probability": 0.2},
                        "trace": {"events": []},
                    }
                elif not predictions.exists():
                    row = {
                        "case_id": "case-1",
                        "result": {"probability": 0.1},
                        "trace": {
                            "events": [
                                {
                                    "stage": "assessment_round",
                                    "fallback": True,
                                }
                            ]
                        },
                    }
                else:
                    return
                predictions.write_text(
                    json.dumps(row) + "\n",
                    encoding="utf-8",
                )

            evaluate_simulations.side_effect = fake_evaluate
            main(
                [
                    "--protocol-dir",
                    str(protocol),
                    "--output-dir",
                    str(output),
                    "--model-id",
                    "test-model",
                    "--limit",
                    "1",
                    "--fallback-retries",
                    "1",
                ]
            )

            final_row = json.loads(
                (output / "simulation" / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(final_row["result"]["probability"], 0.2)
            self.assertFalse((output / "retry-case-ids.txt").exists())
            self.assertEqual(evaluate_simulations.call_count, 3)
            evaluate_next_actions.assert_called_once()
            evaluate_stages.assert_called_once()


if __name__ == "__main__":
    unittest.main()
