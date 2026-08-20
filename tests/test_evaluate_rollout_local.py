from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_rollout_local import read_resume_rows


class EvaluateRolloutLocalTest(unittest.TestCase):
    def test_resume_merges_completed_run_and_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "predictions.jsonl").write_text(
                json.dumps({"case_id": "case-1", "value": "complete"})
                + "\n"
                + json.dumps({"case_id": "case-2", "value": "old"})
                + "\n",
                encoding="utf-8",
            )
            (output / "predictions.partial.jsonl").write_text(
                json.dumps({"case_id": "case-2", "value": "new"})
                + "\n"
                + json.dumps({"case_id": "case-3", "value": "partial"})
                + "\n",
                encoding="utf-8",
            )

            rows = {
                row["case_id"]: row for row in read_resume_rows(output)
            }

            self.assertEqual(set(rows), {"case-1", "case-2", "case-3"})
            self.assertEqual(rows["case-2"]["value"], "new")


if __name__ == "__main__":
    unittest.main()
