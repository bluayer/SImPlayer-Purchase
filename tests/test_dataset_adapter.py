from __future__ import annotations

import json
import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from purchase_behavior_simulator.dataset_adapter import (
    CanonicalJsonlDatasetAdapter,
    CanonicalDataset,
    CanonicalImpression,
    CanonicalItem,
    CanonicalUser,
    DatasetValidationError,
    EvaluationProtocolBuilder,
    MappedTabularDatasetAdapter,
    ProductionExportBuilder,
    ProductionExportConfig,
    ProtocolBuildConfig,
)
from purchase_behavior_simulator.evaluation import read_jsonl, save_protocol
from purchase_behavior_simulator.models import ProductComponent


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "dataset_adapter"


class DatasetAdapterTest(unittest.TestCase):
    def test_mapped_event_rows_are_normalized(self) -> None:
        dataset = MappedTabularDatasetAdapter(EXAMPLE / "config.json").load()

        validation = dataset.validate()
        self.assertEqual(validation["users"], 2)
        self.assertEqual(validation["items"], 4)
        self.assertEqual(validation["impressions"], 12)
        self.assertEqual(validation["purchases"], 3)
        self.assertTrue(validation["complete_exposure"])
        purchased = next(
            row for row in dataset.impressions if row.impression_id == "imp-001"
        )
        self.assertEqual(
            purchased.action_labels(),
            ("ITEM_EXPOSURE", "CLICK", "PURCHASE"),
        )
        self.assertEqual(
            dataset.items["sku-currency"].attributes["character"], "mage"
        )
        self.assertEqual(purchased.game_state["currency_balance"], 30000.0)
        self.assertEqual(
            purchased.game_state["current_goals"],
            ["progression", "event"],
        )
        self.assertEqual(purchased.game_state["owned_item_ids"], [])

    def test_evaluation_protocol_uses_only_pre_cutoff_history(self) -> None:
        dataset = MappedTabularDatasetAdapter(EXAMPLE / "config.json").load()
        protocol = EvaluationProtocolBuilder(
            dataset,
            ProtocolBuildConfig(
                selected_users=2,
                cases_per_user=2,
                history_fraction=0.5,
                history_limit=10,
                seed=7,
            ),
        ).build()
        repeated = EvaluationProtocolBuilder(
            dataset,
            ProtocolBuildConfig(
                selected_users=2,
                cases_per_user=2,
                history_fraction=0.5,
                history_limit=10,
                seed=7,
            ),
        ).build()

        self.assertEqual(len(protocol.cases), 4)
        self.assertEqual(protocol.run_id, repeated.run_id)
        self.assertEqual(
            [case.case_id for case in protocol.cases],
            [case.case_id for case in repeated.cases],
        )
        self.assertEqual(
            len(
                {
                    case.original_user_id
                    for case in protocol.cases[:2]
                }
            ),
            2,
        )
        self.assertTrue(
            all(case.oracle_probability is None for case in protocol.cases)
        )
        for bootstrap in protocol.bootstrap_payloads:
            timestamps = [
                datetime.fromisoformat(event["timestamp"])
                for event in bootstrap["observation"]["events"]
            ]
            self.assertTrue(
                all(timestamp < datetime(2026, 1, 4, tzinfo=timezone.utc)
                    for timestamp in timestamps)
            )
            transition_actions = {
                transition["action"]
                for transition in bootstrap["observation"]["transitions"]
            }
            self.assertTrue(
                transition_actions.issubset(
                    {
                        "CLICK",
                        "SKIP",
                        "EXIT",
                        "PURCHASE_NOW",
                        "PURCHASE",
                        "BACK",
                    }
                )
            )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_protocol(protocol, output)
            answers = list(read_jsonl(output / "answer_key.jsonl"))
            self.assertEqual(len(answers), 4)
            self.assertTrue(
                all(answer["oracle_probability"] is None for answer in answers)
            )

    def test_evaluation_protocol_preserves_game_state(self) -> None:
        timestamps = [
            datetime(2026, 1, day, tzinfo=timezone.utc)
            for day in range(1, 5)
        ]
        dataset = CanonicalDataset(
            dataset_name="state-rich",
            users={"u1": CanonicalUser(user_id="u1")},
            items={
                "i1": CanonicalItem(
                    item_id="i1",
                    categories=("upgrade",),
                    price=9900,
                )
            },
            impressions=tuple(
                CanonicalImpression(
                    impression_id=f"imp-{index}",
                    user_id="u1",
                    item_id="i1",
                    timestamp=timestamp,
                    session_id=f"s-{index}",
                    game_state={
                        "currency_balance": 12000 + index,
                        "progression_need": 0.5 + 0.1 * index,
                        "recent_failure_intensity": 0.4 + 0.1 * index,
                        "inventory_overlap": 0.2 * index,
                        "event_urgency": 0.3 + 0.1 * index,
                        "purchase_cooldown": 0.1 * index,
                        "current_goals": ["progress:upgrade"],
                        "owned_item_ids": [],
                    },
                )
                for index, timestamp in enumerate(timestamps)
            ),
        )
        protocol = EvaluationProtocolBuilder(
            dataset,
            ProtocolBuildConfig(
                selected_users=1,
                cases_per_user=2,
                history_fraction=0.5,
                seed=3,
                require_game_state=True,
            ),
        ).build()

        self.assertEqual(len(protocol.cases), 2)
        self.assertTrue(
            all(
                case.request["game_state"]["progression_need"]
                >= 0.5
                for case in protocol.cases
            )
        )

    def test_state_rich_protocol_accepts_complete_mapped_rows(self) -> None:
        dataset = MappedTabularDatasetAdapter(
            EXAMPLE / "config.json"
        ).load()
        protocol = EvaluationProtocolBuilder(
            dataset,
            ProtocolBuildConfig(
                selected_users=2,
                cases_per_user=2,
                history_fraction=0.5,
                require_game_state=True,
            ),
        ).build()
        self.assertEqual(len(protocol.cases), 4)

    def test_production_export_is_pseudonymized_and_has_no_answer_key(self) -> None:
        dataset = MappedTabularDatasetAdapter(EXAMPLE / "config.json").load()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = ProductionExportBuilder(
                dataset,
                ProductionExportConfig(
                    as_of=datetime(2026, 1, 4, tzinfo=timezone.utc),
                    identity_salt="test-only-salt",
                    allow_synthetic=True,
                ),
            ).write(output)

            self.assertFalse(report["contains_answer_key"])
            self.assertFalse(report["contains_oracle_probability"])
            self.assertFalse((output / "answer_key.jsonl").exists())
            memory_text = (output / "memory_import.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("p-001", memory_text)
            self.assertNotIn("oracle_probability", memory_text)
            self.assertNotIn("2026-01-05", memory_text)
            self.assertNotIn("2026-01-06", memory_text)
            self.assertIn("historical_import", memory_text)
            memory_rows = list(read_jsonl(output / "memory_import.jsonl"))
            transition_pairs = {
                (transition["state"], transition["action"])
                for row in memory_rows
                for transition in row["observation"]["transitions"]
            }
            self.assertIn(("ITEM_EXPOSURE", "CLICK"), transition_pairs)
            self.assertIn(("ITEM_DETAIL", "PURCHASE"), transition_pairs)
            self.assertTrue((output / "neptune" / "nodes.csv").is_file())
            self.assertTrue((output / "neptune" / "edges.csv").is_file())

    def test_production_export_rejects_synthetic_by_default(self) -> None:
        dataset = MappedTabularDatasetAdapter(EXAMPLE / "config.json").load()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                DatasetValidationError, "synthetic data"
            ):
                ProductionExportBuilder(
                    dataset,
                    ProductionExportConfig(
                        as_of=datetime(2026, 1, 4, tzinfo=timezone.utc),
                        identity_salt="test-only-salt",
                    ),
                ).write(Path(temporary))

    def test_incremental_production_export_uses_exclusive_since(self) -> None:
        dataset = MappedTabularDatasetAdapter(EXAMPLE / "config.json").load()
        with tempfile.TemporaryDirectory() as temporary:
            report = ProductionExportBuilder(
                dataset,
                ProductionExportConfig(
                    since=datetime(2026, 1, 3, tzinfo=timezone.utc),
                    as_of=datetime(2026, 1, 5, tzinfo=timezone.utc),
                    identity_salt="test-only-salt",
                    allow_synthetic=True,
                ),
            ).write(Path(temporary))

        self.assertEqual(report["export_mode"], "incremental")
        self.assertEqual(
            report["since"],
            "2026-01-03T00:00:00+00:00",
        )
        self.assertGreater(report["exported_impressions"], 0)

    def test_canonical_round_trip_preserves_protocol_inputs(self) -> None:
        dataset = MappedTabularDatasetAdapter(EXAMPLE / "config.json").load()
        with tempfile.TemporaryDirectory() as temporary:
            canonical_dir = Path(temporary)
            dataset.write(canonical_dir)
            restored = CanonicalJsonlDatasetAdapter(canonical_dir).load()

            self.assertEqual(restored.validate(), dataset.validate())
            manifest = json.loads(
                (canonical_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["schema"], "simuser.canonical-dataset.v2"
            )

    def test_bundle_components_export_as_neptune_contains_edges(self) -> None:
        dataset = CanonicalDataset(
            dataset_name="bundle-demo",
            users={"u1": CanonicalUser(user_id="u1")},
            items={
                "currency-1": CanonicalItem(item_id="currency-1"),
                "upgrade-1": CanonicalItem(item_id="upgrade-1"),
                "bundle-1": CanonicalItem(
                    item_id="bundle-1",
                    product_type="bundle",
                    categories=("bundle",),
                    price=24900,
                    components=(
                        ProductComponent("currency-1", 2),
                        ProductComponent("upgrade-1", 1),
                    ),
                ),
            },
            impressions=(
                CanonicalImpression(
                    impression_id="imp-1",
                    user_id="u1",
                    item_id="bundle-1",
                    timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    session_id="s1",
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ProductionExportBuilder(
                dataset,
                ProductionExportConfig(
                    as_of=datetime(2026, 1, 2, tzinfo=timezone.utc),
                    identity_salt="test-only-salt",
                ),
            ).write(output)
            with (output / "neptune" / "edges.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

        contains = {
            (row[":START_ID"], row[":END_ID"], row["weight:Double"])
            for row in rows
            if row[":TYPE"] == "CONTAINS"
        }
        self.assertEqual(
            contains,
            {
                ("item:bundle-1", "item:currency-1", "2.0"),
                ("item:bundle-1", "item:upgrade-1", "1.0"),
            },
        )

    def test_cart_action_is_rejected_during_configuration(self) -> None:
        config = json.loads(
            (EXAMPLE / "config.json").read_text(encoding="utf-8")
        )
        config["files"] = {
            key: str((EXAMPLE / value).resolve())
            for key, value in config["files"].items()
        }
        config["action_map"]["OPEN_DETAIL"] = "ADD_TO_CART"
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config_path.write_text(
                json.dumps(config), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                DatasetValidationError, "cart action"
            ):
                MappedTabularDatasetAdapter(config_path).load()


if __name__ == "__main__":
    unittest.main()
