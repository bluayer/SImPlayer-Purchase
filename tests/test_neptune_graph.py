from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from purchase_behavior_simulator.models import SimulationRequest
from purchase_behavior_simulator.neptune_graph import (
    NeptuneGraphConfig,
    NeptuneGraphEvidenceProvider,
)


class FakeNeptuneClient:
    def __init__(self) -> None:
        self.calls = []

    def execute_open_cypher_query(self, **kwargs):
        self.calls.append(kwargs)
        if "MATCH (target:Item" in kwargs["openCypherQuery"]:
            return {
                "results": [
                    {"relationType": "IN_CATEGORY", "neighborId": "upgrade"},
                    {"relationType": "TARGETS", "neighborId": "warrior"},
                ]
            }
        return {
            "results": [
                {
                    "sourceItemId": "old-item",
                    "interactionType": "PURCHASED",
                    "interactionTimestamp": "2026-08-17T00:00:00+00:00",
                    "relationType": "IN_CATEGORY",
                    "neighborId": "upgrade",
                },
                {
                    "sourceItemId": "old-item",
                    "interactionType": "PURCHASED",
                    "interactionTimestamp": "2026-08-17T00:00:00+00:00",
                    "relationType": "TARGETS",
                    "neighborId": "mage",
                },
            ]
        }


class RequestDefinedBundleClient:
    def __init__(self) -> None:
        self.calls = []

    def execute_open_cypher_query(self, **kwargs):
        self.calls.append(kwargs)
        if "MATCH (target:Item" in kwargs["openCypherQuery"]:
            return {"results": []}
        return {
            "results": [
                {
                    "sourceItemId": "currency-1",
                    "interactionType": "PURCHASED",
                    "interactionTimestamp": "2026-08-17T00:00:00+00:00",
                    "relationType": None,
                    "neighborId": None,
                }
            ]
        }


class NeptuneGraphTest(unittest.TestCase):
    def test_serializes_open_cypher_parameters_for_data_api(self) -> None:
        client = FakeNeptuneClient()
        provider = NeptuneGraphEvidenceProvider(
            config=NeptuneGraphConfig(endpoint_url="https://example:8182"),
            client=client,
        )

        provider._execute("RETURN $itemId", {"itemId": "item-1", "limit": 10})

        self.assertEqual(
            json.loads(client.calls[0]["parameters"]),
            {"itemId": "item-1", "limit": 10},
        )

    def test_calculates_multi_meta_path_affinity(self) -> None:
        client = FakeNeptuneClient()
        provider = NeptuneGraphEvidenceProvider(
            config=NeptuneGraphConfig(endpoint_url="https://example:8182"),
            client=client,
        )
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "user-1"},
                "item": {"item_id": "item-1", "categories": ["upgrade"]},
                "context": {"timestamp": "2026-08-18T00:00:00+00:00"},
            }
        )
        evidence = provider.get_evidence(request)
        self.assertEqual(len(client.calls), 2)
        self.assertIsNotNone(evidence.precomputed_affinity)
        self.assertGreater(evidence.precomputed_affinity, 0.5)
        self.assertGreater(evidence.precomputed_confidence, 0.0)
        self.assertTrue(evidence.retrieved_evidence)
        self.assertIn("simuser.kg-evidence.v1", evidence.retrieved_evidence[0])
        self.assertIn("in_category", evidence.meta_path_scores)
        self.assertIn("targets", evidence.meta_path_scores)

    def test_request_defined_bundle_relations_work_before_graph_upsert(self) -> None:
        provider = NeptuneGraphEvidenceProvider(
            config=NeptuneGraphConfig(endpoint_url="https://example:8182"),
            client=RequestDefinedBundleClient(),
        )
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "user-1"},
                "target_product": {
                    "product_id": "new-bundle",
                    "product_type": "bundle",
                    "categories": ["bundle"],
                    "components": ["currency-1", "upgrade-1"],
                },
                "context": {"timestamp": "2026-08-18T00:00:00+00:00"},
            }
        )

        evidence = provider.get_evidence(request)

        self.assertGreater(evidence.precomputed_affinity, 0.5)
        self.assertIn("contains", evidence.meta_path_scores)
        self.assertIn("currency-1", evidence.retrieved_evidence[0])

    def test_category_scenario_replaces_stored_target_category(self) -> None:
        provider = NeptuneGraphEvidenceProvider(
            config=NeptuneGraphConfig(endpoint_url="https://example:8182"),
            client=FakeNeptuneClient(),
        )
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "user-1"},
                "target_product": {
                    "product_id": "item-1",
                    "categories": ["upgrade"],
                },
                "product_scenario": {
                    "remove_categories": ["upgrade"],
                    "add_categories": ["cosmetic"],
                },
                "context": {"timestamp": "2026-08-18T00:00:00+00:00"},
            }
        )

        evidence = provider.get_evidence(request)

        self.assertEqual(evidence.precomputed_affinity, 0.5)
        self.assertEqual(evidence.precomputed_confidence, 0.0)

    def test_unrelated_history_does_not_create_confident_negative_evidence(
        self,
    ) -> None:
        provider = NeptuneGraphEvidenceProvider(
            config=NeptuneGraphConfig(endpoint_url="https://example:8182"),
            client=FakeNeptuneClient(),
        )
        history = [
            {
                "sourceItemId": f"unrelated-{index}",
                "interactionType": "PURCHASED",
                "interactionTimestamp": "2026-08-17T00:00:00+00:00",
                "relationType": "IN_CATEGORY",
                "neighborId": "cosmetic",
            }
            for index in range(100)
        ]

        evidence = provider.calculate_evidence(
            target_rows=(
                {"relationType": "IN_CATEGORY", "neighborId": "upgrade"},
            ),
            history_rows=history,
            now=datetime(2026, 8, 18, tzinfo=timezone.utc),
        )

        self.assertEqual(evidence.precomputed_affinity, 0.5)
        self.assertEqual(evidence.precomputed_confidence, 0.0)
        self.assertFalse(evidence.retrieved_evidence)

    def test_empty_graph_returns_neutral_evidence(self) -> None:
        provider = NeptuneGraphEvidenceProvider(
            config=NeptuneGraphConfig(endpoint_url="https://example:8182"),
            client=FakeNeptuneClient(),
        )
        evidence = provider.calculate_evidence(
            target_rows=(),
            history_rows=(),
            now=datetime(2026, 8, 18, tzinfo=timezone.utc),
        )
        self.assertEqual(evidence.precomputed_affinity, 0.5)
        self.assertEqual(evidence.precomputed_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
