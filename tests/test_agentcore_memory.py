from __future__ import annotations

import unittest
from datetime import datetime, timezone

from purchase_behavior_simulator.agentcore_memory import AgentCoreMemoryProvider
from purchase_behavior_simulator.models import (
    Item,
    ObservationBatch,
    SimulationRequest,
    SimulationResult,
)


class FakeMemorySession:
    def __init__(self) -> None:
        self.searches = []
        self.turns = []
        self.events = []

    def search_long_term_memories(self, **kwargs):
        self.searches.append(kwargs)
        return self._long_term_records()

    def list_long_term_memory_records(self, **kwargs):
        return self._long_term_records()

    @staticmethod
    def _long_term_records():
        return [
            {
                "content": {
                    "text": (
                        "<simuser-observation>"
                        '{"schema":"simuser.observation.v1",'
                        '"source":"historical_import","events":[{'
                        '"event_type":"purchase",'
                        '"timestamp":"2026-08-17T00:00:00+00:00",'
                        '"item_id":"old-item","categories":["upgrade"]}]}'
                        "</simuser-observation>"
                    )
                },
                "score": 0.9,
                "namespaces": ["/episodes/user-42/history"],
                "createdAt": "2026-08-17T00:00:00+00:00",
            }
        ]

    def add_turns(self, **kwargs):
        self.turns.append(kwargs)

    def list_events(self, **kwargs):
        return self.events


class FakeMemorySessionManager:
    def __init__(self) -> None:
        self.sessions = []
        self.session = FakeMemorySession()

    def create_memory_session(self, **kwargs):
        self.sessions.append(kwargs)
        return self.session


class FakeDataPlaneClient:
    def __init__(self) -> None:
        self.batches = []

    def batch_create_memory_records(self, **kwargs):
        self.batches.append(kwargs)
        return {
            "successfulRecords": [
                {"requestIdentifier": record["requestIdentifier"]}
                for record in kwargs["records"]
            ],
            "failedRecords": [],
        }


class FakeControlPlaneClient:
    def __init__(self) -> None:
        self.calls = []

    def get_memory(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "memory": {
                "strategies": [
                    {
                        "type": "EPISODIC",
                        "strategyId": "episodic-discovered",
                    },
                    {
                        "type": "SEMANTIC",
                        "strategyId": "semantic-discovered",
                    },
                ]
            }
        }


class AgentCoreMemoryTest(unittest.TestCase):
    def test_runs_each_self_ask_query_against_the_users_episodes(self) -> None:
        manager = FakeMemorySessionManager()
        provider = AgentCoreMemoryProvider(
            memory_id="memory-1",
            session_manager=manager,
            data_plane_client=FakeDataPlaneClient(),
        )

        evidence = provider.retrieve(
            user_id="user-42",
            queries=("positive history", "negative history"),
            item=Item(item_id="target", categories=("upgrade",)),
            now=datetime(2026, 8, 18, tzinfo=timezone.utc),
        )

        self.assertEqual(len(manager.session.searches), 3)
        self.assertTrue(
            all(
                search["namespace_path"] == "/episodes/user-42"
                for search in manager.session.searches[:2]
            )
        )
        self.assertEqual(
            manager.session.searches[-1]["namespace_path"],
            "/users/user-42/observed-transitions",
        )
        self.assertTrue(
            all(
                search["top_k"] == 2
                for search in manager.session.searches[:2]
            )
        )
        self.assertEqual(len(evidence.documents), 1)
        self.assertEqual(evidence.interactions[0].event_type, "purchase")

    def test_reads_current_session_observations_before_async_extraction(self) -> None:
        manager = FakeMemorySessionManager()
        manager.session.events = [
            {
                "eventTimestamp": "2026-08-18T00:00:00+00:00",
                "payload": [
                    {
                        "conversational": {
                            "content": {
                                "text": (
                                    "<simuser-observation>"
                                    '{"schema":"simuser.observation.v1",'
                                    '"source":"external_observation","events":[{'
                                    '"event_type":"refund",'
                                    '"timestamp":"2026-08-18T00:00:00+00:00",'
                                    '"item_id":"target","categories":["upgrade"]}]}'
                                    "</simuser-observation>"
                                )
                            },
                            "role": "USER",
                        }
                    }
                ],
            }
        ]
        provider = AgentCoreMemoryProvider(
            memory_id="memory-1",
            session_manager=manager,
            data_plane_client=FakeDataPlaneClient(),
        )

        evidence = provider.retrieve(
            user_id="user-42",
            queries=("history",),
            item=Item(item_id="target", categories=("upgrade",)),
            now=datetime(2026, 8, 18, tzinfo=timezone.utc),
            session_id="observed-session",
        )

        self.assertTrue(any(event.event_type == "refund" for event in evidence.interactions))
        self.assertTrue(
            any(document.kind == "current_session" for document in evidence.documents)
        )

    def test_ignores_empty_long_term_records(self) -> None:
        manager = FakeMemorySessionManager()
        valid_records = manager.session._long_term_records()
        manager.session._long_term_records = lambda: [{"content": {}}, *valid_records]
        provider = AgentCoreMemoryProvider(
            memory_id="memory-1",
            session_manager=manager,
            data_plane_client=FakeDataPlaneClient(),
        )

        evidence = provider.retrieve(
            user_id="user-42",
            queries=("history",),
            item=Item(item_id="target", categories=("upgrade",)),
            now=datetime(2026, 8, 18, tzinfo=timezone.utc),
        )

        self.assertEqual(len(evidence.documents), 1)
        self.assertEqual(evidence.interactions[0].event_type, "purchase")

    def test_records_only_validated_external_observations(self) -> None:
        manager = FakeMemorySessionManager()
        data_plane = FakeDataPlaneClient()
        provider = AgentCoreMemoryProvider(
            memory_id="memory-1",
            session_manager=manager,
            data_plane_client=data_plane,
            episodic_strategy_id="episodic-strategy-1",
            transition_strategy_id="semantic-transition-strategy-1",
        )
        batch = ObservationBatch.from_dict(
            {
                "user_id": "user-42",
                "session_id": "observed-session",
                "source": "external_observation",
                "events": [
                    {
                        "event_type": "purchase",
                        "timestamp": "2026-08-18T00:00:00+00:00",
                        "item_id": "item-1",
                    }
                ],
            }
        )

        receipt = provider.record_observations(batch, "Observed purchase; cause uncertain.")

        self.assertEqual(receipt.event_count, 1)
        self.assertEqual(receipt.long_term_record_count, 3)
        self.assertEqual(manager.sessions[-1]["session_id"], "observed-session")
        messages = manager.session.turns[0]["messages"]
        self.assertIn("<simuser-observation>", messages[0].text)
        self.assertIn("<simuser-reflection>", messages[1].text)
        records = data_plane.batches[0]["records"]
        self.assertEqual(records[0]["namespaces"], ["/episodes/user-42/observed-session"])
        self.assertEqual(
            records[1]["namespaces"],
            ["/users/user-42/observed-transitions"],
        )
        self.assertEqual(
            records[1]["memoryStrategyId"],
            "semantic-transition-strategy-1",
        )
        self.assertEqual(records[2]["namespaces"], ["/episodes/user-42"])
        self.assertTrue(
            all(
                record["memoryStrategyId"] == "episodic-strategy-1"
                for record in (records[0], records[2])
            )
        )

    def test_discovers_strategy_ids_for_a_new_memory(self) -> None:
        manager = FakeMemorySessionManager()
        data_plane = FakeDataPlaneClient()
        control_plane = FakeControlPlaneClient()
        provider = AgentCoreMemoryProvider(
            memory_id="memory-new",
            session_manager=manager,
            data_plane_client=data_plane,
            control_plane_client=control_plane,
        )
        batch = ObservationBatch.from_dict(
            {
                "user_id": "user-42",
                "session_id": "observed-session",
                "source": "external_observation",
                "events": [
                    {
                        "event_type": "purchase",
                        "timestamp": "2026-08-18T00:00:00+00:00",
                        "item_id": "item-1",
                    }
                ],
            }
        )

        provider.record_observations(batch, "Observed purchase.")

        self.assertEqual(control_plane.calls, [{"memoryId": "memory-new"}])
        records = data_plane.batches[0]["records"]
        self.assertEqual(
            records[0]["memoryStrategyId"],
            "episodic-discovered",
        )
        self.assertEqual(
            records[1]["memoryStrategyId"],
            "semantic-discovered",
        )

    def test_does_not_write_model_predictions_back_by_default(self) -> None:
        manager = FakeMemorySessionManager()
        provider = AgentCoreMemoryProvider(
            memory_id="memory-1",
            session_manager=manager,
            data_plane_client=FakeDataPlaneClient(),
        )
        request = SimulationRequest.from_dict(
            {
                "user": {"user_id": "user-42"},
                "item": {"item_id": "item-1"},
            }
        )
        result = SimulationResult(
            probability=0.8,
            confidence=0.7,
            eligible=True,
            is_calibrated=False,
            components={},
            reasons=(),
            contradictions=(),
            model_version="test",
            calibration_version=None,
        )

        provider.record_prediction(request, result)

        self.assertEqual(manager.sessions, [])
        self.assertEqual(manager.session.turns, [])


if __name__ == "__main__":
    unittest.main()
