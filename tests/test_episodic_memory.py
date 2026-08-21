from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from purchase_behavior_simulator.episodic_memory import (
    behavior_events_from_text,
    empirical_transition_policy,
    observed_transitions_from_text,
    rerank_memory_documents,
    serialize_observation,
    serialize_observed_transitions,
    transitions_from_observation,
)
from purchase_behavior_simulator.models import (
    Item,
    MemoryDocument,
    ObservationBatch,
    ObservedStateTransition,
    ExposureContext,
)


NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


class EpisodicMemoryTest(unittest.TestCase):
    def test_empirical_transition_policy_shrinks_to_actor_prior(self) -> None:
        now = datetime(2026, 8, 19, tzinfo=timezone.utc)
        transition = ObservedStateTransition(
            state="ITEM_EXPOSURE",
            action="SKIP",
            next_state="EXITED",
            timestamp=now,
            item_id="target",
            categories=("upgrade",),
        )
        text = serialize_observed_transitions((transition,))
        actor_prior = {
            "ITEM_EXPOSURE": {
                "CLICK": 0.4,
                "SKIP": 0.4,
                "EXIT": 0.1,
                "PURCHASE_NOW": 0.1,
            },
            "ITEM_DETAIL": {
                "PURCHASE": 0.3,
                "BACK": 0.5,
                "EXIT": 0.2,
            },
        }

        policy, strengths, _ = empirical_transition_policy(
            (text,),
            item=Item(item_id="target", categories=("upgrade",)),
            now=now,
            context=ExposureContext(timestamp=now),
            prior_policy=actor_prior,
        )

        self.assertGreater(policy["ITEM_EXPOSURE"]["SKIP"], 0.4)
        self.assertLess(policy["ITEM_EXPOSURE"]["SKIP"], 0.75)
        self.assertLessEqual(strengths["ITEM_EXPOSURE"], 0.35)

    def test_rejects_cart_observations_for_game_items(self) -> None:
        with self.assertRaisesRegex(ValueError, "add_to_cart"):
            ObservationBatch.from_dict(
                {
                    "user_id": "u1",
                    "session_id": "s1",
                    "events": [{"event_type": "add_to_cart"}],
                }
            )

    def test_rejects_prediction_or_label_as_observation_source(self) -> None:
        for source in ("prediction", "model_output", "synthetic_label"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                ObservationBatch.from_dict(
                    {
                        "user_id": "u1",
                        "session_id": "s1",
                        "source": source,
                        "events": [{"event_type": "purchase"}],
                    }
                )

    def test_observation_marker_round_trips_into_behavior_events(self) -> None:
        batch = ObservationBatch.from_dict(
            {
                "user_id": "u1",
                "session_id": "s1",
                "source": "historical_import",
                "events": [
                    {
                        "event_type": "purchase",
                        "timestamp": "2026-08-17T00:00:00+00:00",
                        "item_id": "item-1",
                        "categories": ["upgrade"],
                        "rating": 5,
                    }
                ],
            }
        )

        events = behavior_events_from_text(serialize_observation(batch))

        self.assertEqual(events, batch.events)

    def test_reranker_prefers_relevant_recent_episode(self) -> None:
        item = Item(item_id="target", categories=("upgrade",))
        old = MemoryDocument(
            content="generic old episode",
            relevance=0.7,
            observed_at=NOW - timedelta(days=180),
        )
        recent = MemoryDocument(
            content="target upgrade purchase episode",
            relevance=0.7,
            observed_at=NOW - timedelta(days=1),
        )

        ranked = rerank_memory_documents((old, recent), item, NOW)

        self.assertEqual(ranked[0], recent)

    def test_observed_transition_round_trip_never_uses_counterfactuals(self) -> None:
        batch = ObservationBatch.from_dict(
            {
                "user_id": "u1",
                "session_id": "s1",
                "source": "external_observation",
                "events": [
                    {
                        "event_type": "click",
                        "timestamp": "2026-08-17T00:00:00+00:00",
                        "item_id": "item-1",
                        "categories": ["upgrade"],
                    },
                    {
                        "event_type": "view",
                        "timestamp": "2026-08-17T00:00:00+00:00",
                        "item_id": "item-1",
                        "categories": ["upgrade"],
                    },
                ],
            }
        )
        transitions = transitions_from_observation(batch)
        encoded = serialize_observed_transitions(transitions)

        decoded = observed_transitions_from_text(encoded)

        self.assertEqual(
            [transition.action for transition in decoded],
            ["CLICK", "BACK", "SKIP"],
        )
        self.assertNotIn("counterfactual", encoded.lower())

    def test_legacy_cart_transitions_are_ignored_by_grounding(self) -> None:
        encoded = serialize_observed_transitions(
            (
                ObservedStateTransition(
                    state="ITEM_DETAIL",
                    action="ADD_TO_CART",
                    next_state="CART",
                    timestamp=NOW,
                    item_id="item-1",
                    categories=("upgrade",),
                ),
            )
        )

        distributions, strengths, count = empirical_transition_policy(
            (encoded,),
            item=Item(item_id="item-1", categories=("upgrade",)),
            now=NOW,
            context=ExposureContext(),
        )

        self.assertEqual(distributions, {})
        self.assertEqual(strengths, {})
        self.assertEqual(count, 0)

    def test_observed_transition_round_trip_preserves_historical_source(
        self,
    ) -> None:
        transition = ObservedStateTransition(
            state="ITEM_EXPOSURE",
            action="SKIP",
            next_state="ITEM_EXPOSURE",
            timestamp=NOW,
            item_id="item-1",
        )

        encoded = serialize_observed_transitions(
            (transition,),
            source="historical_import",
        )

        self.assertEqual(
            observed_transitions_from_text(encoded),
            (transition,),
        )


if __name__ == "__main__":
    unittest.main()
