import json
import unittest
from pathlib import Path

from pipeline.events.aggregate_event_reindex import events_from_count_result
from pipeline.search.aggregate_query import count_episodes, event_matches_action


RESULT = (
    Path(__file__).resolve().parents[1]
    / "notebooks/smarthome-bench-evidence/new-pipeline/ablation-cross-video/smartbench_0188/union-result.json"
)


class AggregateEventReindexTests(unittest.TestCase):
    def test_reviewed_fence_transitions_become_two_searchable_episodes(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        scenes = [
            {"start_time": 4.5, "s3_key": "frames/smartbench_0188/smartbench_0188_000_frame_4.50.jpg"},
            {"start_time": 24.02, "s3_key": "frames/smartbench_0188/smartbench_0188_000_frame_24.02.jpg"},
        ]
        events = events_from_count_result(result, "smartbench_0188", scenes)

        self.assertEqual(len(events), 2)
        self.assertEqual(len(count_episodes(events)), 2)
        self.assertTrue(all(event_matches_action(event, "gate_climb") for event in events))
        self.assertEqual([event["start_time"] for event in events], [3.25, 22.0])
        self.assertTrue(all(event["source_frames"] for event in events))

    def test_unresolved_identity_is_not_persisted_as_confirmed(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        result["track_status"] = "unresolved_error"
        with self.assertRaises(ValueError):
            events_from_count_result(result, "smartbench_0188", [{"start_time": 4.5, "s3_key": "frame.jpg"}])

    def test_empty_confirmation_cannot_erase_existing_events(self):
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        result["confirmed"] = []
        result["confirmed_count"] = 0
        with self.assertRaises(ValueError):
            events_from_count_result(result, "smartbench_0188", [{"start_time": 4.5, "s3_key": "frame.jpg"}])


if __name__ == "__main__":
    unittest.main()
