import unittest
from unittest.mock import patch

from pipeline.events.behavior_event_extractor import (
    extract_behavior_events,
    _get_group_end,
    _get_group_start,
    _normalize_behavior_event,
    _split_long_group,
)
from pipeline.search.aggregate_query import count_episodes, event_matches_action


class BehaviorEventWindowingTests(unittest.TestCase):
    def setUp(self):
        self.frames = [
            {
                "video_id": "dog_escape",
                "timestamp": float(second),
                "motion_start": 0.0,
                "motion_end": 56.0,
                "segment_id": 0,
            }
            for second in range(0, 57, 2)
        ]

    def test_long_segment_is_split_with_overlap(self):
        windows = _split_long_group(self.frames)
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(
            window[0]["motion_end"] - window[0]["motion_start"] <= 12.0
            for window in windows
        ))
        self.assertEqual(windows[0][0]["motion_start"], 0.0)
        self.assertEqual(windows[1][0]["motion_start"], 10.0)
        self.assertEqual(windows[-1][0]["motion_end"], 56.0)

    def test_event_bounds_are_clamped_to_window_and_evidence(self):
        window = _split_long_group(self.frames)[0]
        event = _normalize_behavior_event(
            event={
                "action": "반려동물이 울타리에 올라탐",
                "start_time": 4.0,
                "end_time": 16.0,
            },
            group=window,
            representative_frames=window,
        )
        self.assertEqual((event["start_time"], event["end_time"]), (4.0, 12.0))
        self.assertTrue(all(
            4.0 <= frame["timestamp"] <= 12.0 for frame in event["source_frames"]
        ))

    def test_reviewed_two_episodes_survive_overlapping_windows(self):
        # Offline oracle: only windowing and merging are checked here, not VLM accuracy.
        frames = [
            {**frame, "motion_end": 36.0}
            for frame in self.frames if frame["timestamp"] < 36.0
        ]
        reviewed = [(4.0, 16.0), (22.0, 31.0)]

        def reviewed_events_for_window(*, all_segment_frames, **_kwargs):
            start = _get_group_start(all_segment_frames)
            end = _get_group_end(all_segment_frames)
            return [
                {
                    "action": "반려동물이 울타리에 올라탐",
                    "start_time": max(start, reviewed_start),
                    "end_time": min(end, reviewed_end),
                }
                for reviewed_start, reviewed_end in reviewed
                if max(start, reviewed_start) < min(end, reviewed_end)
            ]

        with patch("pipeline.events.behavior_event_extractor._can_use_vlm", return_value=True), \
             patch("pipeline.events.behavior_event_extractor._analyze_segment_with_vlm", side_effect=reviewed_events_for_window):
            events = extract_behavior_events(indexed_frames=frames, video_id="dog_escape")

        matched = [event for event in events if event_matches_action(event, "gate_climb")]
        episodes = count_episodes(matched)
        self.assertEqual(
            [(episode["start_time"], episode["end_time"]) for episode in episodes],
            reviewed,
        )


if __name__ == "__main__":
    unittest.main()
