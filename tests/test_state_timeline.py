import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.search.state_timeline import classify_state_frames, transitions_from_states


class StateTimelineTests(unittest.TestCase):
    def test_conflicting_window_boundary_labels_become_unknown(self):
        def fake_frames(video, start, end, step):
            return [{"time": start, "jpeg": b""}, {"time": end, "jpeg": b""}]

        def fake_model(prompt, frames):
            state = "before" if frames[0]["time"] == 0 else "after"
            return {"labels": [{"index": i, "state": state, "subject_visible": True}
                               for i in range(len(frames))]}

        with patch("pipeline.search.state_timeline.video_duration", return_value=8.0), \
             patch("pipeline.search.state_timeline.sample_frames", side_effect=fake_frames):
            result = classify_state_frames(Path("unused.mp4"), {}, fake_model)
        by_time = {row["time"]: row["state"] for row in result["labels"]}
        self.assertEqual(by_time[4.0], "unknown")

    def test_continuous_after_state_is_one_event_until_visible_return(self):
        labels = [
            {"time": 0.0, "state": "before"},
            {"time": 0.25, "state": "after"},
            {"time": 0.5, "state": "after"},
            {"time": 0.75, "state": "unknown"},
            {"time": 1.0, "state": "after"},
            {"time": 1.25, "state": "before"},
            {"time": 1.5, "state": "after"},
        ]
        result = transitions_from_states(labels)
        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["events"][1]["reset_time"], 1.25)

    def test_after_state_at_video_start_stays_unarmed(self):
        labels = [
            {"time": 0.0, "state": "after"},
            {"time": 0.25, "state": "after"},
            {"time": 0.5, "state": "before"},
            {"time": 0.75, "state": "after"},
        ]
        result = transitions_from_states(labels)
        self.assertEqual(result["unarmed_after_times"], [0.0])
        self.assertEqual(len(result["events"]), 1)


if __name__ == "__main__":
    unittest.main()
