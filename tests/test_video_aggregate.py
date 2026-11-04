import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.search.video_aggregate import parse_count_intent, reduce_transitions, timeline_windows, verify_resets


def item(time, subject, before, after, *, reset=None, start=None, end=None, status="confirmed"):
    return {
        "candidate": {"time": time},
        "clip": [max(0, time - 2), time + 2],
        "observation": {
            "subject_id": subject,
            "before_time": before,
            "after_time": after,
            "reset_time": reset,
            "start_time": start if start is not None else time,
            "end_time": end if end is not None else time + 0.3,
            "status": status,
        },
    }


class VideoAggregateTests(unittest.TestCase):
    def test_question_maps_to_observable_states(self):
        def fake_model(prompt, frames):
            self.assertIn("소파", prompt)
            self.assertFalse(frames)
            return {"subject": "강아지", "action": "올라가기", "target": "소파",
                    "before_state": "바닥에 몸이 있음", "after_state": "소파 좌석에 몸이 있음",
                    "exclusion": "앞발만 걸친 경우"}

        self.assertEqual(parse_count_intent("강아지가 소파에 몇 번 올라갔어?", fake_model)["target"], "소파")

    def test_windows_cover_entire_video_with_context(self):
        windows = timeline_windows(14.2, core_seconds=6)
        self.assertEqual([(w["core_start"], w["core_end"]) for w in windows],
                         [(0, 6), (6, 12), (12, 14.2)])
        self.assertLess(windows[1]["start"], 6)

    def test_two_animals_can_enter_close_together(self):
        result = reduce_transitions([
            item(19.0, "black dog", 18.5, 19.5),
            item(20.5, "white dog", 20.0, 21.0),
        ])
        self.assertEqual(result["confirmed_count"], 2)

    def test_same_animal_requires_visible_reset_for_second_entry(self):
        result = reduce_transitions([
            item(1.5, "black dog", 1.0, 2.0, start=1.5, end=3.0),
            item(5.0, "black dog", 4.75, 5.25, start=5.0, end=5.5),
            item(9.0, "black dog", 8.25, 9.25, reset=8.5, start=9.0, end=9.5),
        ])
        self.assertEqual(result["confirmed_count"], 2)
        self.assertEqual(result["uncertain_count"], 1)

    def test_overlap_is_duplicate_but_missing_before_state_is_uncertain(self):
        result = reduce_transitions([
            item(1.5, "dog", 1.0, 2.0, start=1.5, end=3.0),
            item(2.5, "dog", 2.1, 2.6, start=2.5, end=3.2),
            item(5.0, "dog", None, 5.25),
        ])
        self.assertEqual(result["confirmed_count"], 1)
        self.assertEqual(len(result["duplicates"]), 1)
        self.assertEqual(result["uncertain_count"], 1)

    def test_resolved_track_prevents_alias_from_becoming_another_animal(self):
        first = item(10.5, "black cat", 10.0, 10.5, start=10.5, end=12.0)
        second = item(11.5, "black and white cat", 10.75, 11.5, start=11.5, end=12.0)
        first["observation"]["track_id"] = "animal_1"
        second["observation"]["track_id"] = "animal_1"
        result = reduce_transitions([first, second])
        self.assertEqual(result["confirmed_count"], 1)
        self.assertEqual(len(result["duplicates"]), 1)

    def test_between_event_frames_can_prove_reset_outside_verification_clip(self):
        first = item(2.0, "dog", 1.5, 2.0, start=2.0, end=2.4)
        second = item(10.0, "dog", 9.5, 10.0, start=10.0, end=10.4)
        for event in (first, second):
            event["observation"]["track_id"] = "animal_1"

        def confirm_reset(prompt, frames):
            self.assertIn("same animal", prompt.lower())
            self.assertEqual(frames, [{"time": 5.0}])
            return {"reset_visible": True, "same_subject": True, "reset_time": 5.0}

        with patch("pipeline.search.video_aggregate.video_duration", return_value=12.0), \
             patch("pipeline.search.video_aggregate.sample_frames", return_value=[{"time": 5.0}]):
            verified = verify_resets(Path("unused.mp4"), {"before_state": "floor"},
                                     [first, second], confirm_reset)

        result = reduce_transitions(verified)
        self.assertEqual(result["confirmed_count"], 2)
        self.assertEqual(result["confirmed"][1]["reset_time"], 5.0)

    def test_reset_rejection_keeps_repeat_uncertain(self):
        first = item(2.0, "dog", 1.5, 2.0, start=2.0, end=2.4)
        second = item(10.0, "dog", 9.5, 10.0, reset=9.6, start=10.0, end=10.4)
        for event in (first, second):
            event["observation"]["track_id"] = "animal_1"

        with patch("pipeline.search.video_aggregate.video_duration", return_value=12.0), \
             patch("pipeline.search.video_aggregate.sample_frames", return_value=[]):
            verified = verify_resets(
                Path("unused.mp4"), {}, [first, second],
                lambda prompt, frames: {"reset_visible": False, "same_subject": True,
                                        "reset_time": None})

        result = reduce_transitions(verified)
        self.assertEqual(result["confirmed_count"], 1)
        self.assertEqual(result["uncertain_count"], 1)


if __name__ == "__main__":
    unittest.main()
