import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.search.video_aggregate import (
    analyze_video_count, challenge_overlapping_tracks, challenge_uncertain_evidence,
    parse_count_intent, reduce_transitions, repair_failed_claims,
    resolve_subject_tracks, timeline_windows, verify_resets,
)


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
    def test_fixed_intent_skips_stochastic_question_parsing(self):
        intent = {key: key for key in (
            "subject", "action", "target", "before_state", "after_state", "exclusion")}
        with patch("pipeline.search.video_aggregate.parse_count_intent",
                   side_effect=AssertionError("must use fixed intent")), \
             patch("pipeline.search.video_aggregate.collect_candidates", return_value=([], [])), \
             patch("pipeline.search.video_aggregate.verify_candidates", return_value=[]), \
             patch("pipeline.search.video_aggregate.check_claimed_evidence", return_value=[]), \
             patch("pipeline.search.video_aggregate.repair_failed_claims", return_value=[]):
            result = analyze_video_count(Path("unused.mp4"), "question", lambda *_: {},
                                         resolve_tracks=False, intent_override=intent)
        self.assertEqual(result["intent"], intent)

    def test_question_maps_to_observable_states(self):
        def fake_model(prompt, frames):
            self.assertIn("소파", prompt)
            self.assertFalse(frames)
            return {"subject": "강아지", "action": "올라가기", "target": "소파",
                    "before_state": "바닥에 몸이 있음", "after_state": "소파 좌석에 몸이 있음",
                    "exclusion": "앞발만 걸친 경우"}

        self.assertEqual(parse_count_intent("강아지가 소파에 몇 번 올라갔어?", fake_model)["target"], "소파")

    def test_front_paw_question_keeps_door_target_and_both_paws(self):
        def drifting_model(_prompt, _frames):
            return {"subject": "dog", "action": "lift one paw", "target": "front paw",
                    "before_state": "one paw on floor", "after_state": "one paw off floor",
                    "exclusion": "none"}

        intent = parse_count_intent(
            "강아지가 유리문 앞에서 앞발을 몇 번 들었다 내렸어? 뒷발이 바닥에 남아도 포함해.",
            drifting_model,
        )
        self.assertEqual(intent["target"], "glass door")
        self.assertIn("two front paws", intent["after_state"])
        self.assertIn("hind paws may remain", intent["after_state"])

    def test_brief_front_paw_action_gets_short_window_candidates(self):
        intent = {"subject": "dog", "action": "raise paws", "target": "door",
                  "before_state": "paws down", "after_state": "two front paws raised",
                  "exclusion": "continued pose"}
        scans = [([{"time": 4.75, "subject": "dog", "window": 0}], []),
                 ([{"time": 6.25, "subject": "dog", "window": 0}], [])]
        with patch("pipeline.search.video_aggregate.parse_count_intent", return_value=intent), \
             patch("pipeline.search.video_aggregate.collect_candidates", side_effect=scans) as collect, \
             patch("pipeline.search.video_aggregate.verify_candidates", return_value=[]), \
             patch("pipeline.search.video_aggregate.check_claimed_evidence", return_value=[]), \
             patch("pipeline.search.video_aggregate.repair_failed_claims", return_value=[]):
            result = analyze_video_count(Path("unused.mp4"), "앞발 몇 번 들었어?", lambda *_: {},
                                         resolve_tracks=False)
        self.assertEqual([row["time"] for row in result["candidates"]], [4.75, 6.25])
        self.assertEqual(collect.call_args_list[1].kwargs["core_seconds"], 2.0)

    def test_uncertain_candidate_needs_wider_pair_and_independent_check(self):
        candidate = item(5.0, "dog", None, 5.0, status="uncertain")
        calls = []

        def model(prompt, _frames):
            calls.append(prompt)
            if len(calls) == 1:
                return {"status": "confirmed", "subject_id": "dog", "before_time": 4.0,
                        "after_time": 5.5, "start_time": 4.0, "end_time": 5.5}
            return {"status": "confirmed", "before_state_visible": True,
                    "after_state_visible": True, "same_subject_visible": True}

        with patch("pipeline.search.video_aggregate.video_duration", return_value=10.0), \
             patch("pipeline.search.video_aggregate.sample_frames", return_value=[{"time": 4.0}]):
            repair_failed_claims(Path("unused.mp4"), {"after_state": "on sofa"}, [candidate], model)

        self.assertIn("uncertain", calls[0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(reduce_transitions([candidate])["confirmed_count"], 1)

    def test_uncertain_retry_without_independent_evidence_stays_uncertain(self):
        candidate = item(5.0, "dog", None, 5.0, status="uncertain")

        def model(prompt, _frames):
            if "wider clip" in prompt:
                return {"status": "confirmed", "subject_id": "dog", "before_time": 4.0,
                        "after_time": 5.5}
            return {"status": "uncertain", "before_state_visible": True,
                    "after_state_visible": True, "same_subject_visible": False}

        with patch("pipeline.search.video_aggregate.video_duration", return_value=10.0), \
             patch("pipeline.search.video_aggregate.sample_frames", return_value=[{"time": 4.0}]):
            repair_failed_claims(Path("unused.mp4"), {}, [candidate], model)

        self.assertEqual(reduce_transitions([candidate])["confirmed_count"], 0)
        self.assertEqual(reduce_transitions([candidate])["uncertain_count"], 1)

    def test_enlarged_uncertain_review_requires_all_three_visible_claims(self):
        import cv2
        import numpy as np

        candidate = item(5.0, "dog", 4.0, 5.0, status="uncertain")
        candidate["clip"] = [4.5, 5.5]  # Wider review may recover an earlier before-state frame.
        jpeg = cv2.imencode(".jpg", np.zeros((80, 120, 3), dtype=np.uint8))[1].tobytes()
        frames = [{"time": 4.0, "jpeg": jpeg}, {"time": 5.0, "jpeg": jpeg}]

        with patch("pipeline.search.video_aggregate.video_duration", return_value=10.0), \
             patch("pipeline.search.video_aggregate.sample_frames", return_value=frames):
            challenge_uncertain_evidence(Path("unused.mp4"), {}, [candidate],
                                         lambda _prompt, shown: {
                                             "status": "confirmed", "before_state_visible": True,
                                             "after_state_visible": True, "same_subject_visible": len(shown) == 4,
                                         })
        self.assertEqual(reduce_transitions([candidate])["confirmed_count"], 1)

    def test_track_resolution_uses_before_and_after_for_two_animals(self):
        first = item(2.0, "black dog", 1.0, 2.0)
        second = item(5.0, "black dog", 4.0, 5.0)
        captured = []

        def model(prompt, frames):
            self.assertIn("before/after", prompt)
            captured.extend(frame["time"] for frame in frames)
            return {"assignments": [
                {"index": 0, "track_id": "animal_1", "certain": True},
                {"index": 1, "track_id": "animal_2", "certain": True},
            ]}

        with patch("pipeline.search.video_aggregate.sample_frames",
                   side_effect=lambda _video, start, _end, _step: [{"time": start}]):
            resolve_subject_tracks(Path("unused.mp4"), [first, second], model)

        self.assertEqual(captured, [1.0, 2.0, 4.0, 5.0])
        self.assertEqual(reduce_transitions([first, second])["confirmed_count"], 2)

    def test_overlapping_same_track_requires_visible_second_animal(self):
        import cv2
        import numpy as np

        first = item(2.0, "black dog", 0.5, 3.0)
        second = item(4.25, "black dog", 3.0, 5.0)
        for event in (first, second):
            event["observation"]["track_id"] = "animal_1"
        jpeg = cv2.imencode(".jpg", np.zeros((80, 120, 3), dtype=np.uint8))[1].tobytes()

        def frame(_video, start, end, _step):
            return [{"time": start, "jpeg": jpeg}, {"time": end, "jpeg": jpeg}]

        def distinct(_prompt, frames):
            self.assertGreater(len(frames), 2)
            return {"same_animal": False, "certain": True,
                    "two_animals_simultaneously_visible": True}

        with patch("pipeline.search.video_aggregate.sample_frames", side_effect=frame):
            challenge_overlapping_tracks(Path("unused.mp4"), [first, second], distinct)
        self.assertEqual(second["observation"]["track_id"], "animal_2")
        self.assertEqual(reduce_transitions([first, second])["confirmed_count"], 2)

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
            item(9.0, "black dog", 8.25, 9.25, reset=8.0, start=9.0, end=9.5),
        ])
        self.assertEqual(result["confirmed_count"], 2)
        self.assertEqual(result["uncertain_count"], 1)

    def test_verified_reset_outweighs_overlong_model_episode_span(self):
        result = reduce_transitions([
            item(4.5, "dog", 3.25, 4.5, start=3.25, end=6.0),
            item(6.25, "dog", 5.0, 6.25, reset=5.0, start=5.0, end=7.75),
        ])
        self.assertEqual(result["confirmed_count"], 2)

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

    def test_independently_checked_second_before_frame_proves_new_entry(self):
        first = item(2.75, "dog", 1.5, 2.75)
        second = item(21.25, "dog", 19.25, 21.25)
        for event in (first, second):
            event["observation"]["track_id"] = "animal_1"
        second["repair_evidence_check"] = {
            "status": "confirmed", "before_state_visible": True,
            "after_state_visible": True, "same_subject_visible": True,
        }
        second["reset_check"] = {"reset_visible": False, "same_subject": False}
        self.assertEqual(reduce_transitions([first, second])["confirmed_count"], 2)

        second["observation"]["reset_time"] = None
        with patch("pipeline.search.video_aggregate.video_duration", return_value=24.0):
            verify_resets(Path("unused.mp4"), {}, [first, second],
                          lambda _prompt, _frames: self.fail("checked before-frame needs no extra model call"))
        self.assertEqual(second["observation"]["reset_time"], 19.25)

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

    def test_unchecked_reset_claim_is_cleared_before_cross_event_check(self):
        first = item(4.5, "dog", 3.25, 4.5)
        second = item(4.75, "dog", 3.5, 4.75, reset=4.6)
        for event in (first, second):
            event["observation"]["track_id"] = "animal_1"
        with patch("pipeline.search.video_aggregate.video_duration", return_value=12.0):
            verify_resets(Path("unused.mp4"), {}, [first, second],
                          lambda prompt, frames: self.fail("overlapping events need no reset API call"))
        self.assertIsNone(second["observation"]["reset_time"])
        self.assertEqual(reduce_transitions([first, second])["confirmed_count"], 1)

    def test_clip_starting_in_after_state_is_unresolved_not_zero_events(self):
        boundary = item(0.375, "dog", 0.0, 0.25, status="rejected")
        boundary["clip"] = [0.0, 1.875]
        boundary["evidence_check"] = {"status": "rejected",
                                      "before_state_visible": False,
                                      "after_state_visible": True}
        result = reduce_transitions([boundary])
        self.assertEqual(result["confirmed_count"], 0)
        self.assertEqual(result["uncertain_count"], 1)
        self.assertFalse(result["rejected"])


if __name__ == "__main__":
    unittest.main()
