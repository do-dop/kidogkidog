import json
import os
import unittest
from unittest.mock import MagicMock, patch

from pipeline.events import scene_change_detector as detector
from pipeline.events.behavior_event_extractor import extract_behavior_events


def frames(*seconds):
    return [{"timestamp": float(s), "frame_path": f"/tmp/f_{s}.jpg"} for s in seconds]


def changed(**overrides):
    body = {
        "changed": True,
        "object": "소파",
        "before": "소파 등받이가 세워져 있음",
        "after": "소파 등받이가 뒤로 젖혀져 있음",
        "summary": "소파가 뒤로 넘어졌어요",
        "confidence": 0.9,
    }
    return json.dumps({**body, **overrides}, ensure_ascii=False)


class SceneChangeDetectorTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SCENE_CHANGE_DETECTION", None)

    def test_does_nothing_without_a_key_or_when_switched_off(self):
        with patch.object(detector, "_call_gemini") as call:
            with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
                self.assertIsNone(detector.detect_scene_change(frames(0, 2)))
            with patch.dict(os.environ, {"SCENE_CHANGE_DETECTION": "0"}):
                self.assertIsNone(detector.detect_scene_change(frames(0, 2)))
            call.assert_not_called()

    def test_no_change_costs_a_single_call(self):
        with patch.object(detector, "_call_gemini", return_value=json.dumps({"changed": False})) as call:
            self.assertIsNone(detector.detect_scene_change(frames(0, 2, 4, 6)))
        self.assertEqual(call.call_count, 1)

    def test_two_frames_give_a_scene_event(self):
        with patch.object(detector, "_call_gemini", return_value=changed()):
            event = detector.detect_scene_change(frames(18.52, 20.52))

        self.assertEqual((event["start_time"], event["end_time"]), (18.52, 20.52))
        self.assertEqual(event["subject_type"], "scene")
        self.assertEqual(event["target_object"], "소파")
        self.assertEqual(event["summary"], "소파가 뒤로 넘어졌어요")
        self.assertEqual(event["evidence"], ["18.52초: 소파 등받이가 세워져 있음", "20.52초: 소파 등받이가 뒤로 젖혀져 있음"])

    def test_time_is_narrowed_by_pixel_change_without_extra_model_calls(self):
        # 20.52 -> 22.52 사이에서 화면이 가장 크게 바뀐다.
        scores = {("/tmp/f_18.52.jpg", "/tmp/f_20.52.jpg"): 4.0,
                  ("/tmp/f_20.52.jpg", "/tmp/f_22.52.jpg"): 60.0,
                  ("/tmp/f_22.52.jpg", "/tmp/f_24.52.jpg"): 2.0}
        with patch.object(detector, "_call_gemini", return_value=changed()) as call, \
             patch.object(detector, "_pixel_change", side_effect=lambda a, b: scores[(a, b)]):
            event = detector.detect_scene_change(frames(18.52, 20.52, 22.52, 24.52))

        self.assertEqual((event["start_time"], event["end_time"]), (20.52, 22.52))
        self.assertEqual(call.call_count, 1)
        # 설명은 구간 전체(첫·끝 프레임) 비교에서 온 것이라 증거 시각도 구간 전체를 가리킨다.
        self.assertTrue(event["evidence"][0].startswith("18.52초"))
        self.assertTrue(event["evidence"][1].startswith("24.52초"))

    def test_unreadable_frames_fall_back_to_the_whole_window(self):
        with patch.object(detector, "_call_gemini", return_value=changed()), \
             patch.object(detector, "_pixel_change", return_value=None):
            event = detector.detect_scene_change(frames(18.52, 20.52, 22.52, 24.52))

        self.assertEqual((event["start_time"], event["end_time"]), (18.52, 24.52))

    def test_pixel_change_measures_real_images(self):
        import tempfile

        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            same_a, same_b, other = (f"{folder}/{name}.jpg" for name in ("a", "b", "c"))
            cv2.imwrite(same_a, np.zeros((90, 160), dtype="uint8"))
            cv2.imwrite(same_b, np.zeros((90, 160), dtype="uint8"))
            cv2.imwrite(other, np.full((90, 160), 200, dtype="uint8"))

            self.assertLess(detector._pixel_change(same_a, same_b), 1.0)
            self.assertGreater(detector._pixel_change(same_a, other), 100.0)
            self.assertIsNone(detector._pixel_change(same_a, f"{folder}/missing.jpg"))

    def test_failed_or_invalid_responses_return_none_without_raising(self):
        for response in (None, "not json", "[]", json.dumps({"changed": True}), changed(summary="")):
            with patch.object(detector, "_call_gemini", return_value=response):
                self.assertIsNone(detector.detect_scene_change(frames(0, 2)), response)

    def test_code_fenced_json_is_accepted(self):
        with patch.object(detector, "_call_gemini", return_value="```json\n" + changed() + "\n```"):
            self.assertIsNotNone(detector.detect_scene_change(frames(0, 2)))

    def test_rate_limit_is_retried_once_then_skipped(self):
        response = MagicMock(status_code=429)
        with patch.object(detector.httpx, "post", return_value=response) as post, \
             patch.object(detector.time, "sleep") as sleep, \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_bytes", return_value=b"jpeg"):
            self.assertIsNone(detector._call_gemini(frames(0, 2)))

        self.assertEqual(post.call_count, detector.MAX_ATTEMPTS)
        sleep.assert_called_once()
        # 키는 URL이 아니라 헤더로만 보낸다.
        self.assertNotIn("test-key", post.call_args.args[0])
        self.assertEqual(post.call_args.kwargs["headers"], {"x-goog-api-key": "test-key"})


class ExtractionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.frames = [
            {"video_id": "v", "timestamp": float(s), "motion_start": 0.0, "motion_end": 10.0,
             "segment_id": 0, "frame_path": f"/tmp/v_{s}.jpg"}
            for s in range(0, 11, 2)
        ]
        self.pet = {"action": "반려동물이 소파 쪽으로 다가감", "summary": "반려동물이 소파 쪽으로 다가갔어요",
                    "start_time": 0.0, "end_time": 4.0}

    def _run(self, analyzed, scene):
        with patch("pipeline.events.behavior_event_extractor._can_use_vlm", return_value=True), \
             patch("pipeline.events.behavior_event_extractor._analyze_segment_with_vlm", return_value=analyzed), \
             patch.object(detector, "detect_scene_change", return_value=scene) as detect:
            return extract_behavior_events(indexed_frames=self.frames, video_id="v"), detect

    def test_scene_event_is_added_next_to_the_pet_events(self):
        scene = {"start_time": 4.0, "end_time": 6.0, "subject_type": "scene", "target_object": "소파",
                 "action": "소파가 넘어졌어요", "summary": "소파가 넘어졌어요", "evidence": []}
        events, _ = self._run([self.pet], scene)

        by_subject = {event["subject"]: event for event in events}
        self.assertEqual(set(by_subject), {"반려동물", "소파"})
        self.assertEqual(by_subject["소파"]["summary"], "소파가 넘어졌어요")

    def test_no_scene_change_leaves_pet_events_untouched(self):
        events, detect = self._run([self.pet], None)

        self.assertEqual([event["subject"] for event in events], ["반려동물"])
        detect.assert_called_once()

    def test_scene_event_is_added_even_when_the_main_analysis_failed(self):
        scene = {"start_time": 4.0, "end_time": 6.0, "subject_type": "scene", "target_object": "소파",
                 "action": "소파가 넘어졌어요", "summary": "소파가 넘어졌어요", "evidence": []}
        events, _ = self._run(None, scene)

        self.assertIn("소파", [event["subject"] for event in events])

    def test_already_reported_scene_change_is_not_asked_again(self):
        from_main = {"subject_type": "scene", "target_object": "소파", "action": "소파가 넘어짐",
                     "summary": "소파가 넘어졌어요", "start_time": 4.0, "end_time": 6.0}
        events, detect = self._run([self.pet, from_main], None)

        detect.assert_not_called()
        self.assertIn("소파", [event["subject"] for event in events])


if __name__ == "__main__":
    unittest.main()
