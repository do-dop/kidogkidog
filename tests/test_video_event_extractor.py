import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.events import video_event_extractor as vee


def frames(times, video_id="clip"):
    return [
        {"frame_id": f"{video_id}_petcam_20261119_000000_000_frame_{t:.2f}", "frame_path": f"/tmp/f_{t}.jpg",
         "s3_key": f"frames/{video_id}/f_{t}.jpg", "timestamp": float(t), "motion_start": float(t),
         "motion_end": float(t), "segment_id": 0, "video_id": video_id, "object_labels": []}
        for t in times
    ]


RAW = {
    "start_time": 2.0, "end_time": 6.0, "subject_type": "pet", "action": "강아지가 소파 쪽으로 걸어옴",
    "summary": "강아지가 왼쪽에서 소파 쪽으로 걸어와요", "appearance": "갈색 강아지", "count": 1,
    "location": "화면 왼쪽", "direction": "오른쪽", "objects": ["소파"], "evidence_description": "왼쪽에서 걸어 들어옴",
    "confidence": 0.9, "interestingness": 0.8, "action_type": "walk", "continues_previous": False,
}


class ParseTests(unittest.TestCase):
    def test_array_object_and_code_fence(self):
        self.assertEqual(len(vee.parse_events(json.dumps([RAW]))), 1)
        self.assertEqual(len(vee.parse_events(json.dumps({"events": [RAW, RAW]}))), 2)
        self.assertEqual(len(vee.parse_events("```json\n" + json.dumps([RAW]) + "\n```")), 1)

    def test_invalid_returns_none(self):
        self.assertIsNone(vee.parse_events("not json"))
        self.assertIsNone(vee.parse_events('"text"'))
        self.assertEqual(vee.parse_events("[]"), [])


class MappingTests(unittest.TestCase):
    def test_event_is_normalized_for_storage(self):
        events = vee._to_pipeline_events([RAW], frames(range(0, 10)), "clip")
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["subject"], "반려동물")
        self.assertTrue(event["action"].startswith("반려동물"))
        self.assertTrue(event["summary"].startswith("반려동물"))
        self.assertEqual((event["start_time"], event["end_time"]), (2.0, 6.0))
        self.assertEqual(event["evidence"][0], "2.00초: 왼쪽에서 걸어 들어옴")
        self.assertIn("외형 갈색 강아지", event["evidence"][1])
        self.assertEqual(event["source_frames"][0]["event_labels"]["action_type"], "walk")

    def test_scene_change_keeps_object_subject(self):
        raw = {**RAW, "subject_type": "scene", "target_object": "소파", "action": "소파가 넘어짐", "summary": "소파가 넘어졌어요"}
        event = vee._to_pipeline_events([raw], frames(range(0, 10)), "clip")[0]
        self.assertEqual(event["subject"], "소파")

    def test_interval_without_frame_snaps_to_nearest_frame(self):
        raw = {**RAW, "start_time": 3.2, "end_time": 3.8}
        event = vee._to_pipeline_events([raw], frames([0, 2, 4, 6]), "clip")[0]
        self.assertLessEqual(event["start_time"], 4.0 - 0.0)
        self.assertGreaterEqual(event["end_time"], 3.8)
        self.assertTrue(event["source_frames"])

    def test_far_from_any_frame_is_dropped(self):
        raw = {**RAW, "start_time": 20.0, "end_time": 22.0}
        self.assertEqual(vee._to_pipeline_events([raw], frames([0, 2]), "clip"), [])

    def test_bad_times_are_skipped(self):
        bad = [{**RAW, "start_time": "x"}, {**RAW, "start_time": 5, "end_time": 4}]
        self.assertEqual(vee._to_pipeline_events(bad, frames(range(10)), "clip"), [])


class ExtractTests(unittest.TestCase):
    def test_returns_none_when_call_fails(self):
        with patch.object(vee, "_call_gemini_video", return_value=None):
            self.assertIsNone(vee.extract_video_events("c.mp4", frames(range(10)), "clip"))

    def test_empty_model_answer_is_empty_list_not_failure(self):
        with patch.object(vee, "_call_gemini_video", return_value="[]"):
            self.assertEqual(vee.extract_video_events("c.mp4", frames(range(10)), "clip"), [])

    def test_sorted_by_interestingness_and_limited(self):
        raws = [{**RAW, "start_time": 0.0, "end_time": 3.0, "interestingness": 0.2},
                {**RAW, "start_time": 4.0, "end_time": 7.0, "action": "강아지가 앉음", "interestingness": 0.9}]
        with patch.object(vee, "_call_gemini_video", return_value=json.dumps(raws)):
            events = vee.extract_video_events("c.mp4", frames(range(10)), "clip", limit=1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_time"], 4.0)


class CallTests(unittest.TestCase):
    def _response(self, text):
        response = MagicMock(status_code=200)
        response.json.return_value = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
        return response

    def test_small_video_is_sent_inline_with_key_in_header(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            path = Path(d) / "c.mp4"
            path.write_bytes(b"xx")
            with patch.object(vee.httpx, "post", return_value=self._response("[]")) as post:
                self.assertEqual(vee._call_gemini_video(str(path)), "[]")
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "k")
        self.assertNotIn("key=", post.call_args.args[0])
        self.assertIn("inline_data", kwargs["json"]["contents"][0]["parts"][0])

    def test_large_video_uses_files_api_and_deletes_it(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"GEMINI_API_KEY": "k", "VIDEO_EVENT_INLINE_MAX_BYTES": "0"}):
            path = Path(d) / "c.mp4"
            path.write_bytes(b"xx")
            with patch.object(vee, "_upload_file", return_value={"name": "files/a", "uri": "u"}), \
                 patch.object(vee.httpx, "post", return_value=self._response("[]")) as post, \
                 patch.object(vee, "_delete_file") as delete:
                self.assertEqual(vee._call_gemini_video(str(path)), "[]")
        self.assertIn("file_data", post.call_args.kwargs["json"]["contents"][0]["parts"][0])
        delete.assert_called_once_with("files/a")

    def test_no_key_or_missing_file_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(vee._call_gemini_video("/nonexistent.mp4"))

    def test_retries_then_gives_up_without_raising(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            path = Path(d) / "c.mp4"
            path.write_bytes(b"xx")
            with patch.object(vee.httpx, "post", return_value=MagicMock(status_code=503)) as post, \
                 patch.object(vee.time, "sleep"):
                self.assertIsNone(vee._call_gemini_video(str(path)))
        self.assertEqual(post.call_count, vee.MAX_ATTEMPTS)


class DispatchTests(unittest.TestCase):
    FRAMES_RESULT = [{"action": "frames"}]

    def _run(self, source, video_result, video_path="c.mp4"):
        env = {} if source is None else {"BEHAVIOR_EVENT_SOURCE": source}
        with patch.dict(os.environ, env, clear=True), \
             patch.object(vee, "extract_behavior_events", return_value=self.FRAMES_RESULT) as fr, \
             patch.object(vee, "extract_video_events", return_value=video_result) as vid, \
             patch.object(vee, "_write_shadow") as shadow:
            out = vee.extract_events_for_chunk(frames([0, 1]), "clip", video_path)
        return out, fr, vid, shadow

    def test_default_is_frames_and_never_calls_video(self):
        out, fr, vid, _ = self._run(None, [{"action": "video"}])
        self.assertEqual(out, self.FRAMES_RESULT)
        vid.assert_not_called()

    def test_invalid_value_falls_back_to_frames(self):
        out, _, vid, _ = self._run("nonsense", [{"action": "video"}])
        self.assertEqual(out, self.FRAMES_RESULT)
        vid.assert_not_called()

    def test_video_mode_uses_video_result(self):
        out, fr, _, _ = self._run("video", [{"action": "video"}])
        self.assertEqual(out, [{"action": "video"}])
        fr.assert_not_called()

    def test_video_mode_empty_list_is_trusted(self):
        out, fr, _, _ = self._run("video", [])
        self.assertEqual(out, [])
        fr.assert_not_called()

    def test_video_mode_failure_falls_back_to_frames(self):
        out, fr, _, _ = self._run("video", None)
        self.assertEqual(out, self.FRAMES_RESULT)
        fr.assert_called_once()

    def test_both_returns_frames_and_stores_video_as_shadow(self):
        out, _, vid, shadow = self._run("both", [{"action": "video"}])
        self.assertEqual(out, self.FRAMES_RESULT)
        shadow.assert_called_once()

    def test_both_survives_video_exception(self):
        with patch.dict(os.environ, {"BEHAVIOR_EVENT_SOURCE": "both"}, clear=True), \
             patch.object(vee, "extract_behavior_events", return_value=self.FRAMES_RESULT), \
             patch.object(vee, "extract_video_events", side_effect=RuntimeError("boom")):
            out = vee.extract_events_for_chunk(frames([0, 1]), "clip", "c.mp4")
        self.assertEqual(out, self.FRAMES_RESULT)

    def test_missing_video_path_uses_frames(self):
        out, _, vid, _ = self._run("video", [{"action": "video"}], video_path=None)
        self.assertEqual(out, self.FRAMES_RESULT)
        vid.assert_not_called()

    def test_shadow_file_is_written(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"VIDEO_EVENT_SHADOW_DIR": d}):
            vee._write_shadow("clip", "/x/chunk_000.mp4", [{"action": "a"}])
            files = list(Path(d).glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text())["events"], [{"action": "a"}])


if __name__ == "__main__":
    unittest.main()
