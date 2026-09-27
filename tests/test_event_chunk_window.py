import unittest
from unittest.mock import patch

from pipeline import rag_chain


def frame(timestamp, chunk="petcam_20260926_173123_000", video_id="IMG_8450_2"):
    return {
        "video_id": video_id,
        "timestamp": timestamp,
        "s3_key": f"frames/{video_id}/{chunk}_frame_{timestamp:.2f}.jpg",
    }


class EventChunkWindowTests(unittest.TestCase):
    def setUp(self):
        self.event = {
            "id": 3,
            "video_id": "IMG_8450_2",
            "start_time": 4.5,
            "end_time": 9.0,
            "confidence": 0.8,
            "source_frames": [frame(t) for t in (4.5, 6.5, 8.5)],
        }

    def test_same_local_time_in_other_chunk_is_rejected(self):
        candidates = [frame(2, "petcam_20260926_173123_001"), frame(6.5), frame(8.5)]
        self.assertEqual(
            rag_chain._filter_frames_to_event_window(candidates, self.event),
            candidates[1:],
        )

    def test_same_chunk_number_from_other_run_or_video_is_rejected(self):
        candidates = [
            frame(6.5, "petcam_20260927_173123_000"),
            frame(6.5, video_id="other_video"),
            frame(6.5),
        ]
        self.assertEqual(rag_chain._filter_frames_to_event_window(candidates, self.event), candidates[2:])

    def test_margin_and_candidate_order_are_preserved(self):
        candidates = [frame(t) for t in (12, 6.5, 1.5, 1.49, 12.01)]
        self.assertEqual(rag_chain._filter_frames_to_event_window(candidates, self.event), candidates[:3])

    def test_long_event_still_rejects_other_chunk(self):
        event = {**self.event, "start_time": 12, "end_time": 66.5}
        candidates = [frame(34), frame(34, "petcam_20260926_173123_001"), frame(66)]
        self.assertEqual(rag_chain._filter_frames_to_event_window(candidates, event), [candidates[0], candidates[2]])

    def test_storage_path_and_frame_id_identify_same_chunk(self):
        stem = "petcam_20260926_173123_000"
        candidates = [
            {"frame_path": f"/app/pipeline/frames/IMG_8450_2/{stem}_frame_6.50.jpg", "timestamp": 6.5},
            {"frame_id": f"IMG_8450_2__{stem}_frame_8.50", "timestamp": 8.5},
        ]
        self.assertEqual(rag_chain._filter_frames_to_event_window(candidates, self.event), candidates)
        # FFmpeg fallback uses sequence numbers instead of timestamps in filenames.
        self.assertEqual(rag_chain._frame_chunk_stem({"s3_key": f"frames/x/{stem}_frame_000001.jpg"}), stem)

    def test_unknown_or_ambiguous_chunk_is_not_time_matched(self):
        for sources in ([], [{"timestamp": 6.5}], [frame(6.5), frame(6.5, "other_chunk")]):
            with self.subTest(sources=sources):
                self.assertEqual(rag_chain._filter_frames_to_event_window([frame(6.5)], {**self.event, "source_frames": sources}), [])
        self.assertEqual(rag_chain._filter_frames_to_event_window([{"timestamp": 6.5}], self.event), [])

    def test_missing_times_do_not_bypass_chunk_check(self):
        event = {**self.event, "start_time": None}
        candidates = [frame(6.5), frame(2, "petcam_20260926_173123_001")]
        self.assertEqual(rag_chain._filter_frames_to_event_window(candidates, event), candidates[:1])

    def test_empty_filtered_search_uses_existing_source_frame_fallback(self):
        with (
            patch.object(rag_chain, "get_indexed_frames_in_time_window", return_value=[frame(2, "petcam_20260926_173123_001")]),
            patch.object(rag_chain, "search_within_frames", return_value=[]) as search,
            patch.object(rag_chain, "build_prompt_hint", return_value=""),
            patch.object(rag_chain, "_generate_answer_with_langchain", return_value="answer"),
        ):
            result = rag_chain._run_event_grounded_query("급식기", self.event, 3, None)
        search.assert_called_once_with("급식기", [], top_k=3)
        self.assertEqual([item["timestamp"] for item in result["results"]], [4.5, 6.5, 8.5])
        self.assertEqual(result["behavior_events"], [self.event])
        self.assertTrue(all(item["clip_similarity"] is None for item in result["results"]))
        self.assertTrue(all(item["retrieval_source"] == "source_frames_fallback" for item in result["results"]))


if __name__ == "__main__":
    unittest.main()
