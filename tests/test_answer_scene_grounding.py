import unittest
from unittest.mock import patch

from pipeline.search import rag_chain as rag
from pipeline.storage import vector_store as vs


def frame(timestamp, **extra):
    return {
        "frame_id": f"chunk_001_frame_{timestamp}",
        "video_id": "video",
        "timestamp": timestamp,
        "object_labels": "cat",
        "score": 0.25,
        **extra,
    }


def wide_event(event_id=1):
    return {
        "id": event_id,
        "video_id": "video",
        "start_time": 2.0,
        "end_time": 58.56,
        "action": "탐색",
        "summary": "반려동물이 움직임",
    }


class AnswerSceneGroundingTests(unittest.TestCase):
    def test_global_top_events_are_not_mixed_in_when_frames_have_overlapping_events(self):
        overlapping = {**wide_event(7), "start_time": 12.0, "end_time": 20.0}
        with patch.object(rag, "get_behavior_events_overlapping", return_value=[overlapping]), \
                patch.object(rag, "get_top_behavior_events", return_value=[wide_event(99)]) as top:
            events = rag._get_relevant_behavior_events(
                query="쓰레기통 뒤진 장면 있어?",
                video_id="video",
                retrieved_frames=[frame(18.0)],
            )
        self.assertEqual([event["id"] for event in events], [7])
        top.assert_not_called()

    def test_global_top_events_are_used_when_nothing_overlaps(self):
        with patch.object(rag, "get_behavior_events_overlapping", return_value=[]), \
                patch.object(rag, "get_top_behavior_events", return_value=[wide_event(99)]):
            events = rag._get_relevant_behavior_events(
                query="쓰레기통 뒤진 장면 있어?",
                video_id="video",
                retrieved_frames=[frame(18.0)],
            )
        self.assertEqual([event["id"] for event in events], [99])

    def test_scene_range_matches_card_format(self):
        text = rag._format_scene_range(
            frame(18.0, playback_start=11.0, playback_end=23.0, representative_timestamp=18.0)
        )
        self.assertEqual(text, "00:11~00:23 (가장 비슷한 순간 00:18)")

    def test_scene_range_falls_back_to_timestamp_window(self):
        self.assertEqual(
            rag._format_scene_range(frame(18.0)),
            "00:15~00:23 (가장 비슷한 순간 00:18)",
        )

    def test_fallback_answer_uses_frame_ranges_not_wide_event_range(self):
        answer = rag._build_fallback_answer(
            query="쓰레기통 뒤진 장면 있어?",
            retrieved_frames=[frame(18.0, playback_start=15.0, playback_end=23.0)],
            behavior_events=[wide_event()],
        )
        self.assertIn("00:15~00:23", answer)
        self.assertNotIn("58", answer)
        self.assertNotIn("호기심", answer)

    def test_adjacent_scene_span_is_capped(self):
        results = [
            {
                "video_id": "video",
                "s3_key": f"videos/chunk_frame_{t}.jpg",
                "timestamp": float(t),
                "score": 0.2,
            }
            for t in (2, 5, 8, 11, 14, 17)
        ]
        scenes = vs._merge_adjacent_result_scenes(results)
        for scene in scenes:
            self.assertLessEqual(scene["scene_end"] - scene["scene_start"], 10.0)
        self.assertGreater(len(scenes), 1)


if __name__ == "__main__":
    unittest.main()
