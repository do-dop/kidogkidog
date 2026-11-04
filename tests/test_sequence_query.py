import sys
import unittest
from unittest.mock import MagicMock, patch

from pipeline.search.sequence_query import is_sequence_query, run_sequence_query


def event(event_id, chunk, second, summary):
    return {
        "id": event_id,
        "video_id": "dog-video",
        "start_time": second,
        "end_time": second + 4,
        "summary": summary,
        "confidence": 0.8,
        "source_frames": [{
            "frame_id": f"frame-{event_id}",
            "s3_key": f"frames/dog-video/petcam_20261104_090000_{chunk:03d}_frame_{second}.jpg",
            "timestamp": second,
        }],
    }


class SequenceQueryTests(unittest.TestCase):
    def test_classifies_sequence_question(self):
        self.assertTrue(is_sequence_query("강아지가 무엇을 한 다음에 쉬었어?"))
        self.assertTrue(is_sequence_query("오늘 행동 순서 알려줘"))
        self.assertTrue(is_sequence_query("강아지 행동을 시간순으로 알려줘"))
        self.assertFalse(is_sequence_query("물을 몇 번 마셨어?"))

    def test_orders_across_chunks_and_keeps_playable_evidence(self):
        events = [
            event(1, 1, 2, "잠을 잠"),
            event(2, 0, 40, "물을 마심"),
        ]
        result = run_sequence_query("무슨 순서로 행동했어?", video_id="dog-video", events=events)

        self.assertEqual([item["id"] for item in result["behavior_events"]], [2, 1])
        self.assertEqual([item["source_event_id"] for item in result["results"]], [2, 1])
        self.assertIn("09:00:40", result["answer"])
        self.assertIn("09:01:02", result["answer"])
        self.assertEqual(result["total_event_count"], 2)

    def test_date_filter_excludes_unknown_timestamp(self):
        unknown = event(3, 0, 20, "미확인")
        unknown["source_frames"] = []
        result = run_sequence_query("행동 순서", recording_date="2026-11-05", events=[event(1, 0, 10, "움직임"), unknown])

        self.assertEqual(result["behavior_events"], [])
        self.assertIn("확인할 수 없습니다", result["answer"])

    def test_four_event_demo_needs_larger_result_limit(self):
        events = [
            event(4, 1, 8, "다시 나타남"),
            event(2, 0, 20, "소파 위에 있음"),
            event(1, 0, 5, "바닥을 달림"),
            event(3, 1, 2, "소파에서 내려옴"),
        ]
        question = "강아지 행동을 시간순으로 알려줘"
        full = run_sequence_query(question, events=events, top_k=10)
        short = run_sequence_query(question, events=events, top_k=3)

        self.assertEqual([item["id"] for item in full["behavior_events"]], [1, 2, 3, 4])
        self.assertEqual([item["source_event_id"] for item in full["evidence_items"]], [1, 2, 3, 4])
        self.assertEqual(full["total_event_count"], 4)
        self.assertEqual(len(short["behavior_events"]), 3)
        self.assertIn("4개 중 앞의 3개", short["answer"])


class SequenceRoutingTests(unittest.TestCase):
    """run_rag_query가 집계형·이벤트 근거 질문·시퀀스형을 올바른 순서로 가르는지 확인한다."""

    @classmethod
    def setUpClass(cls):
        global rag
        from pipeline.search import rag_chain as rag

    def test_sequence_question_skips_frame_search_and_reads_full_timeline(self):
        events = [event(2, 0, 40, "물을 마심"), event(1, 1, 2, "잠을 잠")]
        # CLIP/torch를 불러오지 않도록 vector_store를 가짜 모듈로 바꾼다. 시퀀스 경로가 검색을 건너뛰어야 한다.
        fake_vector_store = MagicMock()
        with patch("db.behavior_events.get_behavior_events_for_aggregation", return_value=events), \
                patch.dict(sys.modules, {"pipeline.storage.vector_store": fake_vector_store}):
            result = rag.run_rag_query("오늘 강아지가 어떤 순서로 행동했어?", video_id="dog-video", top_k=5)

        fake_vector_store.search_with_query_expansion.assert_not_called()
        self.assertEqual(result["query_type"], "sequence")
        self.assertEqual([item["id"] for item in result["behavior_events"]], [2, 1])

    def test_count_question_stays_aggregate_even_with_sequence_words(self):
        with patch.object(rag, "_run_aggregate_query", return_value={"aggregate_result": None}) as aggregate, \
                patch.object(rag, "run_sequence_query") as sequence:
            rag.run_rag_query("물 마시고 나서 몇 번 먹었어?", video_id="dog-video")

        aggregate.assert_called_once()
        sequence.assert_not_called()

    def test_suggested_question_keeps_its_source_event(self):
        source = {"id": 7, "video_id": "dog-video", "start_time": 1.0, "end_time": 5.0}
        with patch.object(rag, "_resolve_source_event", return_value=source), \
                patch.object(rag, "_run_event_grounded_query", return_value={"answer": "grounded"}) as grounded, \
                patch.object(rag, "run_sequence_query") as sequence:
            result = rag.run_rag_query("행동 순서를 알려줘", video_id="dog-video", source_event_id=7)

        grounded.assert_called_once()
        sequence.assert_not_called()
        self.assertEqual(result["answer"], "grounded")


if __name__ == "__main__":
    unittest.main()
