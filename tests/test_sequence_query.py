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
        self.assertTrue(is_sequence_query("문을 나간 다음 뭐했어?"))
        self.assertTrue(is_sequence_query("숲길을 걷기 전에 뭐했어?"))
        self.assertFalse(is_sequence_query("물을 몇 번 마셨어?"))

    def test_orders_across_chunks_and_keeps_playable_evidence(self):
        events = [
            event(1, 1, 2, "잠을 잠"),
            event(2, 0, 40, "물을 마심"),
        ]
        result = run_sequence_query("무슨 순서로 행동했어?", video_id="dog-video", events=events)

        self.assertEqual([item["id"] for item in result["behavior_events"]], [2, 1])
        self.assertEqual([item["source_event_id"] for item in result["results"]], [2, 1])
        self.assertIn("먼저 물을 마심", result["answer"])
        self.assertIn("그다음에는 잠을 잠", result["answer"])
        self.assertEqual(result["total_event_count"], 2)

    def test_date_filter_excludes_unknown_timestamp(self):
        unknown = event(3, 0, 20, "미확인")
        unknown["source_frames"] = []
        result = run_sequence_query("행동 순서", recording_date="2026-11-05", events=[event(1, 0, 10, "움직임"), unknown])

        self.assertEqual(result["behavior_events"], [])
        self.assertIn("분석된 행동이 없어요", result["answer"])

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
        self.assertIn("전체 4개 장면 중 처음 3개만", short["answer"])
        self.assertIn("이어서 소파에서 내려옴", short["answer"])

    def test_sequence_answer_reads_as_prose_without_changing_event_descriptions(self):
        events = [
            event(1, 0, 2, "사람과 함께 놀고 있는 모습입니다."),
            event(2, 1, 2, "사람 가까이에서 관심을 보이는 장면으로 보임."),
            event(3, 2, 2, "주인을 향해 소리를 내는 모습이다."),
        ]
        result = run_sequence_query("강아지가 어떤 순서로 행동했어?", events=events, top_k=3)

        self.assertIn("먼저 사람과 함께 놀고 있는 모습이 보여요.", result["answer"])
        self.assertIn("그다음에는 사람 가까이에서 관심을 보이는 장면으로 보여요.", result["answer"])
        self.assertIn("이어서 주인을 향해 소리를 내는 모습이 보여요.", result["answer"])
        self.assertNotIn("행동 이벤트", result["answer"])
        self.assertEqual([item["id"] for item in result["behavior_events"]], [1, 2, 3])

    def test_unreviewed_note_style_endings_read_naturally(self):
        cases = {
            "반려동물이 편안하게 누워 있는 모습이 관찰됨.": "반려동물이 편안하게 누워 있는 모습이 관찰돼요",
            "반려동물이 물 그릇에서 물을 마시는 모습을 보임.": "반려동물이 물 그릇에서 물을 마시는 모습을 보여요",
            "반려동물이 물 그릇에 다가가 물을 마시는 행동을 하고 있음.": "반려동물이 물 그릇에 다가가 물을 마시는 행동을 하고 있어요",
            "반려동물이 사료를 먹으며 식사 중인 상황으로 보인다.": "반려동물이 사료를 먹으며 식사 중인 상황으로 보여요",
            "반려동물이 사료 그릇에서 먹이를 섭취하고 있는 모습입니다.": "반려동물이 사료 그릇에서 먹이를 섭취하고 있는 모습이 보여요",
            "음식을 먹는 모습이며, 이는 건강을 유지하는 데 중요한 행동임.": "음식을 먹는 모습이며, 이는 건강을 유지하는 데 중요한 행동이에요",
            "반려동물이 6.0초 동안 움직임을 보인 것으로 추정되는 구간입니다.": "반려동물이 6.0초 동안 움직임을 보인 것으로 보여요",
            "소파에서 움직임": "소파에서 움직임",
        }
        # 말투만 확인하므로 같은 행동이 묶이지 않도록 이벤트 사이를 충분히 띄운다.
        events = [event(i, i * 5, 2, text) for i, text in enumerate(cases, start=1)]
        result = run_sequence_query("행동 순서", events=events, top_k=10)
        for expected in cases.values():
            self.assertIn(expected + ".", result["answer"])
        self.assertNotIn("움직이에요", result["answer"])

    def test_consecutive_same_action_is_merged_into_one_step(self):
        events = [
            event(1, 0, 2, "반려동물이 침대에서 편안하게 누워 있는 모습이 관찰됨."),
            event(2, 1, 2, "반려동물이 물 그릇에서 물을 마시는 모습을 보임."),
            event(3, 2, 2, "반려동물이 물을 마시며 갈증을 해소하는 모습을 보임."),
            event(4, 3, 2, "반려동물이 물그릇에 다가가 수분을 섭취하는 행동을 보임."),
            event(5, 4, 2, "반려동물이 마당을 돌아다니며 탐색하는 모습이 관찰됨."),
        ]
        result = run_sequence_query("행동 순서 알려줘", events=events, top_k=10)

        self.assertEqual([item["source_event_id"] for item in result["results"]], [1, 2, 5])
        self.assertEqual([item["merged_count"] for item in result["results"]], [1, 3, 1])
        self.assertEqual([item["id"] for item in result["behavior_events"]], [1, 2, 3, 4, 5])
        self.assertIn("비슷한 장면이 모두 3번 이어져요.", result["answer"])
        self.assertEqual(result["total_event_count"], 5)
        self.assertEqual(result["total_step_count"], 3)

    def test_same_action_far_apart_or_unsure_is_not_merged(self):
        far = [
            event(1, 0, 2, "반려동물이 사료를 먹고 있는 모습이 보여요."),
            event(2, 6, 2, "반려동물이 사료를 먹고 있는 모습이 보여요."),
        ]
        waiting = [
            event(1, 0, 2, "반려동물이 자동 급식기 주변에서 음식을 기다리는 행동을 보여요."),
            event(2, 1, 2, "반려동물이 사료 그릇에서 먹이를 섭취하고 있는 모습이 보여요."),
        ]
        trying = [
            event(1, 0, 2, "반려동물이 물을 마시고 있는 모습이 보여요."),
            event(2, 1, 2, "반려동물이 물을 마시려고 다가가는 모습이 보여요."),
        ]
        for events in (far, waiting, trying):
            result = run_sequence_query("행동 순서 알려줘", events=events, top_k=10)
            self.assertEqual([item["merged_count"] for item in result["results"]], [1, 1])

    def test_reviewed_video_omits_unverified_events_and_uses_reviewed_anchor(self):
        approved = event(59, 5, 2, "잘못된 예전 설명")
        approved["video_id"] = "funny_moments"
        approved["source_frames"] = [
            {"s3_key": "frames/funny_moments/petcam_20261014_161557_005_frame_2.00.jpg", "timestamp": 2.0},
            {"s3_key": "frames/funny_moments/petcam_20261014_161557_005_frame_28.00.jpg", "timestamp": 28.0},
        ]
        rejected = event(60, 6, 2, "물이 튀는 장치와 상호작용")
        rejected["video_id"] = "funny_moments"
        result = run_sequence_query("어떤 순서로 행동했어?", video_id="funny_moments", events=[rejected, approved])
        self.assertEqual([item["id"] for item in result["behavior_events"]], [59])
        self.assertEqual(result["results"][0]["timestamp"], 2.0)
        self.assertIn("문 아래 출입구", result["answer"])
        self.assertEqual(result["excluded_event_count"], 1)

    def test_relative_questions_return_only_neighbor_or_honest_no_evidence(self):
        events = [
            event(1, 0, 2, "침대 위에 있음"),
            event(2, 1, 2, "문 아래 출입구로 몸을 내밈"),
            event(3, 2, 2, "마당에서 걸음"),
        ]
        after = run_sequence_query("문 아래로 몸을 내민 다음 뭐했어?", video_id="dog-video", events=events)
        before = run_sequence_query("문 아래로 몸을 내밀기 전에 뭐했어?", video_id="dog-video", events=events)
        last = run_sequence_query("마지막에는 뭐했어?", video_id="dog-video", events=events)
        no_next = run_sequence_query("마당에서 걸은 다음 뭐했어?", video_id="dog-video", events=events)
        self.assertEqual([item["source_event_id"] for item in after["results"]], [3])
        self.assertEqual([item["source_event_id"] for item in before["results"]], [1])
        self.assertEqual([item["source_event_id"] for item in last["results"]], [3])
        self.assertEqual(no_next["results"], [])
        self.assertIn("이어지는 장면이 없어요", no_next["answer"])

    def test_does_not_treat_pushing_through_door_as_confirmed_exit(self):
        events = [event(1, 0, 2, "문 아래 출입구로 몸을 내밈"), event(2, 1, 2, "마당에서 걸음")]
        result = run_sequence_query("문을 나간 다음 뭐했어?", video_id="dog-video", events=events)
        self.assertEqual(result["results"], [])
        self.assertIn("질문하신 행동까지는 보이지 않아요", result["answer"])

    def test_relative_question_requires_unambiguous_anchor_and_video(self):
        events = [event(1, 0, 2, "문 앞에 있음"), event(2, 1, 2, "문을 통과함")]
        ambiguous = run_sequence_query("문 다음 뭐했어?", video_id="dog-video", events=events)
        no_video = run_sequence_query("문 다음 뭐했어?", events=events)
        self.assertEqual(ambiguous["results"], [])
        self.assertIn("하나로 좁히지", ambiguous["answer"])
        self.assertIn("영상", no_video["answer"])

    def test_each_step_carries_ordered_frames_inside_playback_window(self):
        multi = event(10, 2, 10, "소파에서 내려옴")
        multi["source_frames"] = [
            {"s3_key": f"frames/dog-video/petcam_20261104_090000_002_frame_{sec}.00.jpg", "timestamp": float(sec)}
            for sec in (14, 8, 10, 11, 12, 13, 40)
        ]
        result = run_sequence_query("행동 순서", events=[multi])
        step = result["results"][0]

        self.assertEqual(step["summary"], "소파에서 내려옴")
        timestamps = [item["timestamp"] for item in step["step_frames"]]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertLessEqual(len(timestamps), 4)
        self.assertNotIn(40.0, timestamps)
        self.assertIn(step["timestamp"], timestamps)


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

    def test_sequence_question_shows_full_timeline_even_with_small_top_k(self):
        events = [event(i, i, 2, f"행동 {i}") for i in range(1, 7)]
        fake_vector_store = MagicMock()
        with patch("db.behavior_events.get_behavior_events_for_aggregation", return_value=events), \
                patch.dict(sys.modules, {"pipeline.storage.vector_store": fake_vector_store}):
            result = rag.run_rag_query("오늘 강아지가 어떤 순서로 행동했어?", video_id="dog-video", top_k=3)

        self.assertEqual(len(result["results"]), 6)
        self.assertNotIn("처음", result["answer"])

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
