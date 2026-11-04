import unittest
from contextlib import ExitStack
from unittest.mock import patch

from pipeline.search import rag_chain as rag
from pipeline.storage import vector_store as vs


def frame(frame_id, score, timestamp=1.0):
    return {
        "frame_id": frame_id,
        "frame_path": f"{frame_id}.jpg",
        "s3_key": f"frames/video/{frame_id}.jpg",
        "timestamp": timestamp,
        "video_id": "video",
        "object_labels": "",
        "score": score,
    }


def event(event_id, summary):
    return {
        "id": event_id,
        "video_id": "video",
        "start_time": 1.0,
        "end_time": 5.0,
        "action": summary,
        "summary": summary,
        "confidence": 0.9,
        "interestingness": 0.9,
        "source_frames": [{"timestamp": 1.0, "s3_key": f"frames/video/event-{event_id}.jpg"}],
    }


class SearchRelevanceTests(unittest.TestCase):
    def run_query(self, query, found, top_events=(), env=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(vs, "search_with_query_expansion", return_value=found))
            stack.enter_context(patch.object(rag, "get_behavior_events_overlapping", return_value=[]))
            stack.enter_context(patch.object(rag, "get_top_behavior_events", return_value=list(top_events)))
            stack.enter_context(patch.object(rag, "build_prompt_hint", return_value=""))
            answer = stack.enter_context(
                patch.object(rag, "_generate_answer_with_langchain", return_value="생성된 답변")
            )
            stack.enter_context(patch.dict("os.environ", env or {}, clear=False))
            result = rag.run_rag_query(query=query, video_id="video", top_k=3)
        return result, answer

    def test_low_similarity_frames_are_not_shown_and_llm_is_skipped(self):
        found = [frame("a", 0.19), frame("b", 0.18)]
        result, answer = self.run_query(
            "우주선이 날아가는 장면", found, top_events=[event(1, "반려동물이 물 그릇 근처에서 움직임")],
        )

        self.assertEqual(result["results"], [])
        self.assertEqual(result["behavior_events"], [])
        self.assertIn("찾지 못했어요", result["answer"])
        answer.assert_not_called()

    def test_only_frames_above_threshold_are_kept(self):
        found = [frame("good", 0.27), frame("weak", 0.20)]
        result, _ = self.run_query("물 마시는 장면", found)

        self.assertEqual([item["frame_id"] for item in result["results"]], ["good"])

    def test_event_with_matching_words_still_answers_when_clip_is_weak(self):
        found = [frame("weak", 0.20)]
        drinking = event(7, "반려동물이 물을 마시는 모습")
        unrelated = event(8, "반려동물이 침대에서 쉬는 모습")
        result, _ = self.run_query("강아지가 물 마시는 장면", found, top_events=[drinking, unrelated])

        self.assertEqual([item["id"] for item in result["behavior_events"]], [7])
        self.assertEqual(len(result["results"]), 1)

    def test_common_word_alone_does_not_match_an_event(self):
        # "장면"은 이벤트 설명에 흔히 들어가지만 질문과 맞는다는 근거는 아니다.
        found = [frame("weak", 0.19)]
        result, _ = self.run_query(
            "우주선이 날아가는 장면", found, top_events=[event(1, "즐거운 시간을 보내는 장면")],
        )

        self.assertEqual(result["behavior_events"], [])
        self.assertEqual(result["results"], [])

    def test_water_keyword_does_not_match_the_word_pet_animal(self):
        # "반려동물"의 "물"이 물 마시기 질문과 일치하면 안 된다.
        resting = event(1, "반려동물이 침대에서 쉬는 모습")
        drinking = event(2, "반려동물이 물을 마시는 모습")

        self.assertLess(rag._behavior_query_score("물 마시는 장면", resting), 1.0)
        self.assertGreaterEqual(rag._behavior_query_score("물 마시는 장면", drinking), 1.0)

    def test_threshold_can_be_tuned_with_environment_variable(self):
        found = [frame("a", 0.20)]
        result, _ = self.run_query("물 마시는 장면", found, env={"SEARCH_MIN_SCORE": "0.15"})

        self.assertEqual([item["frame_id"] for item in result["results"]], ["a"])

    def test_invalid_threshold_falls_back_to_default(self):
        found = [frame("a", 0.20)]
        result, _ = self.run_query("물 마시는 장면", found, env={"SEARCH_MIN_SCORE": "not-a-number"})

        self.assertEqual(result["results"], [])


if __name__ == "__main__":
    unittest.main()
