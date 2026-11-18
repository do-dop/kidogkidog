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
    def run_query(self, query, found, top_events=(), all_events=(), env=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(vs, "search_with_query_expansion", return_value=found))
            stack.enter_context(patch.object(rag, "get_behavior_events", return_value=list(all_events)))
            stack.enter_context(patch.object(rag, "get_behavior_events_overlapping", return_value=[]))
            stack.enter_context(patch.object(rag, "get_top_behavior_events", return_value=list(top_events)))
            stack.enter_context(patch.object(rag, "build_prompt_hint", return_value=""))
            answer = stack.enter_context(
                patch.object(rag, "_generate_answer_with_langchain", return_value="생성된 답변")
            )
            stack.enter_context(patch.dict("os.environ", env or {}, clear=False))
            result = rag.run_rag_query(query=query, video_id="video", top_k=3)
        return result, answer

    def test_cat_question_drops_frames_where_only_a_dog_is_detected(self):
        dog = {**frame("dog", 0.30), "object_labels": "dog,couch"}
        cat = {**frame("cat", 0.28), "object_labels": "cat"}
        unknown = {**frame("unknown", 0.27), "object_labels": ""}
        result, _ = self.run_query("고양이가 러그 위에서 주변을 둘러봄", [dog, cat, unknown])

        self.assertEqual([item["frame_id"] for item in result["results"]], ["cat", "unknown"])

    def test_cat_question_with_only_dog_frames_is_not_answered(self):
        dog = {**frame("dog", 0.30), "object_labels": "dog"}
        result, answer = self.run_query("고양이가 급식기로 걸어감", [dog])

        self.assertEqual(result["results"], [])
        self.assertIn("찾지 못했", result["answer"])
        answer.assert_not_called()

    def test_question_naming_both_or_neither_animal_is_not_filtered(self):
        dog = {**frame("dog", 0.30), "object_labels": "dog"}
        for query in ("고양이와 강아지가 같이 있는 장면", "물 마시는 장면", "몇 개 먹었어 장면"):
            result, _ = self.run_query(query, [dog])
            self.assertEqual(len(result["results"]), 1, query)

    def test_species_filter_can_be_disabled(self):
        dog = {**frame("dog", 0.30), "object_labels": "dog"}
        result, _ = self.run_query("고양이가 걷는 장면", [dog], env={"SEARCH_SPECIES_FILTER": "0"})
        self.assertEqual(len(result["results"]), 1)

    def test_below_threshold_candidates_are_offered_as_related_not_as_a_match(self):
        found = [frame("near", 0.215, 3.0), frame("nearer", 0.225, 6.0), frame("far", 0.15)]
        result, answer = self.run_query("물 마시는 장면", found)

        self.assertEqual(result["match_level"], "related")
        self.assertEqual([item["frame_id"] for item in result["results"]], ["nearer", "near"])
        self.assertTrue(all(item["match_level"] == "related" for item in result["results"]))
        self.assertIn("정확히 맞는 장면은 찾지 못했어요", result["answer"])
        self.assertIn("정확히 일치하지 않을 수 있어요", result["answer"])
        self.assertFalse(result["used_llm"])
        answer.assert_not_called()

    def test_related_candidates_can_be_turned_off(self):
        found = [frame("near", 0.215)]
        result, _ = self.run_query("물 마시는 장면", found, env={"SEARCH_RELATED_MIN_SCORE": "0"})

        self.assertEqual(result["results"], [])
        self.assertNotIn("match_level", result)

    def test_a_real_match_is_not_mixed_with_related_candidates(self):
        found = [frame("good", 0.27), frame("near", 0.215)]
        result, _ = self.run_query("물 마시는 장면", found)

        self.assertEqual([item["frame_id"] for item in result["results"]], ["good"])
        self.assertNotEqual(result.get("match_level"), "related")

    def test_search_failure_offers_only_events_that_match_the_question_words(self):
        drinking = event(7, "반려동물이 물을 마시는 모습")
        unrelated = event(8, "반려동물이 침대에서 쉬는 모습")
        with patch.object(vs, "search_with_query_expansion", side_effect=RuntimeError("offline")), \
                patch.object(rag, "get_behavior_events", return_value=[]), \
                patch.object(rag, "get_behavior_events_overlapping", return_value=[]), \
                patch.object(rag, "get_top_behavior_events", return_value=[unrelated, drinking]), \
                patch.object(rag, "build_prompt_hint", return_value=""), \
                patch.object(rag, "_generate_answer_with_langchain") as llm:
            result = rag.run_rag_query("강아지가 물 마시는 장면 보여줘", video_id="video", top_k=3)

        self.assertEqual(result["match_level"], "related_events")
        self.assertEqual([item["id"] for item in result["behavior_events"]], [7])
        self.assertIn("검색 서버에 연결하지 못해", result["answer"])
        self.assertFalse(result["used_llm"])
        llm.assert_not_called()

    def test_search_failure_does_not_present_arbitrary_recording(self):
        with patch.object(vs, "search_with_query_expansion", side_effect=RuntimeError("offline")), \
                patch.object(rag, "get_behavior_events", return_value=[]), \
                patch.object(rag, "_get_relevant_behavior_events", return_value=[]), \
                patch.object(rag, "_fallback_indexed_frames") as fallback:
            result = rag.run_rag_query("강아지가 소파 위로 올라가는 장면 보여줘", top_k=3)
        self.assertEqual(result["results"], [])
        self.assertFalse(result["used_llm"])
        self.assertIn("연결하지 못해", result["answer"])
        fallback.assert_not_called()

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
        # 잘못된 값이면 기본 기준(0.23)을 쓴다. 0.20은 기준 미달이라 "비슷한 장면"으로만 나온다.
        found = [frame("a", 0.20)]
        result, _ = self.run_query("물 마시는 장면", found, env={"SEARCH_MIN_SCORE": "not-a-number"})

        self.assertEqual(result["match_level"], "related")

    def test_clear_event_description_promotes_its_precise_time_window(self):
        broad_frame = frame("visual-hit", 0.31, timestamp=30)
        pool_exit = {
            **event(9, "강아지가 수영장 얕은 계단을 타고 물 밖으로 올라온다"),
            "start_time": 4.8,
            "end_time": 9.0,
        }
        result, _ = self.run_query(
            "강아지가 수영장 얕은 계단을 타고 밖으로 올라온 장면 보여줘",
            [broad_frame],
            all_events=[pool_exit],
        )

        self.assertEqual(result["results"][0]["source_event_id"], 9)
        self.assertEqual(result["results"][0]["playback_start"], 4.3)
        self.assertEqual(result["results"][0]["playback_end"], 9.5)


class FrameCountPenaltyTests(unittest.TestCase):
    def frames(self):
        return [
            {"frame_id": "big", "video_id": "big_video", "score": 0.290},
            {"frame_id": "small", "video_id": "small_video", "score": 0.275},
        ]

    def test_large_video_loses_its_head_start_in_global_search(self):
        with patch.object(rag, "_video_frame_counts", return_value={"big_video": 300, "small_video": 8}):
            result = rag._apply_frame_count_penalty(self.frames(), video_id=None)

        self.assertEqual([item["video_id"] for item in result], ["small_video", "big_video"])
        self.assertAlmostEqual(result[1]["clip_score"], 0.290)
        self.assertLess(result[1]["score"], 0.290)

    def test_video_scoped_search_is_not_adjusted(self):
        with patch.object(rag, "_video_frame_counts", return_value={"big_video": 300, "small_video": 8}) as counts:
            result = rag._apply_frame_count_penalty(self.frames(), video_id="big_video")

        self.assertEqual([item["video_id"] for item in result], ["big_video", "small_video"])
        counts.assert_not_called()

    def test_penalty_can_be_disabled(self):
        with patch.dict("os.environ", {"RANK_FRAME_COUNT_PENALTY": "0"}), \
                patch.object(rag, "_video_frame_counts", return_value={"big_video": 300, "small_video": 8}):
            result = rag._apply_frame_count_penalty(self.frames(), video_id=None)

        self.assertEqual([item["video_id"] for item in result], ["big_video", "small_video"])

    def test_count_lookup_failure_keeps_original_order(self):
        with patch.object(rag, "_video_frame_counts", side_effect=RuntimeError("chroma down")):
            result = rag._apply_frame_count_penalty(self.frames(), video_id=None)

        self.assertEqual([item["video_id"] for item in result], ["big_video", "small_video"])

    def test_penalty_is_applied_after_the_similarity_threshold(self):
        # 보정 후 점수가 기준 아래로 내려가도 원래 CLIP 점수가 기준 이상이면 결과에서 빠지면 안 된다.
        found = [{**frame("only", 0.24), "video_id": "big_video"}, {**frame("other", 0.23), "video_id": "small_video"}]
        with patch.object(vs, "search_with_query_expansion", return_value=found), \
                patch.object(rag, "_video_frame_counts", return_value={"big_video": 300, "small_video": 8}), \
                patch.object(rag, "get_behavior_events_overlapping", return_value=[]), \
                patch.object(rag, "get_top_behavior_events", return_value=[]), \
                patch.object(rag, "build_prompt_hint", return_value=""), \
                patch.object(rag, "_generate_answer_with_langchain", return_value="답변"):
            result = rag.run_rag_query(query="물 마시는 장면", top_k=3)

        self.assertEqual(len(result["results"]), 2)


if __name__ == "__main__":
    unittest.main()
