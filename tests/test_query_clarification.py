import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.search import router
from pipeline.search.aggregate_query import parse_count_action
from pipeline.search.query_clarification import (
    build_count_clarification,
    build_scene_clarification,
    needs_scene_clarification,
)
from pipeline.search.rag_chain import run_rag_query


class QueryClarificationTests(unittest.TestCase):
    def test_short_front_paw_question_uses_video_event_context(self):
        events = [{"action": "반려동물이 유리문 앞에서 두 앞발을 들어 올림",
                   "target_object": "유리문"}]
        with patch("pipeline.search.rag_chain.get_behavior_events_for_aggregation", return_value=events):
            result = run_rag_query("앞 발 몇번 듦?", video_id="example_video")
        clarification = result["clarification"]
        self.assertEqual(clarification["question"], "유리문 앞에서 앞발을 든 횟수를 물으셨나요?")
        self.assertEqual(clarification["options"][0]["query"], "유리문 앞에서 앞발을 몇 번 들었어?")
        self.assertEqual(parse_count_action(clarification["options"][0]["query"]),
                         ("front_paw_raise", "앞발 들기"))
        self.assertEqual(result["results"], [])

    def test_no_video_does_not_invent_a_place(self):
        clarification = build_count_clarification("앞발 몇 번 듦?", [])
        self.assertEqual(clarification["options"][0]["query"], "앞발을 몇 번 들었어?")
        self.assertNotIn("유리문", clarification["question"])

    def test_unknown_action_asks_for_clarification_without_false_answer(self):
        clarification = build_count_clarification("몇 번 했어?", [{"action": "물을 마심"}])
        self.assertEqual(clarification["options"], [])
        self.assertIn("어떤 행동", clarification["question"])

    def test_multiple_recorded_actions_offer_distinct_requeries(self):
        events = [{"action": "물을 마심"}, {"action": "사료를 먹음"}]
        clarification = build_count_clarification("물 마시기와 사료 먹기 횟수?", events)
        self.assertEqual(len(clarification["options"]), 2)
        self.assertEqual({parse_count_action(item["query"])[0] for item in clarification["options"]},
                         {"drink", "eat"})

    def test_vague_scene_question_offers_concrete_recorded_action(self):
        events = [{"action": "반려동물이 유리문 앞에서 두 앞발을 들어 올림",
                   "target_object": "유리문"}]
        with patch("pipeline.search.rag_chain.get_behavior_events_for_aggregation", return_value=events):
            result = run_rag_query("유리문 앞에서 그거 보여줘", video_id="example_video")
        self.assertIn("유리문 앞에서 앞발을 들어 올리는 장면", result["clarification"]["question"])
        self.assertEqual(result["clarification"]["options"][0]["query"],
                         "유리문 앞에서 앞발을 들어 올리는 장면 보여줘")
        self.assertEqual(result["results"], [])

    def test_concrete_question_or_daily_overview_is_not_rewritten(self):
        for question in ("강아지가 소파에 올라간 장면 보여줘", "그 장면에서 두 마리야?",
                         "오늘 강아지가 뭐 했어?"):
            with self.subTest(question=question):
                self.assertFalse(needs_scene_clarification(question))
        self.assertTrue(needs_scene_clarification("그거 보여줘"))

    def test_generic_vague_question_without_video_asks_for_details(self):
        clarification = build_scene_clarification("그거 보여줘", [{"action": "물을 마심"}])
        self.assertEqual(clarification["options"], [])
        self.assertIn("무엇을 보여드릴까요", clarification["question"])

    def test_api_exposes_clarification_to_search_screen(self):
        app = FastAPI()
        app.include_router(router)
        with patch("pipeline.search.rag_chain.run_rag_query", return_value={
            "answer": "앞발을 든 횟수를 물으셨나요?", "results": [], "behavior_events": [],
            "used_llm": False, "clarification": {"question": "앞발을 든 횟수를 물으셨나요?",
                                                 "options": [{"label": "앞발을 몇 번 들었어?",
                                                              "query": "앞발을 몇 번 들었어?"}]},
        }):
            response = TestClient(app).post("/query", json={"query": "앞발 몇 번 듦?", "video_id": "video"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["clarification"]["options"][0]["query"], "앞발을 몇 번 들었어?")


if __name__ == "__main__":
    unittest.main()
