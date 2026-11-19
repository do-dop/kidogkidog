import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.search import router
from pipeline.search import question_gate as gate


def llm_says(text):
    return patch.object(gate, "_ask_llm", return_value=text)


class CheckQuestionFitTests(unittest.TestCase):
    def setUp(self):
        gate._cache.clear()
        env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def test_pet_words_pass_without_calling_the_llm(self):
        for query in ("강아지가 밥 먹는 장면", "고양이가 소파에 올라간 장면", "개가 짖는 장면", "코코가 산책하는 장면"):
            with llm_says('{"fit": false}') as ask:
                decision = gate.check_question_fit(query)
            self.assertTrue(decision.fit, query)
            self.assertEqual(decision.source, "rules")
            ask.assert_not_called()

    def test_clearly_unrelated_question_is_blocked_by_the_llm(self):
        with llm_says('{"fit": false, "reason": "우주선은 펫캠과 무관"}'):
            decision = gate.check_question_fit("우주선이 날아가는 장면")
        self.assertFalse(decision.fit)
        self.assertEqual(decision.source, "llm")
        self.assertIn("우주선", decision.reason)

    def test_general_motion_words_do_not_bypass_the_llm(self):
        # "달리는"은 자동차에도 쓰이므로 도메인 단어로 보지 않는다.
        with llm_says('{"fit": false, "reason": "자동차"}') as ask:
            decision = gate.check_question_fit("자동차가 도로를 달리는 장면")
        self.assertFalse(decision.fit)
        ask.assert_called_once()

    def test_ambiguous_question_follows_the_llm_but_defaults_to_fit(self):
        with llm_says('{"fit": true, "reason": "사람이 나올 수 있음"}'):
            self.assertTrue(gate.check_question_fit("사람이 요리하는 장면").fit)

    def test_llm_failure_or_garbage_lets_the_question_through(self):
        with patch.object(gate, "_ask_llm", side_effect=RuntimeError("timeout")):
            failed = gate.check_question_fit("우주선이 날아가는 장면")
        self.assertTrue(failed.fit)
        self.assertEqual(failed.source, "error")
        with llm_says("모르겠어요"):
            garbage = gate.check_question_fit("결혼식 장면")
        self.assertTrue(garbage.fit)
        self.assertEqual(garbage.source, "error")

    def test_missing_api_key_lets_the_question_through(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), llm_says('{"fit": false}') as ask:
            decision = gate.check_question_fit("우주선이 날아가는 장면")
        self.assertTrue(decision.fit)
        self.assertEqual(decision.source, "no_key")
        ask.assert_not_called()

    def test_repeated_question_is_served_from_cache(self):
        with llm_says('{"fit": false, "reason": "무관"}') as ask:
            gate.check_question_fit("결혼식 장면")
            second = gate.check_question_fit("결혼식 장면")
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(second.source, "cache")
        self.assertFalse(second.fit)

    def test_mode_defaults_to_off_and_ignores_unknown_values(self):
        with patch.dict(os.environ, {"QUESTION_GATE_MODE": ""}):
            self.assertEqual(gate.get_gate_mode(), "off")
        with patch.dict(os.environ, {"QUESTION_GATE_MODE": "bogus"}):
            self.assertEqual(gate.get_gate_mode(), "off")
        with patch.dict(os.environ, {"QUESTION_GATE_MODE": "Shadow"}):
            self.assertEqual(gate.get_gate_mode(), "shadow")


RAG_RESULT = {"answer": "검색 답변", "results": [], "evidence_items": [], "behavior_events": [], "used_llm": False}


class SearchRouterGateTests(unittest.TestCase):
    def setUp(self):
        gate._cache.clear()
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def query(self, text, mode, llm='{"fit": false, "reason": "무관"}', **extra):
        with patch.dict(os.environ, {"QUESTION_GATE_MODE": mode}), llm_says(llm), \
                patch("pipeline.search.rag_chain.run_rag_query", return_value=dict(RAG_RESULT)) as rag:
            response = self.client.post("/query", json={"query": text, **extra})
        return response.json(), rag

    def test_off_mode_does_nothing(self):
        body, rag = self.query("우주선이 날아가는 장면", "off")
        rag.assert_called_once()
        self.assertNotIn("question_gate", body)

    def test_shadow_mode_records_the_decision_but_still_searches(self):
        body, rag = self.query("우주선이 날아가는 장면", "shadow")
        rag.assert_called_once()
        self.assertEqual(body["answer"], "검색 답변")
        self.assertFalse(body["question_gate"]["fit"])

    def test_on_mode_blocks_unrelated_questions_without_searching(self):
        body, rag = self.query("우주선이 날아가는 장면", "on")
        rag.assert_not_called()
        self.assertEqual(body["match_level"], "out_of_scope")
        self.assertEqual(body["results"], [])
        self.assertIn("반려동물 영상에서 찾을 수 있는 질문이 아니에요", body["answer"])

    def test_on_mode_lets_pet_questions_through(self):
        body, rag = self.query("강아지가 밥 먹는 장면", "on")
        rag.assert_called_once()
        self.assertTrue(body["question_gate"]["fit"])

    def test_recommended_questions_skip_the_gate(self):
        body, rag = self.query("우주선이 날아가는 장면", "on", source_event_id=5)
        rag.assert_called_once()
        self.assertNotIn("question_gate", body)

    def test_gate_failure_never_blocks_search(self):
        with patch.dict(os.environ, {"QUESTION_GATE_MODE": "on"}), \
                patch.object(gate, "check_question_fit", side_effect=RuntimeError("boom")), \
                patch("pipeline.search.rag_chain.run_rag_query", return_value=dict(RAG_RESULT)) as rag:
            response = self.client.post("/query", json={"query": "우주선이 날아가는 장면"})
        self.assertEqual(response.status_code, 200)
        rag.assert_called_once()

    def test_match_level_from_the_search_result_is_forwarded(self):
        with patch.dict(os.environ, {"QUESTION_GATE_MODE": "off"}), \
                patch("pipeline.search.rag_chain.run_rag_query", return_value={**RAG_RESULT, "match_level": "related"}):
            body = self.client.post("/query", json={"query": "물 마시는 장면"}).json()
        self.assertEqual(body["match_level"], "related")


if __name__ == "__main__":
    unittest.main()
