import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.search import question_understanding, rag_chain


class QuestionUnderstandingTests(unittest.TestCase):
    def setUp(self):
        question_understanding.question_is_understood.cache_clear()

    def test_model_interprets_only_query_text(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = SimpleNamespace(
                output_text=json.dumps({"understood": False})
            )
            self.assertFalse(question_understanding.question_is_understood("nonsense request"))
            content = client.return_value.responses.create.call_args.kwargs["input"][0]["content"]
            self.assertIn("Question: nonsense request", content)
            client.return_value.responses.create.assert_called_once()

    def test_uninterpretable_question_stops_before_any_retrieval(self):
        with patch.object(question_understanding, "question_is_understood", return_value=False), \
             patch.object(rag_chain, "_run_rag_query") as retrieve:
            result = rag_chain.run_rag_query("극어보여줘", video_id="any_video")
        retrieve.assert_not_called()
        self.assertEqual(result["results"], [])
        self.assertEqual(result["clarification"]["options"], [])
        self.assertIn("이해하지 못했어요", result["answer"])

    def test_clear_query_keeps_the_existing_retrieval_path(self):
        payload = {"answer": "검색 결과", "results": [], "used_llm": False}
        with patch.object(question_understanding, "question_is_understood", return_value=True), \
             patch.object(rag_chain, "_run_rag_query", return_value=payload) as retrieve, \
             patch.object(rag_chain, "verify_uncertain_scene", side_effect=lambda *args: args[2]):
            result = rag_chain.run_rag_query("강아지 보여줘", video_id="any_video")
        retrieve.assert_called_once()
        self.assertIs(result, payload)


if __name__ == "__main__":
    unittest.main()
