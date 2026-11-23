import unittest
from unittest.mock import patch

from scripts.evaluate_action_search import evaluate_case


class EvaluateActionSearchTests(unittest.TestCase):
    def test_separates_extraction_and_search_failures_using_human_labels(self):
        base = {"video_id": "video", "query": "강아지가 몸을 터는 장면", "action_visible": True}
        with patch("scripts.evaluate_action_search.get_behavior_events_for_aggregation",
                   return_value=[{"id": 10}]), \
             patch("scripts.evaluate_action_search.run_rag_query", return_value={"results": []}):
            self.assertEqual(evaluate_case({**base, "expected_event_ids": []})["outcome"],
                             "extraction_failure")
            self.assertEqual(evaluate_case({**base, "expected_event_ids": [10]})["outcome"],
                             "search_failure")
        with patch("scripts.evaluate_action_search.get_behavior_events_for_aggregation",
                   return_value=[{"id": 10}]), \
             patch("scripts.evaluate_action_search.run_rag_query",
                   return_value={"results": [{"source_event_id": 10}]}):
            self.assertEqual(evaluate_case({**base, "expected_event_ids": [10]})["outcome"],
                             "success")


if __name__ == "__main__":
    unittest.main()
