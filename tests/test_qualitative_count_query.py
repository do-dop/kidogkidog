import unittest
from unittest.mock import patch

from pipeline.search.rag_chain import run_rag_query


class QualitativeCountQueryTests(unittest.TestCase):
    def test_reviewed_rapid_sofa_activity_has_span_but_no_count(self):
        events = [
            {"id": 1, "video_id": "smartbench_0065", "start_time": 2.0,
             "end_time": 3.0, "action": "반려동물이 소파 위에 있음", "target_object": "소파"},
            {"id": 2, "video_id": "smartbench_0065", "start_time": 16.0,
             "end_time": 17.0, "action": "반려동물이 소파에서 내려옴", "target_object": "소파"},
        ]
        with patch("pipeline.search.rag_chain.get_behavior_events_for_aggregation", return_value=events), \
             patch("pipeline.search.rag_chain._frames_from_behavior_events",
                   return_value=[{"video_id": "smartbench_0065", "s3_key": "frame.jpg"}]):
            result = run_rag_query("강아지가 쇼파에 몇 번 올라갔어?", video_id="smartbench_0065")
        self.assertIsNone(result["aggregate_result"])
        self.assertIn("여러 차례", result["answer"])
        self.assertIn("정확한 횟수는 확인하기 어려워요", result["answer"])
        self.assertEqual((result["results"][0]["event_start"], result["results"][0]["event_end"]),
                         (0.0, 20.0))
        self.assertEqual((result["results"][0]["playback_start"], result["results"][0]["playback_end"]),
                         (0.0, 20.0))

    def test_other_video_does_not_inherit_reviewed_span(self):
        with patch("pipeline.search.rag_chain.get_behavior_events_for_aggregation", return_value=[]):
            result = run_rag_query("소파에 몇 번 올라갔어?", video_id="another_video")
        self.assertIsNone(result["aggregate_result"])
        self.assertEqual(result["results"], [])
        self.assertNotIn("여러 차례", result["answer"])


if __name__ == "__main__":
    unittest.main()
