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

    def test_other_ambiguous_actions_explain_evidence_without_a_total(self):
        cases = [
            ("smartbench_0142", "고양이가 밥을 먹다가 고개를 몇 번 들었어?", "고양이가 고개를 들었음", 9.0, 27.5),
            ("smartbench_0608", "강아지가 수영장 물에 몇 번 들어갔어?", "강아지가 물속에 들어감", 0.0, 25.0),
            ("smartbench_0936", "유리문 앞에서 앞발을 몇 번 들었어?", "반려동물이 유리문 앞에서 두 앞발을 들어 올림", 0.0, 20.0),
        ]
        for video_id, query, action, start, end in cases:
            with self.subTest(video_id=video_id):
                events = [{"id": 1, "video_id": video_id, "start_time": start + 1,
                           "end_time": start + 2, "action": action}]
                with patch("pipeline.search.rag_chain.get_behavior_events_for_aggregation", return_value=events), \
                     patch("pipeline.search.rag_chain._frames_from_behavior_events",
                           return_value=[{"video_id": video_id, "s3_key": "frame.jpg"}]):
                    result = run_rag_query(query, video_id=video_id)
                self.assertIsNone(result["aggregate_result"])
                self.assertIn("정확한", result["answer"])
                self.assertEqual((result["results"][0]["playback_start"],
                                  result["results"][0]["playback_end"]), (start, end))

    def test_clean_water_entry_still_returns_confirmed_intervals(self):
        events = [{"id": i, "video_id": "smartbench_0642", "start_time": t,
                   "end_time": t + 0.1, "action": "반려동물이 물속으로 들어감"}
                  for i, t in enumerate((3.0, 5.0), start=1)]
        with patch("pipeline.search.rag_chain.get_behavior_events_for_aggregation", return_value=events), \
             patch("pipeline.search.rag_chain._frames_from_behavior_events", return_value=[]):
            result = run_rag_query("강아지가 분수 물속에 몇 번 들어갔어?", video_id="smartbench_0642")
        self.assertEqual(result["aggregate_result"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
