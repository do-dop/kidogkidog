import unittest
from unittest.mock import patch

from pipeline.search.rag_chain import run_rag_query


class GateSceneGroundingTests(unittest.TestCase):
    def test_gate_scene_uses_behavior_record_without_trusting_wrong_object_label(self):
        event = {
            "id": 1,
            "video_id": "smartbench_0188",
            "start_time": 6.5,
            "end_time": 12.5,
            "action": "반려동물이 울타리를 타고 올라가는 것으로 보임",
            "target_object": "울타리",
            "source_frames": [{
                "timestamp": 12.5,
                "s3_key": "frames/smartbench_0188/frame_12.5.jpg",
                "object_labels": ["cat", "couch"],
            }],
        }
        with patch("pipeline.search.rag_chain.get_behavior_events_for_aggregation", return_value=[event]), \
             patch("pipeline.search.rag_chain._generate_answer_with_langchain",
                   side_effect=AssertionError("a matching event must not be contradicted by LLM")):
            result = run_rag_query(
                "울타리를 타고 올라가는 장면이 있어?",
                video_id="smartbench_0188",
                time_range={"start_hour": 0, "end_hour": 24},
            )

        self.assertIn("6.5~12.5초", result["answer"])
        self.assertNotIn("확인되지", result["answer"])
        self.assertEqual(result["behavior_events"], [event])
        self.assertEqual(result["results"][0]["object_labels"], "")
        self.assertFalse(result["used_llm"])

    def test_missing_gate_record_reports_uncertainty_without_irrelevant_frames(self):
        with patch("pipeline.search.rag_chain.get_behavior_events_for_aggregation", return_value=[]), \
             patch("pipeline.search.rag_chain._generate_answer_with_langchain",
                   side_effect=AssertionError("no grounded event")):
            result = run_rag_query("울타리를 타고 올라가는 장면이 있어?", video_id="video")

        self.assertIn("기록이 아직 없어요", result["answer"])
        self.assertEqual(result["results"], [])


if __name__ == "__main__":
    unittest.main()
