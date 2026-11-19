import unittest
from unittest.mock import patch

from pipeline.search import grounded_action_search, rag_chain


def event(event_id, start, end, action, target):
    return {
        "id": event_id,
        "video_id": "video",
        "start_time": start,
        "end_time": end,
        "subject": "반려동물",
        "action": action,
        "target_object": target,
        "summary": action,
        "confidence": 0.7,
        "source_frames": [{
            "frame_id": f"frame-{event_id}",
            "s3_key": f"frames/video/frame_{event_id}.jpg",
            "timestamp": start,
        }],
    }


class GroundedActionSearchTests(unittest.TestCase):
    def test_freeform_camera_query_merges_overlapping_episodes(self):
        events = [
            event(1, 28, 33, "반려동물이 카메라 쪽으로 다가옴", "카메라"),
            event(2, 31, 33, "반려동물이 카메라 쪽으로 다가옴", "카메라"),
            event(3, 41, 43, "반려동물이 카메라 쪽으로 다가옴", "카메라"),
            event(4, 45, 46, "반려동물이 카메라를 바라봄", "카메라"),
        ]
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=events):
            result = grounded_action_search.find_grounded_action_scenes(
                "고양이가 카메라에 얼굴 가까이한 장면 있어?", "video", 5,
            )
        self.assertEqual([item["source_event_id"] for item in result["results"]], [1, 3])
        self.assertTrue(all(item["s3_key"] for item in result["results"]))
        self.assertFalse(result["used_llm"])

    def test_paraphrased_door_and_leash_questions_find_recorded_actions(self):
        for query, source in (
            ("강아지가 문 쪽으로 가는 장면 있어?", event(5, 9, 10, "반려동물이 문 쪽으로 이동함", "문")),
            ("목줄이 떨어지는 장면 있어?", event(6, 2, 3, "목줄이 벽에서 떨어짐", "목줄")),
            ("강아지가 장난감 바라봐?", event(7, 0, 3, "반려동물이 장난감을 바라봄", "장난감")),
        ):
            with self.subTest(query=query), patch.object(
                grounded_action_search, "get_behavior_events_for_aggregation", return_value=[source]
            ):
                result = grounded_action_search.find_grounded_action_scenes(query, "video", 3)
            self.assertEqual(result["results"][0]["source_event_id"], source["id"])

    def test_title_or_object_alone_does_not_invent_action(self):
        candidate = event(1, 2, 3, "반려동물이 카메라를 바라봄", "카메라")
        with patch.object(grounded_action_search, "get_behavior_events_for_aggregation", return_value=[candidate]):
            self.assertIsNone(grounded_action_search.find_grounded_action_scenes(
                "고양이가 카메라에 가까이 왔어?", "video", 3,
            ))
        self.assertIsNone(grounded_action_search.find_grounded_action_scenes(
            "목줄을 맨 강아지 있어?", "video", 3,
        ))

    def test_aggregate_camera_query_does_not_count_close_intervals_as_separate_approaches(self):
        events = [
            event(1, 28, 33, "반려동물이 카메라 쪽으로 다가옴", "카메라"),
            event(2, 31, 33, "반려동물이 카메라 쪽으로 다가옴", "카메라"),
            event(3, 41, 43, "반려동물이 카메라 쪽으로 다가옴", "카메라"),
        ]
        with patch.object(rag_chain, "get_behavior_events_for_aggregation", return_value=events):
            result = rag_chain.run_rag_query("고양이가 카메라에 얼굴 몇 번 가까이해?", video_id="video")
        self.assertIsNone(result["aggregate_result"])
        self.assertEqual(len(result["results"]), 2)
        self.assertIn("정확한 접근 횟수는 단정하기 어려워요", result["answer"])
        self.assertFalse(result["used_llm"])


if __name__ == "__main__":
    unittest.main()
