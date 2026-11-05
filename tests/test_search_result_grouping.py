import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.search import router
from pipeline.search.result_grouping import group_results_by_video


class SearchResultGroupingTests(unittest.TestCase):
    def test_one_card_per_video_keeps_all_scene_times_and_best_hit(self):
        rows = [
            {"video_id": "dog_a", "frame_id": "a1", "timestamp": 4, "score": 0.75,
             "source_chunk": "first", "merged_frame_count": 2},
            {"video_id": "dog_b", "frame_id": "b1", "timestamp": 8, "score": 0.82},
            {"video_id": "dog_a", "frame_id": "a2", "timestamp": 40, "score": 0.91,
             "source_chunk": "second", "merged_frame_count": 3},
        ]
        grouped = group_results_by_video(rows, limit=3)
        self.assertEqual([item["video_id"] for item in grouped], ["dog_a", "dog_b"])
        self.assertEqual(grouped[0]["frame_id"], "a2")
        self.assertEqual(grouped[0]["matched_scene_count"], 2)
        self.assertEqual(grouped[0]["matched_frame_count"], 5)
        self.assertEqual([scene["timestamp"] for scene in grouped[0]["related_scenes"]], [40, 4])
        self.assertEqual(rows[0].keys(), {"video_id", "frame_id", "timestamp", "score", "source_chunk", "merged_frame_count"})

    def test_missing_video_id_is_not_merged_with_another_unknown(self):
        grouped = group_results_by_video([
            {"frame_id": "one", "score": 0.8},
            {"frame_id": "two", "score": 0.7},
        ], limit=2)
        self.assertEqual(len(grouped), 2)

    def test_query_api_returns_one_video_card_with_scene_metadata(self):
        app = FastAPI()
        app.include_router(router)
        hits = [
            {"video_id": "video_a", "frame_id": "a1", "timestamp": 2, "score": 0.9},
            {"video_id": "video_a", "frame_id": "a2", "timestamp": 30, "score": 0.8},
            {"video_id": "video_b", "frame_id": "b1", "timestamp": 5, "score": 0.7},
        ]
        with patch("pipeline.search.rag_chain.run_rag_query", return_value={
            "answer": "ok", "results": hits, "evidence_items": hits,
            "behavior_events": [], "used_llm": False,
        }):
            response = TestClient(app).post("/query", json={"query": "강아지 보여?", "top_k": 3})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual([row["video_id"] for row in payload["results"]], ["video_a", "video_b"])
        self.assertEqual(payload["results"][0]["matched_scene_count"], 2)
        self.assertEqual([row["frame_id"] for row in payload["results"][0]["related_scenes"]], ["a1", "a2"])

    def test_query_api_keeps_every_step_of_a_sequence_answer(self):
        app = FastAPI()
        app.include_router(router)
        steps = [
            {"video_id": "video_a", "frame_id": "a1", "timestamp": 2, "score": 0.8},
            {"video_id": "video_a", "frame_id": "a2", "timestamp": 2, "score": 0.8},
            {"video_id": "video_a", "frame_id": "a3", "timestamp": 2, "score": 0.8},
        ]
        with patch("pipeline.search.rag_chain.run_rag_query", return_value={
            "answer": "ok", "results": steps, "evidence_items": steps, "query_type": "sequence",
            "behavior_events": [], "used_llm": False,
        }):
            response = TestClient(app).post("/query", json={"query": "행동 순서 알려줘", "top_k": 3})
        payload = response.json()
        self.assertEqual([row["frame_id"] for row in payload["results"]], ["a1", "a2", "a3"])
        self.assertEqual([row["frame_id"] for row in payload["evidence_items"]], ["a1", "a2", "a3"])
        self.assertNotIn("related_scenes", payload["results"][0])


if __name__ == "__main__":
    unittest.main()
