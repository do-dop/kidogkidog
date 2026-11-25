import unittest

from pipeline.search.query_normalization import normalize_search_query
from pipeline.search.sequence_query import is_sequence_query, run_sequence_query


class QueryNormalizationTests(unittest.TestCase):
    def test_colloquial_furniture_change_uses_scene_change_route(self):
        for query in ("쇼파넘어간 장면", "소파 넘어간 장면", "소파가 넘어가는 장면 보여줘"):
            with self.subTest(query=query):
                normalized = normalize_search_query(query)
                self.assertIn("소파가 넘어진 장면", normalized)
                self.assertTrue(is_sequence_query(normalized))

    def test_does_not_turn_pet_movement_into_furniture_tipping(self):
        for query in ("강아지가 소파 넘어간 장면", "소파를 넘어간 장면", "소파 위로 올라간 장면",
                      "소파 너머로 간 장면", "소파가 넘어가지 않은 장면"):
            with self.subTest(query=query):
                self.assertEqual(normalize_search_query(query), query)

    def test_same_scene_is_found_in_global_and_scoped_search(self):
        def event(event_id, video_id, subject, summary):
            return {"id": event_id, "video_id": video_id, "start_time": 19, "end_time": 21,
                    "subject": subject, "action": summary, "summary": summary,
                    "target_object": "소파", "confidence": 0.9,
                    "source_frames": [{"frame_id": f"f{event_id}", "timestamp": 20,
                                       "s3_key": f"frames/{video_id}/{video_id}_000_frame_20.jpg"}]}

        tipping = event(141, "sofa-video", "소파", "소파가 뒤집어졌어요")
        climbing = event(340, "kitten-video", "반려동물", "반려동물이 소파 위로 올라가고 있어요")
        query = normalize_search_query("쇼파넘어간 장면")
        global_result = run_sequence_query(query, events=[climbing, tipping])
        scoped_result = run_sequence_query(query, video_id="sofa-video", events=[tipping])
        for result in (global_result, scoped_result):
            self.assertEqual([row["source_event_id"] for row in result["results"]], [141])
            self.assertIn("뒤집어", result["answer"])


if __name__ == "__main__":
    unittest.main()
