import unittest

from scripts.local_search_proxy import apply_search_policy


class LocalSearchPolicyTests(unittest.TestCase):
    def test_remove_conflicting_scene_and_its_confident_answer(self):
        result = {"query_type": "search", "answer": "물을 마셨어요", "results": [
            {"video_id": "food", "summary": "사료를 먹고 있음", "score": 0.9}]}
        updated = apply_search_policy(result, "물 마시는 장면", 3)
        self.assertEqual(updated["results"], [])
        self.assertEqual(updated["evidence_items"], [])
        self.assertNotIn("물을 마셨어요", updated["answer"])

    def test_group_after_retrieval_preserves_other_videos_and_scene_links(self):
        result = {"query_type": "search", "results": [
            {"video_id": video, "timestamp": i, "summary": "물을 마심", "score": 0.9}
            for i, video in enumerate(["long"] * 6 + ["second", "third"])]}
        updated = apply_search_policy(result, "물 마시는 장면", 3)
        self.assertEqual([row["video_id"] for row in updated["results"]], ["long", "second", "third"])
        self.assertEqual(updated["results"][0]["matched_scene_count"], 6)

    def test_partial_match_is_not_described_as_confirmed(self):
        result = {"query_type": "search", "results": [
            {"video_id": "video", "timestamp": 2, "summary": "젖을 먹음", "match_level": "related_event"}]}
        updated = apply_search_policy(result, "어미 젖을 먹는 장면", 3)
        self.assertIn("세부 조건까지는 확인하지 못했어요", updated["answer"])

    def test_sequence_and_aggregate_evidence_are_not_reordered(self):
        for result in ({"query_type": "sequence", "results": [{"id": 1}, {"id": 2}]},
                       {"aggregate_result": {"count": 3}, "results": [{"id": 1}]}):
            self.assertIs(apply_search_policy(result, "순서", 1), result)


if __name__ == "__main__":
    unittest.main()
