import unittest

from pipeline.search.search_policy import contradicts_action, diverse_video_candidates


class SearchPolicyTests(unittest.TestCase):
    def test_long_video_does_not_take_every_candidate_slot(self):
        rows = [{"id": i, "video_id": "long"} for i in range(80)]
        rows += [{"id": 100, "video_id": "second"}, {"id": 101, "video_id": "third"}]
        selected = diverse_video_candidates(rows, 6)
        self.assertEqual([row["video_id"] for row in selected[:3]], ["long", "second", "third"])
        self.assertEqual([row["id"] for row in selected if row["video_id"] == "long"], [0, 1, 2, 3])

    def test_unknown_videos_are_never_merged_and_limit_is_respected(self):
        self.assertEqual(len(diverse_video_candidates([{}, {}, {}], 2)), 2)
        self.assertEqual(diverse_video_candidates([{}], 0), [])

    def test_explicit_opposite_actions_are_rejected(self):
        for query, wrong in (
            ("강아지가 물 마시는 장면", "강아지가 사료를 먹고 있음"),
            ("강아지가 밥 먹는 장면", "강아지가 젖을 먹고 있음"),
            ("수영장에 들어간 장면", "수영장에서 나오는 강아지"),
            ("소파가 넘어진 장면", "반려동물이 소파 위로 올라가고 있어요"),
        ):
            with self.subTest(query=query):
                self.assertTrue(contradicts_action(query, {"summary": wrong}))

    def test_same_action_unknown_action_and_multi_action_are_not_rejected(self):
        for query, summary in (
            ("물 마시는 장면", "물을 마시고 있음"),
            ("사람이 강아지를 들어 올리는 장면", "사람이 강아지를 들어 올림"),
            ("수영장에 들어갔다 나오는 장면", "수영장에서 나오는 강아지"),
            ("소파가 넘어진 장면", "소파에 올라가자 소파가 넘어졌어요"),
        ):
            with self.subTest(query=query):
                self.assertFalse(contradicts_action(query, {"summary": summary}))


if __name__ == "__main__":
    unittest.main()
