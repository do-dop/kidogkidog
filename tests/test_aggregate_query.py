import unittest

from pipeline.search.aggregate_query import count_episodes, event_matches_action, parse_count_action


class AggregateQueryTests(unittest.TestCase):
    def test_count_query_parses_drinking_but_not_duration(self):
        self.assertEqual(parse_count_action("오늘 물 몇 번 마셨어?"), ("drink", "물 마시기"))
        self.assertIsNone(parse_count_action("오늘 물 얼마나 오래 마셨어?"))

    def test_proximity_is_not_drinking(self):
        self.assertTrue(event_matches_action({"action": "물을 마심"}, "drink"))
        self.assertFalse(event_matches_action({"action": "물그릇 근처에 있음"}, "drink"))

    def test_overlap_merges_but_separate_actions_remain_separate(self):
        events = [
            {"id": 1, "video_id": "v1", "start_time": 10, "end_time": 14},
            {"id": 2, "video_id": "v1", "start_time": 12, "end_time": 15},
            {"id": 3, "video_id": "v1", "start_time": 30, "end_time": 34},
            {"id": 4, "video_id": "v2", "start_time": 12, "end_time": 15},
        ]
        episodes = count_episodes(events)
        self.assertEqual(len(episodes), 3)
        self.assertEqual(episodes[0]["event_ids"], [1, 2])


if __name__ == "__main__":
    unittest.main()
