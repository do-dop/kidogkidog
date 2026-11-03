import unittest

from pipeline.search.aggregate_query import count_episodes, event_matches_action, parse_count_action


class AggregateQueryTests(unittest.TestCase):
    def test_count_query_parses_drinking_but_not_duration(self):
        self.assertEqual(parse_count_action("오늘 물 몇 번 마셨어?"), ("drink", "물 마시기"))
        self.assertIsNone(parse_count_action("오늘 물 얼마나 오래 마셨어?"))

    def test_proximity_is_not_drinking(self):
        self.assertTrue(event_matches_action({"action": "물을 마심"}, "drink"))
        self.assertFalse(event_matches_action({"action": "물그릇 근처에 있음"}, "drink"))

    def test_gate_climb_requires_climbing_not_nearness_or_attempt(self):
        self.assertEqual(
            parse_count_action("강아지가 울타리에 몇 번 올라탔어?"),
            ("gate_climb", "울타리 올라타기"),
        )
        self.assertTrue(event_matches_action({"action": "강아지가 울타리에 올라탐"}, "gate_climb"))
        self.assertFalse(event_matches_action({"action": "강아지가 울타리 근처에 있음"}, "gate_climb"))
        self.assertFalse(event_matches_action({"action": "강아지가 울타리에 올라가려고 시도"}, "gate_climb"))

    def test_gate_climb_accepts_visual_evidence_described_as_crossing(self):
        self.assertTrue(event_matches_action(
            {"action": "반려동물이 울타리를 넘어가는 것으로 보임", "target_object": "울타리"},
            "gate_climb",
        ))
        self.assertTrue(event_matches_action(
            {
                "action": "반려동물이 우리를 넘으려고 시도함",
                "evidence": ["반려동물이 우리 쪽으로 손을 뻗고 점프를 시도함"],
            },
            "gate_climb",
        ))

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
