import unittest

from pipeline.search.event_text_search import best_specific_event, event_as_search_result


class EventTextSearchTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"id": 1, "video_id": "pool", "start_time": 0, "end_time": 4.8,
             "subject": "강아지", "action": "수영장 안으로 뛰어들어 수영한다",
             "summary": "강아지가 수영장 가장자리에서 물속으로 뛰어들어 헤엄친다"},
            {"id": 2, "video_id": "pool", "start_time": 4.8, "end_time": 9,
             "subject": "강아지", "action": "수영장 얕은 계단을 타고 밖으로 올라온다",
             "summary": "강아지가 물 밖으로 나와 데크 위로 올라온다"},
            {"id": 3, "video_id": "sofa", "start_time": 18, "end_time": 22,
             "subject": "강아지", "action": "소파 위로 뛰어올라 여성 곁에 간다",
             "summary": "강아지가 소파 위로 올라간다"},
        ]

    def test_specific_action_selects_its_event_and_interval(self):
        match = best_specific_event("강아지가 수영장 얕은 계단을 타고 밖으로 올라온 장면 보여줘", self.events)
        self.assertIsNotNone(match)
        event, score = match
        self.assertEqual(event["id"], 2)
        result = event_as_search_result(event, score)
        self.assertEqual(result["video_id"], "pool")
        self.assertEqual((result["playback_start"], result["playback_end"]), (4.3, 9.5))

    def test_generic_query_does_not_override_visual_ranking(self):
        self.assertIsNone(best_specific_event("강아지 보여줘", self.events))

    def test_repeated_indistinguishable_actions_are_not_forced_to_one_time(self):
        repeated = [
            {"video_id": "cat", "start_time": 0, "end_time": 5,
             "action": "고양이가 밥그릇에서 먹이를 먹는다"},
            {"video_id": "cat", "start_time": 10, "end_time": 15,
             "action": "고양이가 밥그릇에서 먹이를 먹는다"},
        ]
        self.assertIsNone(best_specific_event("고양이가 밥그릇에서 먹이를 먹는 장면 보여줘", repeated))

    def test_bad_interval_is_not_returned(self):
        self.assertIsNone(event_as_search_result({"video_id": "pool", "start_time": 9, "end_time": 4}, 0.9))


if __name__ == "__main__":
    unittest.main()
