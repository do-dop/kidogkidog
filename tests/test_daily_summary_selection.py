import unittest
from unittest.mock import patch

from pipeline.search.query_suggester import _event_category, get_suggestion_behavior_events


class DailySummarySelectionTests(unittest.TestCase):
    def test_pet_and_obstacle_words_do_not_imply_water(self):
        self.assertEqual(_event_category({"action": "반려동물이 장애물을 넘어감"}), "other")
        self.assertEqual(_event_category({"action": "반려동물이 욕조에서 목욕함"}), "water")

    def test_obstacle_and_bath_can_both_be_summary_events(self):
        events = [
            {"id": 1, "action": "반려동물이 장애물을 넘어감", "confidence": 0.9, "interestingness": 0.9},
            {"id": 2, "action": "반려동물이 욕조에서 목욕함", "confidence": 0.9, "interestingness": 0.9},
        ]
        with patch("pipeline.search.query_suggester.get_top_behavior_events", return_value=events):
            shown = get_suggestion_behavior_events(video_id="video", limit=5)
        self.assertEqual({event["id"] for event in shown}, {1, 2})

    def test_directly_observed_bath_can_beat_hedged_water_event(self):
        events = [
            {"id": 1, "action": "반려동물이 물을 마시는 것으로 보임", "confidence": 0.9,
             "interestingness": 0.9},
            {"id": 2, "action": "반려동물이 욕조에서 목욕을 하고 있음", "confidence": 0.9,
             "interestingness": 0.85},
        ]
        with patch("pipeline.search.query_suggester.get_top_behavior_events", return_value=events):
            shown = get_suggestion_behavior_events(limit=5)
        self.assertEqual([event["id"] for event in shown], [2])


if __name__ == "__main__":
    unittest.main()
