import unittest

from scripts.evaluate_video_aggregate import match_events
from pipeline.search.aggregate_query import event_matches_action, parse_count_action


class AggregateEventMatchingTests(unittest.TestCase):
    def test_count_action_does_not_count_eating_for_head_lifts(self):
        self.assertEqual(parse_count_action("고양이가 밥을 먹다가 고개를 몇 번 들었어?"),
                         ("head_lift", "고개 들기"))
        self.assertTrue(event_matches_action(
            {"action": "고양이가 그릇에서 고개를 듦", "summary": "고개를 든 채 주변을 바라봐요"},
            "head_lift"))
        self.assertEqual(parse_count_action("고양이가 밥을 몇 번 먹었어?"), ("eat", "먹기"))

    def test_matching_count_does_not_hide_wrong_events(self):
        predictions = [{"after_time": 10.5}, {"after_time": 16.0}]
        references = [{"start": 10.0, "end": 18.0}, {"start": 24.0, "end": 27.0}]
        result = match_events(predictions, references)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["unmatched_prediction_indices"], [1])
        self.assertEqual(result["missed_reference_indices"], [1])


if __name__ == "__main__":
    unittest.main()
