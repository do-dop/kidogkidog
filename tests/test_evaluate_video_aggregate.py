import unittest

from scripts.evaluate_video_aggregate import match_events


class AggregateEventMatchingTests(unittest.TestCase):
    def test_matching_count_does_not_hide_wrong_events(self):
        predictions = [{"after_time": 10.5}, {"after_time": 16.0}]
        references = [{"start": 10.0, "end": 18.0}, {"start": 24.0, "end": 27.0}]
        result = match_events(predictions, references)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["unmatched_prediction_indices"], [1])
        self.assertEqual(result["missed_reference_indices"], [1])


if __name__ == "__main__":
    unittest.main()
