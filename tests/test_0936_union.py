import unittest

from scripts.validate_0936_union import merge_candidates


class UnionCandidateTests(unittest.TestCase):
    def test_nearby_scan_proposals_merge_without_erasing_next_action(self):
        runs = {
            "0.5": {"candidates": [{"time": 4.5, "subject": "dog"},
                                    {"time": 6.25, "subject": "dog"}]},
            "0.25": {"candidates": [{"time": 4.75, "subject": "dog"},
                                     {"time": 6.5, "subject": "dog"}]},
        }
        merged = merge_candidates(runs)
        self.assertEqual(len(merged), 2)
        self.assertEqual([len(row["source_proposals"]) for row in merged], [2, 2])


if __name__ == "__main__":
    unittest.main()
