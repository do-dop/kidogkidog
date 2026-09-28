import unittest

from scripts.cats_5min_legacy_cleanup import (
    EXPECTED_CHROMA,
    EXPECTED_SCENES,
    select_exact_targets,
    validate_delete_preconditions,
)


class Cats5MinLegacyCleanupTests(unittest.TestCase):
    def test_exact_scope_selects_only_cats_5min(self):
        rows = [
            {"frame_id": "cats_5min__a", "video_id": "cats_5min"},
            {"frame_id": "dog_escape__b", "video_id": "dog_escape"},
        ]
        self.assertEqual(select_exact_targets(rows), [rows[0]])

    def test_duplicate_or_missing_frame_ids_rejected(self):
        with self.assertRaises(ValueError):
            select_exact_targets([
                {"frame_id": "same", "video_id": "cats_5min"},
                {"frame_id": "same", "video_id": "cats_5min"},
            ])
        with self.assertRaises(ValueError):
            select_exact_targets([{"video_id": "cats_5min"}])

    def test_wrong_expected_count_stops_before_delete(self):
        with self.assertRaises(RuntimeError):
            validate_delete_preconditions(
                {"chroma": EXPECTED_CHROMA - 1, "scenes": EXPECTED_SCENES,
                 "behavior_events": 0, "gcs_chunks": 0},
                backup_exists=True,
            )

    def test_missing_backup_stops_before_delete(self):
        with self.assertRaises(RuntimeError):
            validate_delete_preconditions(
                {"chroma": EXPECTED_CHROMA, "scenes": EXPECTED_SCENES,
                 "behavior_events": 0, "gcs_chunks": 0},
                backup_exists=False,
            )

    def test_expected_preconditions_allow_exact_cleanup(self):
        validate_delete_preconditions(
            {"chroma": EXPECTED_CHROMA, "scenes": EXPECTED_SCENES,
             "behavior_events": 0, "gcs_chunks": 0},
            backup_exists=True,
        )


if __name__ == "__main__":
    unittest.main()
