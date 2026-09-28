import unittest

from scripts.yolo_clip_canary_backfill import (
    _chroma_matches_backup,
    _deep_equal,
    _load_targets,
)


class YoloClipCanaryBackfillTests(unittest.TestCase):
    def test_fixed_canaries_are_the_reviewed_eligible_shadow_intersection(self):
        canary, eligible, shadow = _load_targets()
        self.assertEqual(len(canary), 15)
        for frame in canary:
            self.assertIn(frame["frame_id"], eligible)
            self.assertIn(frame["frame_id"], shadow)

    def test_json_float_round_trip_tolerance(self):
        self.assertTrue(_deep_equal({"score": 0.1234567890123456}, {"score": 0.1234567890123457}))
        self.assertFalse(_deep_equal({"score": 0.1}, {"score": 0.2}))

    def test_chroma_rollback_accepts_only_empty_sentinels_for_new_keys(self):
        saved = {
            "metadata": {"video_id": "v", "object_labels": "dog"},
            "document": "same document",
            "embedding_float32_bytes_hex": "abcd",
        }
        current = {
            "metadata": {**saved["metadata"], "object_detections_json": "", "species_resolution_json": ""},
            "document": saved["document"],
            "embedding_float32_bytes_hex": saved["embedding_float32_bytes_hex"],
        }
        self.assertTrue(_chroma_matches_backup(current, saved))
        current["metadata"]["object_detections_json"] = "[]"
        self.assertFalse(_chroma_matches_backup(current, saved))


if __name__ == "__main__":
    unittest.main()
