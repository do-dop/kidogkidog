import unittest
import json
from pathlib import Path

from scripts.yolo_clip_full_backfill import (
    _change_statistics,
    _deep_equal,
    _event_updates,
    _labels,
    _load_manifests,
)


class YoloClipFullBackfillTests(unittest.TestCase):
    def test_manifest_is_exact_approved_240_frame_allowlist(self):
        frames, shadow, _, _ = _load_manifests()
        self.assertEqual(len(frames), 240)
        self.assertEqual(len(shadow), 240)
        self.assertEqual(len({row["frame_id"] for row in frames}), 240)

    def test_label_parser_accepts_chroma_and_mysql_encodings(self):
        self.assertEqual(_labels("cat,dog"), ["cat", "dog"])
        self.assertEqual(_labels('["dog", "cat"]'), ["cat", "dog"])
        self.assertEqual(_labels(["cat", "cat"]), ["cat"])

    def test_change_statistics_tracks_species_correction_and_low_confidence_recovery(self):
        frames = [{"frame_id": "f1", "video_id": "dog_escape"}]
        old_chroma = {"f1": {"metadata": {"object_labels": "dog"}}}
        computed = {
            "f1": {
                "object_labels": ["cat"],
                "object_detections": [{"label": "dog", "confidence": 0.426, "bbox": [1, 2, 3, 4]}],
                "species_resolution": [{"bbox": [1, 2, 3, 4], "resolved_label": "cat"}],
            }
        }
        stats = _change_statistics(frames, old_chroma, computed, {})
        self.assertEqual(stats["overall"]["cat_to_dog"], 0)
        self.assertEqual(stats["overall"]["dog_to_cat"], 1)
        self.assertEqual(stats["overall"]["low_confidence_animal_recovery"], 1)

    def test_json_comparison_allows_storage_rounding_but_not_real_differences(self):
        self.assertTrue(_deep_equal({"x": 0.1234567890123}, {"x": 0.1234567890124}))
        self.assertFalse(_deep_equal({"x": 0.1}, {"x": 0.2}))

    def test_event_patch_changes_eligible_source_and_preserves_ineligible_source(self):
        eligible, _, _, _ = _load_manifests()
        target = next(row for row in eligible if row["video_id"] == "two_dogs")
        ineligible_doc = json.loads(Path("docs/evaluations/yolo-backfill-ineligible-2026-09-27.json").read_text())
        stale = next(row for row in ineligible_doc["frames"] if row["video_id"] == "two_dogs")
        stale_source = {"frame_id": stale["frame_id"], "object_labels": ["dog"], "s3_key": stale["s3_key"]}
        eligible_source = {"frame_id": target["frame_id"], "object_labels": ["cat"], "s3_key": target["s3_key"]}
        sources = [eligible_source, stale_source]
        event = {
            "row": {"id": 99, "video_id": "two_dogs", "source_frames_json": json.dumps(sources)},
            "matched_sources": [(0, target["frame_id"])],
            "ambiguous": False,
        }
        computed = {target["frame_id"]: {
            "object_labels": ["dog"],
            "object_detections": [{"label": "dog", "confidence": 0.8, "bbox": [1, 2, 3, 4]}],
            "species_resolution": [{"resolved_label": "dog"}],
        }}
        patches, skipped = _event_updates([event], computed, {target["frame_id"]})
        updated = patches[99]["source_frames"]
        self.assertEqual(updated[0]["object_labels"], ["dog"])
        self.assertEqual(updated[1], stale_source)
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
