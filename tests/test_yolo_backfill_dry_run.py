import json
import tempfile
import unittest
from pathlib import Path

from scripts.yolo_backfill_dry_run import (
    build_report,
    classify_changes,
    parse_args,
    summarize_records,
    _read_checkpoint,
    _safe_host_name,
    _write_json,
)


class YoloBackfillDryRunTests(unittest.TestCase):
    def test_change_classification_handles_multiple_change_types(self):
        self.assertEqual(
            classify_changes("cat,person", ["dog"]),
            ["label_added", "label_removed", "cat_to_dog", "person_removed"],
        )
        self.assertEqual(classify_changes([], ["cat", "person"]), [
            "label_added", "empty_to_detected", "person_added", "empty_to_cat", "empty_to_person",
        ])
        self.assertEqual(classify_changes("dog", "cat"), ["label_added", "label_removed", "dog_to_cat"])
        self.assertEqual(classify_changes("dog", ["dog"]), ["unchanged"])

    def test_statistics_are_aggregated_globally_and_by_video(self):
        records = [
            {"video_id": "v1", "old_object_labels": [], "new_object_labels": ["dog"], "change_types": classify_changes([], ["dog"])},
            {"video_id": "v1", "old_object_labels": ["cat"], "new_object_labels": ["dog"], "change_types": classify_changes(["cat"], ["dog"])},
            {"video_id": "v2", "old_object_labels": ["person"], "new_object_labels": [], "change_types": classify_changes(["person"], [])},
        ]
        result = summarize_records(records)
        self.assertEqual(result["summary"]["total"], 3)
        self.assertEqual(result["summary"]["changed"], 3)
        self.assertEqual(result["summary"]["empty_to_dog"], 1)
        self.assertEqual(result["summary"]["old_cat_to_new_dog"], 1)
        self.assertEqual(result["summary"]["person_removed"], 1)
        self.assertEqual(result["by_video"]["v1"]["total"], 2)
        self.assertEqual(result["by_video"]["v2"]["detected_to_empty"], 1)

    def test_cli_defaults_and_mismatch_override(self):
        args = parse_args(["--expected-count", "617"])
        self.assertEqual(args.expected_count, 617)
        self.assertFalse(args.allow_count_mismatch)
        args = parse_args(["--allow-count-mismatch", "--checkpoint-every", "5"])
        self.assertTrue(args.allow_count_mismatch)
        self.assertEqual(args.checkpoint_every, 5)

    def test_host_display_redacts_url_credentials(self):
        self.assertEqual(_safe_host_name("https://user:secret@example.test/path"), "example.test")

    def test_report_lists_candidate_categories_without_selecting_canary(self):
        row = {
            "frame_id": "v__f1", "video_id": "v", "s3_key": "frames/v/f1.jpg",
            "old_object_labels": ["cat"], "new_object_labels": ["dog"],
            "object_detections": [{"label": "dog", "confidence": 0.45, "bbox": [1, 2, 3, 4]}],
            "change_types": classify_changes(["cat"], ["dog"]),
        }
        data = {"frames": [row], "success_count": 1, "failures": [], "statistics": summarize_records([row])}
        report = build_report(data)
        self.assertIn("cat/dog 변경 후보 (1)", report)
        self.assertIn("0.4~0.5 confidence detection 후보 (1)", report)
        self.assertIn("canary 미선정", report)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            _write_json(path, data)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["success_count"], 1)
            md_path = Path(directory) / "result.md"
            md_path.write_text(report, encoding="utf-8")
            self.assertIn("유형별 후보", md_path.read_text(encoding="utf-8"))

    def test_checkpoint_is_bound_to_collection_and_model_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            _write_json(path, {"run_config": {"model": "yolov8s.pt"}, "frames": [], "failures": []})
            self.assertEqual(_read_checkpoint(path, {"model": "yolov8s.pt"}), ({}, []))
            with self.assertRaises(ValueError):
                _read_checkpoint(path, {"model": "yolov8n.pt"})


if __name__ == "__main__":
    unittest.main()
