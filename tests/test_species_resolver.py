import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from db import scenes
from pipeline import species_resolver as resolver
from pipeline import vector_store, yolo_detector


def det(label, confidence, bbox):
    return {"label": label, "confidence": confidence, "bbox": bbox}


class SpeciesResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temp.name) / "frame.jpg"
        Image.new("RGB", (100, 100), "white").save(self.image_path)

    def tearDown(self):
        self.temp.cleanup()

    def resolve(self, detections, decisions):
        return resolver.resolve_detection_result(
            self.image_path,
            {"object_detections": detections,
             "object_labels": yolo_detector.build_detection_result(detections)["object_labels"]},
            enabled=True,
            scorer=lambda _crop: decisions.pop(0),
        )

    def test_dog_raw_detection_is_preserved_when_clip_resolves_cat(self):
        raw = [det("dog", .88, [10, 10, 60, 60])]
        result = self.resolve(raw, [(.31, .24)])
        self.assertEqual(result["object_detections"], raw)
        self.assertEqual(result["species_resolution"][0]["yolo_label"], "dog")
        self.assertEqual(result["species_resolution"][0]["resolved_label"], "cat")
        self.assertEqual(result["object_labels"], ["cat"])

    def test_cat_raw_detection_can_resolve_dog(self):
        result = self.resolve([det("cat", .73, [10, 10, 60, 60])], [(.22, .32)])
        self.assertEqual(result["object_detections"][0]["label"], "cat")
        self.assertEqual(result["object_labels"], ["dog"])

    def test_low_confidence_dog_candidate_is_recovered(self):
        result = self.resolve([det("dog", .42, [10, 10, 60, 60])], [(.22, .32)])
        self.assertEqual(result["object_labels"], ["dog"])

    def test_low_confidence_cat_candidate_can_resolve_dog(self):
        result = self.resolve([det("cat", .41, [10, 10, 60, 60])], [(.22, .32)])
        self.assertEqual(result["object_detections"][0]["label"], "cat")
        self.assertEqual(result["object_labels"], ["dog"])

    def test_no_animal_bbox_does_not_run_full_frame_clip(self):
        scorer = unittest.mock.Mock()
        result = resolver.resolve_detection_result(
            self.image_path,
            {"object_detections": [det("bowl", .7, [10, 10, 60, 60])],
             "object_labels": ["bowl"]},
            enabled=True,
            scorer=scorer,
        )
        scorer.assert_not_called()
        self.assertEqual(result["species_resolution"], [])
        self.assertEqual(result["object_labels"], ["bowl"])

    def test_two_boxes_can_resolve_to_both_species(self):
        result = self.resolve([
            det("dog", .88, [5, 5, 45, 45]),
            det("cat", .76, [50, 50, 95, 95]),
        ], [(.31, .24), (.20, .33)])
        self.assertEqual(result["object_labels"], ["cat", "dog"])
        self.assertEqual([x["resolved_label"] for x in result["species_resolution"]], ["cat", "dog"])

    def test_regular_object_labels_keep_threshold_policy(self):
        result = self.resolve([
            det("bowl", .7, [1, 1, 20, 20]),
            det("bowl", .45, [30, 30, 50, 50]),
        ], [])
        self.assertEqual(result["object_labels"], ["bowl"])
        self.assertEqual(len(result["object_detections"]), 2)

    def test_feature_flag_off_preserves_legacy_labels_and_skips_clip(self):
        with patch.dict(os.environ, {resolver.SPECIES_RESOLVER_ENV_VAR: "false"}):
            self.assertFalse(resolver.clip_species_resolver_enabled())
            scorer = unittest.mock.Mock()
            raw = [det("cat", .73, [10, 10, 60, 60])]
            result = resolver.resolve_detection_result(
                self.image_path,
                {"object_detections": raw, "object_labels": ["cat"]},
                scorer=scorer,
            )
        scorer.assert_not_called()
        self.assertEqual(result["object_labels"], ["cat"])
        self.assertEqual(result["species_resolution"], [])

    def test_bbox_padding_is_ten_percent_and_clamped(self):
        self.assertEqual(resolver.DEFAULT_BBOX_PADDING, .10)
        self.assertEqual(resolver._padded_crop_box([-5, 10, 55, 60], (100, 100), .10), (0, 5, 61, 65))

    def test_chroma_species_resolution_json_round_trip(self):
        values = [{"bbox": [1, 2, 30, 40], "yolo_label": "dog",
                   "yolo_confidence": .88, "clip_label": "cat",
                   "clip_cat_similarity": .31, "clip_dog_similarity": .24,
                   "resolved_label": "cat"}]
        metadata = {"species_resolution_json": vector_store._normalize_species_resolution(values)}
        self.assertEqual(vector_store._parse_species_resolution(metadata), values)
        self.assertEqual(vector_store._parse_species_resolution({}), [])

    def test_indexed_frame_reads_species_resolution_metadata(self):
        values = [{"bbox": [1, 2, 30, 40], "resolved_label": "cat"}]
        class Collection:
            def get(self, include=None):
                return {"ids": ["v__frame_1.00"], "metadatas": [{
                    "video_id": "v", "frame_path": "frame.jpg", "timestamp": 1.0,
                    "species_resolution_json": json.dumps(values),
                }]}
        with patch.object(vector_store, "get_collection", return_value=Collection()):
            frames = vector_store.get_indexed_frames(video_id="v")
        self.assertEqual(frames[0]["species_resolution"], values)

    def test_index_frame_writes_species_resolution_as_chroma_json_scalar(self):
        class Collection:
            def __init__(self):
                self.metadata = None
            def get(self, ids):
                return {"ids": []}
            def add(self, ids, embeddings, metadatas):
                self.metadata = metadatas[0]

        collection = Collection()
        values = [{"bbox": [1, 2, 30, 40], "resolved_label": "cat"}]
        with patch.object(vector_store, "get_collection", return_value=collection), \
             patch.object(vector_store, "embed_image", return_value=[.1, .2]):
            vector_store.index_frame(
                self.image_path,
                frame_root=self.image_path.parent,
                video_id="v",
                object_labels=["cat"],
                object_detections=[det("dog", .8, [1, 2, 30, 40])],
                species_resolution=values,
                timestamp=0.0,
            )
        self.assertEqual(json.loads(collection.metadata["species_resolution_json"]), values)

    def test_scene_species_resolution_is_serialized_only_when_provided(self):
        cursor = unittest.mock.Mock()
        conn = unittest.mock.Mock()
        conn.cursor.return_value = cursor
        values = [{"resolved_label": "cat"}]
        with patch.object(scenes, "connect", return_value=conn):
            scenes.insert_scene("v", 1, 2, object_labels='["cat"]',
                                object_detections=[], species_resolution=values)
        sql, params = cursor.execute.call_args.args
        self.assertIn("species_resolution_json", sql)
        self.assertEqual(json.loads(params[5]), values)
        conn.commit.assert_called_once()

    def test_scene_storage_without_resolution_keeps_legacy_sql(self):
        cursor = unittest.mock.Mock()
        conn = unittest.mock.Mock()
        conn.cursor.return_value = cursor
        with patch.object(scenes, "connect", return_value=conn):
            scenes.insert_scene("v", 1, 2, object_labels='["dog"]', object_detections=[])
        sql, _ = cursor.execute.call_args.args
        self.assertNotIn("species_resolution_json", sql)


if __name__ == "__main__":
    unittest.main()
