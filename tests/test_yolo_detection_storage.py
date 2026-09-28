import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import chromadb
from db import scenes
from pipeline import behavior_event_extractor as behavior
from pipeline import vector_store
from pipeline import yolo_detector


def detection(label, confidence, bbox):
    return {"label": label, "confidence": confidence, "bbox": bbox}


class _Tensor(list):
    def tolist(self):
        return list(self)


class _Box:
    def __init__(self, class_id, confidence, bbox):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [_Tensor(bbox)]


class _FakeCollection:
    def __init__(self):
        self.rows = {}

    def get(self, ids, include=None):
        found = [frame_id for frame_id in ids if frame_id in self.rows]
        return {
            "ids": found,
            "metadatas": [self.rows[frame_id] for frame_id in found],
        }

    def add(self, ids, embeddings, metadatas):
        for frame_id, metadata in zip(ids, metadatas):
            self.rows[frame_id] = dict(metadata)

    def update(self, ids, metadatas):
        for frame_id, metadata in zip(ids, metadatas):
            self.rows[frame_id] = dict(metadata)


class YoloDetectionStorageTests(unittest.TestCase):
    def test_separate_collection_and_label_thresholds(self):
        result = yolo_detector.build_detection_result([
            detection("dog", 0.83, [1, 2, 30, 40]),
            detection("cat", 0.44, [5, 6, 25, 35]),
        ])
        self.assertEqual(result["object_labels"], ["dog"])
        self.assertEqual(result["object_detections"], [
            detection("dog", 0.83, [1.0, 2.0, 30.0, 40.0]),
            detection("cat", 0.44, [5.0, 6.0, 25.0, 35.0]),
        ])

    def test_detection_below_label_threshold_is_retained_only_as_detection(self):
        result = yolo_detector.build_detection_result([
            detection("dog", 0.42, [10, 20, 100, 200]),
        ])
        self.assertEqual(result["object_labels"], [])
        self.assertEqual(len(result["object_detections"]), 1)
        self.assertEqual(result["object_detections"][0]["confidence"], 0.42)

    def test_detection_above_label_threshold_appears_in_both(self):
        result = yolo_detector.build_detection_result([
            detection("cat", 0.81, [10, 20, 100, 200]),
        ])
        self.assertEqual(result["object_labels"], ["cat"])
        self.assertEqual(result["object_detections"][0]["label"], "cat")

    def test_repeated_class_keeps_each_bbox_but_unique_legacy_label(self):
        result = yolo_detector.build_detection_result([
            detection("cat", 0.81, [0, 0, 20, 20]),
            detection("cat", 0.72, [30, 40, 60, 80]),
        ])
        self.assertEqual(result["object_labels"], ["cat"])
        self.assertEqual(len(result["object_detections"]), 2)

    def test_predict_collects_at_point_four_and_keeps_xyxy_pixel_bbox(self):
        model = Mock()
        model.predict.return_value = [Mock(
            names={0: "dog", 1: "cat"},
            boxes=[
                _Box(0, 0.42, [1, 2, 30, 40]),
                _Box(1, 0.81, [5, 6, 25, 35]),
                _Box(0, 0.39, [0, 0, 2, 2]),
            ],
        )]
        with patch.object(yolo_detector, "get_yolo_model", return_value=model):
            result = yolo_detector.detect_objects_with_details("frame.jpg")
        model.predict.assert_called_once()
        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.4)
        self.assertEqual(result["object_labels"], ["cat"])
        self.assertEqual(len(result["object_detections"]), 2)
        self.assertEqual(result["object_detections"][0]["bbox"], [1.0, 2.0, 30.0, 40.0])

    def test_default_model_and_environment_override(self):
        yolo_detector.get_yolo_model.cache_clear()
        try:
            with patch.dict(os.environ, {}, clear=True), patch.object(yolo_detector, "YOLO") as model:
                yolo_detector.get_yolo_model()
                model.assert_called_once_with("yolov8s.pt")
            yolo_detector.get_yolo_model.cache_clear()
            with patch.dict(os.environ, {"YOLO_MODEL": "custom.pt"}), patch.object(yolo_detector, "YOLO") as model:
                yolo_detector.get_yolo_model()
                model.assert_called_once_with("custom.pt")
        finally:
            yolo_detector.get_yolo_model.cache_clear()

    def test_chroma_serializes_detection_metadata_and_update_preserves_other_fields(self):
        collection = _FakeCollection()
        detections = [detection("dog", 0.83, [1, 2, 30, 40])]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame = root / "video" / "frame_1.00.jpg"
            frame.parent.mkdir()
            frame.write_bytes(b"frame")
            with patch.object(vector_store, "get_collection", return_value=collection), \
                 patch.object(vector_store, "embed_image", return_value=[0.1, 0.2]):
                frame_id = vector_store.index_frame(
                    frame,
                    frame_root=root,
                    video_id="video",
                    object_labels=["dog"],
                    object_detections=detections,
                    object_detection_model="yolov8s.pt",
                )
                metadata = collection.get(ids=[frame_id])["metadatas"][0]
                self.assertEqual(metadata["object_labels"], "dog")
                self.assertEqual(json.loads(metadata["object_detections_json"]), detections)
                self.assertEqual(vector_store._parse_object_detections(metadata), detections)
                self.assertEqual(vector_store._parse_object_detections({"object_labels": "dog"}), [])
                self.assertEqual(metadata["object_detection_model"], "yolov8s.pt")

                vector_store.update_frame_detection_metadata(
                    frame_id,
                    ["cat"],
                    [detection("cat", 0.81, [4, 5, 40, 50])],
                    detection_model="yolov8s.pt",
                    detection_confidence_threshold=0.4,
                    object_labels_confidence_threshold=0.5,
                )
                updated = collection.get(ids=[frame_id])["metadatas"][0]
                self.assertEqual(updated["video_id"], "video")
                self.assertEqual(updated["object_labels"], "cat")
                self.assertEqual(json.loads(updated["object_detections_json"])[0]["bbox"], [4, 5, 40, 50])
                self.assertEqual(updated["object_detection_confidence_threshold"], 0.4)

    def test_real_chroma_accepts_json_scalar_and_rejects_nested_detection_metadata(self):
        client = chromadb.Client()
        collection = client.create_collection(f"detection-probe-{uuid.uuid4().hex}")
        values = [detection("dog", 0.83, [1, 2, 30, 40])]
        resolutions = [{"bbox": [1, 2, 30, 40], "resolved_label": "cat"}]
        collection.add(
            ids=["frame-1"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{
                "video_id": "video",
                "object_labels": "dog",
                "object_detections_json": json.dumps(values),
                "species_resolution_json": json.dumps(resolutions),
            }],
        )
        result = collection.get(ids=["frame-1"], include=["metadatas"])
        self.assertEqual(json.loads(result["metadatas"][0]["object_detections_json"]), values)
        self.assertEqual(
            vector_store._parse_species_resolution(result["metadatas"][0]),
            resolutions,
        )
        with self.assertRaises(ValueError):
            collection.add(
                ids=["nested-frame"],
                embeddings=[[0.1, 0.2]],
                metadatas=[{"object_detections": values}],
            )
        with patch.object(vector_store, "get_collection", return_value=collection):
            vector_store.update_frame_detection_metadata(
                "frame-1",
                ["cat"],
                [detection("cat", 0.81, [4, 5, 40, 50])],
                detection_model="yolov8s.pt",
            )
        updated = collection.get(ids=["frame-1"], include=["metadatas", "embeddings"])
        self.assertEqual(updated["metadatas"][0]["video_id"], "video")
        self.assertEqual(updated["metadatas"][0]["object_labels"], "cat")
        for actual, expected in zip(updated["embeddings"][0].tolist(), [0.1, 0.2]):
            self.assertAlmostEqual(actual, expected, places=6)
        client.delete_collection(collection.name)

    def test_scene_storage_passes_detection_json_without_db_migration(self):
        cursor = Mock()
        conn = Mock()
        conn.cursor.return_value = cursor
        value = [detection("cat", 0.81, [1, 2, 30, 40])]
        with patch.object(scenes, "connect", return_value=conn):
            scenes.insert_scene("video", 1.0, 2.0, object_labels='["cat"]', object_detections=value)
        sql, params = cursor.execute.call_args.args
        self.assertIn("object_detections_json", sql)
        self.assertEqual(json.loads(params[4]), value)
        conn.commit.assert_called_once()

    def test_behavior_event_source_frames_keep_detection_details(self):
        frame = {
            "frame_id": "frame-1",
            "video_id": "video",
            "timestamp": 1.0,
            "object_labels": ["dog"],
            "object_detections": [detection("dog", 0.83, [1, 2, 30, 40])],
            "species_resolution": [{"resolved_label": "dog", "yolo_label": "dog"}],
        }
        event = behavior._normalize_behavior_event(
            {"action": "움직임", "summary": "움직임"},
            [frame],
            [frame],
        )
        self.assertEqual(event["source_frames"][0]["object_labels"], ["dog"])
        self.assertEqual(event["source_frames"][0]["object_detections"], [
            detection("dog", 0.83, [1.0, 2.0, 30.0, 40.0]),
        ])
        self.assertEqual(event["source_frames"][0]["species_resolution"], frame["species_resolution"])


if __name__ == "__main__":
    unittest.main()
