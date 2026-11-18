import unittest
from unittest.mock import MagicMock, patch

from pipeline.models import yolo_detector
from pipeline.storage import vector_store


class IndexFrameRefreshTests(unittest.TestCase):
    def test_existing_frame_refreshes_metadata_without_reembedding(self):
        collection = MagicMock()
        collection.get.return_value = {"ids": ["clip/f_1.00.jpg"]}
        with patch.object(vector_store, "get_collection", return_value=collection), \
             patch.object(vector_store, "embed_image") as embed:
            vector_store.index_frame(
                "frames/clip/f_1.00.jpg", frame_root="frames", video_id="clip",
                object_labels=["dog"], timestamp=61.0,
            )
        embed.assert_not_called()
        collection.add.assert_not_called()
        metadata = collection.update.call_args.kwargs["metadatas"][0]
        self.assertEqual(metadata["timestamp"], 61.0)
        self.assertEqual(metadata["object_labels"], "dog")


class ClassConfidenceTests(unittest.TestCase):
    def _detect(self, env):
        box = MagicMock()
        box.cls = [0]
        box.conf = [0.55]
        result = MagicMock()
        result.names = {0: "dog"}
        result.boxes = [box]
        model = MagicMock()
        model.predict.return_value = [result]
        with patch.object(yolo_detector, "get_yolo_model", return_value=model), \
             patch.dict("os.environ", env, clear=False):
            return yolo_detector.detect_objects("a.jpg", target_labels={"dog"})

    def test_default_keeps_detection(self):
        self.assertEqual(self._detect({"YOLO_CLASS_CONFIDENCE": ""}), ["dog"])

    def test_class_override_drops_low_confidence(self):
        self.assertEqual(self._detect({"YOLO_CLASS_CONFIDENCE": "dog=0.6"}), [])

    def test_invalid_override_ignored(self):
        self.assertEqual(self._detect({"YOLO_CLASS_CONFIDENCE": "dog=abc,,x"}), ["dog"])


if __name__ == "__main__":
    unittest.main()
