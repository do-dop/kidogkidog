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
        box.xyxy = [MagicMock(tolist=lambda: [0, 0, 10, 10])]
        result = MagicMock()
        result.names = {0: "dog"}
        result.boxes = [box]
        model = MagicMock()
        model.predict.return_value = [result]
        with patch.object(yolo_detector, "get_yolo_model", return_value=model), \
             patch.dict("os.environ", {"YOLO_VERIFY_DOG": "0", **env}, clear=False):
            return yolo_detector.detect_objects("a.jpg", target_labels={"dog"})

    def test_default_keeps_detection(self):
        self.assertEqual(self._detect({"YOLO_CLASS_CONFIDENCE": ""}), ["dog"])

    def test_class_override_drops_low_confidence(self):
        self.assertEqual(self._detect({"YOLO_CLASS_CONFIDENCE": "dog=0.6"}), [])

    def test_invalid_override_ignored(self):
        self.assertEqual(self._detect({"YOLO_CLASS_CONFIDENCE": "dog=abc,,x"}), ["dog"])


class DogVerificationTests(unittest.TestCase):
    def _detect(self, conf, looks_like_dog, env=None):
        box = MagicMock()
        box.cls = [0]
        box.conf = [conf]
        box.xyxy = [MagicMock(tolist=lambda: [0, 0, 10, 10])]
        result = MagicMock()
        result.names = {0: "dog"}
        result.boxes = [box]
        model = MagicMock()
        model.predict.return_value = [result]
        with patch.object(yolo_detector, "get_yolo_model", return_value=model), \
             patch.object(yolo_detector.species_verifier, "looks_like_dog",
                          return_value=looks_like_dog) as verify, \
             patch.dict("os.environ", env or {}, clear=False):
            labels = yolo_detector.detect_objects("a.jpg", target_labels={"dog"})
        return labels, verify

    def test_low_confidence_non_dog_crop_is_dropped(self):
        labels, verify = self._detect(0.55, looks_like_dog=False)
        self.assertEqual(labels, [])
        verify.assert_called_once()

    def test_low_confidence_dog_crop_is_kept(self):
        self.assertEqual(self._detect(0.55, looks_like_dog=True)[0], ["dog"])

    def test_confident_detection_skips_verification(self):
        labels, verify = self._detect(0.8, looks_like_dog=False)
        self.assertEqual(labels, ["dog"])
        verify.assert_not_called()

    def test_verification_can_be_disabled(self):
        labels, verify = self._detect(0.55, looks_like_dog=False, env={"YOLO_VERIFY_DOG": "0"})
        self.assertEqual(labels, ["dog"])
        verify.assert_not_called()

class DogPromptTests(unittest.TestCase):
    def test_cat_is_among_the_alternatives(self):
        verifier = yolo_detector.species_verifier
        self.assertIn("a photo of a cat", verifier.OTHER_PROMPTS)
        # dog_probability는 앞쪽 DOG_PROMPTS 개수만큼을 강아지 확률로 합산한다.
        self.assertFalse(set(verifier.DOG_PROMPTS) & set(verifier.OTHER_PROMPTS))


if __name__ == "__main__":
    unittest.main()
