import unittest
from unittest.mock import MagicMock, patch

from pipeline.video import frame_extractor


class FrameExtractorMotionFlagTests(unittest.TestCase):
    def test_motion_frames_are_flagged(self):
        capture = MagicMock()
        capture.get.return_value = 30.0
        capture.read.return_value = (True, object())

        with patch.object(frame_extractor, "detect_motion", return_value=[(2.0, 6.0)]), \
             patch.object(frame_extractor.cv2, "VideoCapture", return_value=capture), \
             patch.object(frame_extractor.cv2, "imwrite", return_value=True), \
             patch.object(frame_extractor.os, "makedirs"):
            frames = frame_extractor.extract_frames("clip.mp4", output_dir="/tmp/frames")

        self.assertEqual([frame["timestamp"] for frame in frames], [2.0, 4.0, 6.0])
        self.assertTrue(all(frame["motion_detected"] is True for frame in frames))

    def test_whole_chunk_fallback_frames_are_not_flagged_as_motion(self):
        fallback = [
            {"frame_path": "a.jpg", "timestamp": 0.0, "motion_start": 0.0,
             "motion_end": 2.0, "segment_id": 0, "motion_detected": False},
        ]
        with patch.object(frame_extractor, "detect_motion", return_value=[]), \
             patch.object(frame_extractor, "_extract_frames_with_ffmpeg", return_value=fallback), \
             patch.object(frame_extractor.os, "makedirs"):
            frames = frame_extractor.extract_frames("clip.mp4", output_dir="/tmp/frames")
        self.assertTrue(all(frame["motion_detected"] is False for frame in frames))

    def test_ffmpeg_fallback_builder_marks_frames_not_motion(self):
        with patch.object(frame_extractor.subprocess, "run"), \
             patch.object(frame_extractor, "Path") as path_cls:
            path_cls.return_value.glob.return_value = []
            path_cls.return_value.__truediv__.return_value = "out"
            frames = frame_extractor._extract_frames_with_ffmpeg("clip.mp4", "/tmp/frames", "clip", 0.5)
        self.assertEqual(frames, [])


if __name__ == "__main__":
    unittest.main()
