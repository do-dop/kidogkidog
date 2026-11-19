import unittest
from unittest.mock import MagicMock, patch

from pipeline.video import frame_extractor


class IdleSampleTimesTests(unittest.TestCase):
    def test_idle_frames_on_opposite_sides_of_motion_are_separate_groups(self):
        self.assertEqual(frame_extractor._idle_time_groups([0, 1, 2, 9, 10, 11], 1),
                         [[0, 1, 2], [9, 10, 11]])

    def test_disabled_when_interval_is_zero(self):
        self.assertEqual(frame_extractor.idle_sample_times(30, [(5, 10)], 0), [])

    def test_skips_times_near_motion_segments(self):
        times = frame_extractor.idle_sample_times(20, [(8, 12)], 5, pad=1.0)
        self.assertEqual(times, [0.0, 5.0, 15.0, 20.0])

    def test_no_motion_means_every_interval(self):
        self.assertEqual(frame_extractor.idle_sample_times(10, [], 5), [0.0, 5.0, 10.0])


class ExtractFramesIdleTests(unittest.TestCase):
    def _capture(self):
        capture = MagicMock()
        capture.get.side_effect = lambda prop: {
            frame_extractor.cv2.CAP_PROP_FPS: 30.0,
            frame_extractor.cv2.CAP_PROP_FRAME_COUNT: 600.0,  # 20초
        }.get(prop, 0)
        capture.read.return_value = (True, object())
        return capture

    def _run(self, interval):
        with patch.object(frame_extractor, "detect_motion", return_value=[(8.0, 12.0)]), \
             patch.object(frame_extractor.cv2, "VideoCapture", return_value=self._capture()), \
             patch.object(frame_extractor.cv2, "imwrite", return_value=True), \
             patch.object(frame_extractor.os, "makedirs"):
            return frame_extractor.extract_frames(
                "clip.mp4", output_dir="/tmp/frames", idle_sample_interval=interval
            )

    def test_explicit_zero_keeps_motion_only(self):
        frames = self._run(0)
        self.assertEqual([f["timestamp"] for f in frames], [8.0, 9.0, 10.0, 11.0, 12.0])
        self.assertTrue(all(f["motion_detected"] for f in frames))

    def test_default_samples_outside_partial_motion(self):
        with patch.dict(frame_extractor.os.environ, {}, clear=True):
            frames = self._run(None)
        idle = [f for f in frames if not f["motion_detected"]]
        self.assertIn(0.0, [f["timestamp"] for f in idle])
        self.assertIn(14.0, [f["timestamp"] for f in idle])
        self.assertEqual(idle[0]["segment_id"], idle[1]["segment_id"])

    def test_idle_frames_are_added_and_flagged(self):
        frames = self._run(5)
        self.assertEqual([f["timestamp"] for f in frames], [0.0, 5.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0])
        idle = [f for f in frames if not f["motion_detected"]]
        self.assertEqual(len(idle), 4)
        self.assertEqual(len({f["segment_id"] for f in idle}), 2)  # 12초 창으로 묶어 전환을 분석
        self.assertEqual(idle[0]["segment_id"], idle[1]["segment_id"])

class IdleFallbackTests(unittest.TestCase):
    def _run(self, idle, fps=1.0):
        with patch.object(frame_extractor, "detect_motion", return_value=[]), \
             patch.object(frame_extractor, "_extract_frames_with_ffmpeg", return_value=[]) as ff, \
             patch.object(frame_extractor.os, "makedirs"):
            frame_extractor.extract_frames("clip.mp4", output_dir="/tmp/f", fps=fps, idle_sample_interval=idle)
        return ff.call_args.kwargs["fps"]

    def test_no_motion_chunk_follows_idle_interval(self):
        self.assertAlmostEqual(self._run(6), 1 / 6)

    def test_no_motion_chunk_keeps_fps_when_idle_disabled(self):
        self.assertEqual(self._run(0), 1.0)

    def test_idle_interval_never_raises_fps(self):
        self.assertEqual(self._run(0.5, fps=0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
