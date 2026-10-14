import sys
import unittest
from unittest.mock import MagicMock, patch

STUBBED_MODULES = [
    "pipeline.video.frame_extractor",
    "pipeline.storage.gcs_uploader",
    "pipeline.storage.vector_store",
    "pipeline.models.yolo_detector",
    "pipeline.events.behavior_event_extractor",
]


class ProcessChunkRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # CLIP/YOLO/ChromaDB 같은 무거운 의존성 없이 Celery 설정만 검증한다.
        cls.modules = patch.dict(sys.modules, {name: MagicMock() for name in STUBBED_MODULES})
        cls.modules.start()
        sys.modules.pop("pipeline.tasks", None)
        from pipeline import tasks

        cls.tasks = tasks

    @classmethod
    def tearDownClass(cls):
        cls.modules.stop()
        sys.modules.pop("pipeline.tasks", None)

    def test_worker_acks_after_completion_and_requeues_lost_work(self):
        task = self.tasks.process_chunk
        conf = self.tasks.celery_app.conf

        self.assertTrue(task.acks_late)
        self.assertTrue(task.reject_on_worker_lost)
        self.assertEqual(conf.worker_prefetch_multiplier, 1)
        self.assertEqual(conf.broker_heartbeat, 0)
        self.assertEqual(task.max_retries, 3)

    def _run(self, error):
        with patch.object(self.tasks, "init_db"), \
                patch.object(self.tasks, "is_chunk_processed", return_value=False), \
                patch.object(self.tasks, "get_object_metadata", side_effect=error) as metadata:
            result = self.tasks.process_chunk.apply(args=["chunks/v/a.mp4"], throw=False)
        return result, metadata

    def test_transient_error_is_retried_up_to_the_limit(self):
        result, metadata = self._run(RuntimeError("chroma unavailable"))

        self.assertEqual(metadata.call_count, 1 + self.tasks.process_chunk.max_retries)
        self.assertTrue(result.failed())

    def test_missing_object_is_not_retried(self):
        result, metadata = self._run(FileNotFoundError("missing"))

        self.assertEqual(metadata.call_count, 1)
        self.assertTrue(result.failed())

    def test_already_processed_chunk_returns_duplicate_without_work(self):
        with patch.object(self.tasks, "init_db"), \
                patch.object(self.tasks, "is_chunk_processed", return_value=True), \
                patch.object(self.tasks, "get_object_metadata") as metadata:
            result = self.tasks.process_chunk.apply(args=["chunks/v/a.mp4"])

        self.assertEqual(result.get()["status"], "duplicate")
        metadata.assert_not_called()


if __name__ == "__main__":
    unittest.main()
