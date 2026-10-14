import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from kombu.exceptions import OperationalError

from api.routers.upload import router


class UploadRouterTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        self.token = patch.dict(os.environ, {"UPLOAD_API_TOKEN": "lab-token"})
        self.token.start()
        self.addCleanup(self.token.stop)

    @patch("api.routers.upload.celery_app.send_task")
    @patch("api.routers.upload.object_exists")
    def test_rejects_unauthorized_request_before_gcs_lookup(self, exists, send_task):
        response = self.client.post("/upload", json={"chunk_path": "chunks/test/video.mp4"})

        self.assertEqual(response.status_code, 401)
        exists.assert_not_called()
        send_task.assert_not_called()

    @patch("api.routers.upload.celery_app.send_task")
    @patch("api.routers.upload.object_exists")
    def test_rejects_parent_path_before_gcs_lookup(self, exists, send_task):
        response = self.client.post(
            "/upload",
            json={"chunk_path": "chunks/../private.mp4"},
            headers={"X-Upload-Token": "lab-token"},
        )

        self.assertEqual(response.status_code, 400)
        exists.assert_not_called()
        send_task.assert_not_called()

    @patch("api.routers.upload.celery_app.send_task")
    @patch("api.routers.upload.object_exists", return_value=False)
    def test_rejects_missing_gcs_object(self, exists, send_task):
        response = self.client.post(
            "/upload",
            json={"chunk_path": "chunks/test/video.mp4"},
            headers={"X-Upload-Token": "lab-token"},
        )

        self.assertEqual(response.status_code, 404)
        exists.assert_called_once_with("chunks/test/video.mp4")
        send_task.assert_not_called()

    @patch("api.routers.upload.celery_app.send_task", side_effect=OperationalError("offline"))
    @patch("api.routers.upload.object_exists", return_value=True)
    def test_reports_unavailable_broker(self, exists, send_task):
        response = self.client.post(
            "/upload",
            json={"chunk_path": "chunks/test/video.mp4"},
            headers={"X-Upload-Token": "lab-token"},
        )

        self.assertEqual(response.status_code, 503)
        send_task.assert_called_once_with(
            "pipeline.tasks.process_chunk", args=["chunks/test/video.mp4"]
        )

    @patch("api.routers.upload.celery_app.send_task", return_value=SimpleNamespace(id="task-123"))
    @patch("api.routers.upload.object_exists", return_value=True)
    def test_returns_task_id_after_enqueue(self, exists, send_task):
        response = self.client.post(
            "/upload",
            json={"chunk_path": "chunks/test/video.mp4"},
            headers={"X-Upload-Token": "lab-token"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task_id"], "task-123")
        self.assertEqual(response.json()["s3_path"], "chunks/test/video.mp4")
        send_task.assert_called_once_with(
            "pipeline.tasks.process_chunk", args=["chunks/test/video.mp4"]
        )


if __name__ == "__main__":
    unittest.main()
