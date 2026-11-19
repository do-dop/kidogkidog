import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import recordings


class RecordingListVisibilityTests(unittest.TestCase):
    def test_replaced_sofa_recording_is_hidden_only_from_unfiltered_list(self):
        app = FastAPI()
        app.include_router(recordings.router)
        client = TestClient(app)
        chunks = [
            {"video_id": "smartbench_0518", "s3_key": "chunks/smartbench_0518/old.mp4"},
            {"video_id": "smartbench_0518_rerun", "s3_key": "chunks/smartbench_0518_rerun/new.mp4"},
            {"video_id": "smartbench_0608", "s3_key": "chunks/smartbench_0608/pool.mp4"},
        ]
        payload = {"status": "ok", "prefix": "chunks/", "count": 3, "chunks": chunks}
        with patch.object(recordings, "list_s3_recording_chunks", return_value=payload):
            visible = client.get("/recordings/chunks").json()
            direct = client.get("/recordings/chunks", params={"video_id": "smartbench_0518"}).json()
        self.assertEqual(visible["count"], 2)
        self.assertEqual([item["video_id"] for item in visible["chunks"]],
                         ["smartbench_0518_rerun", "smartbench_0608"])
        self.assertEqual(direct, payload)

    def test_original_stays_visible_if_replacement_is_absent(self):
        app = FastAPI()
        app.include_router(recordings.router)
        client = TestClient(app)
        payload = {"status": "ok", "count": 1, "chunks": [{"video_id": "smartbench_0518"}]}
        with patch.object(recordings, "list_s3_recording_chunks", return_value=payload):
            self.assertEqual(client.get("/recordings/chunks").json()["count"], 1)


if __name__ == "__main__":
    unittest.main()
