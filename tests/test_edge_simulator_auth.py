import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 영상 분할과 GCS 클라이언트는 이 테스트와 무관하므로 가져오기만 막는다.
with patch.dict(sys.modules, {
    "simulator.chunk_splitter": MagicMock(),
    "pipeline.storage.gcs_uploader": MagicMock(),
}):
    from simulator import edge_simulator


class UploadHeaderTests(unittest.TestCase):
    def test_no_credentials_means_no_headers(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(edge_simulator.build_upload_headers("https://lab.run.app"), {})

    def test_upload_token_is_sent_without_iam_auth(self):
        with patch.dict(os.environ, {"UPLOAD_API_TOKEN": "t"}, clear=True):
            headers = edge_simulator.build_upload_headers("https://lab.run.app")
        self.assertEqual(headers, {"X-Upload-Token": "t"})

    def test_iam_auth_adds_bearer_token_for_service_audience(self):
        with patch.dict(os.environ, {"UPLOAD_API_TOKEN": "t"}, clear=True), \
                patch.object(edge_simulator, "get_id_token", return_value="id-token") as get:
            headers = edge_simulator.build_upload_headers("https://lab.run.app/", iam_auth=True)

        get.assert_called_once_with("https://lab.run.app")
        self.assertEqual(headers["Authorization"], "Bearer id-token")
        self.assertEqual(headers["X-Upload-Token"], "t")

    def test_upload_request_carries_iam_header(self):
        chunks = ["/tmp/a.mp4"]
        with patch.object(edge_simulator, "split_video_into_chunks", return_value=chunks), \
                patch.object(edge_simulator, "upload_video"), \
                patch.object(edge_simulator, "get_id_token", return_value="id-token"), \
                patch.object(edge_simulator.requests, "post") as post, \
                patch.dict(os.environ, {}, clear=True):
            post.return_value.json.return_value = {"status": "received"}
            edge_simulator.run_edge_simulator("clip.mp4", "https://lab.run.app", iam_auth=True)

        self.assertEqual(post.call_args.kwargs["headers"], {"Authorization": "Bearer id-token"})


if __name__ == "__main__":
    unittest.main()
