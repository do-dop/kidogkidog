import unittest
from unittest.mock import patch

from api.routers import recordings


class RecordingIndexStatusTests(unittest.TestCase):
    def test_distinguishes_uploaded_chunks_from_searchable_frames(self):
        chunks = [
            {"video_id": "smartbench_0747", "processing": {"state": "pending"}},
            {"video_id": "smartbench_0747", "processing": {"state": "pending"}},
            {"video_id": "smartbench_0608", "processing": {"state": "pending"}},
        ]

        with patch.object(recordings, "list_s3_recording_chunks", return_value={"chunks": chunks}), \
             patch.object(recordings, "get_indexed_frames", side_effect=lambda video_id: [1, 2] if video_id == "smartbench_0608" else []):
            result = recordings.list_recording_index_status()

        self.assertEqual(result["videos"], [
            {"video_id": "smartbench_0608", "chunk_count": 1, "processing_states": {"pending": 1},
             "indexed_frame_count": 2, "ready_for_search": True},
            {"video_id": "smartbench_0747", "chunk_count": 2, "processing_states": {"pending": 2},
             "indexed_frame_count": 0, "ready_for_search": False},
        ])


if __name__ == "__main__":
    unittest.main()
