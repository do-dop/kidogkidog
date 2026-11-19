import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from api.services import recording_service
from pipeline.storage.capture_time import trusted_capture_time
from pipeline.storage import gcs_uploader
from pipeline.storage import vector_store
from pipeline.search.sequence_query import run_sequence_query


class CaptureTimeTests(unittest.TestCase):
    def test_filename_and_storage_update_do_not_become_capture_time(self):
        key = "chunks/smartbench_0608/petcam_20261105_073500_000.mp4"
        updated = datetime(2026, 11, 5, 7, 35, tzinfo=timezone.utc)
        self.assertIsNone(recording_service.get_chunk_recorded_at(key, updated))
        self.assertIsNone(trusted_capture_time({"source_type": "imported", "captured_at": "2026-11-05T07:35:00+09:00"}))
        self.assertIsNone(trusted_capture_time({"source_type": "petcam", "captured_at": "2026-11-05T07:35:00"}))
        self.assertIsNone(vector_store._frame_recorded_at({
            "s3_key": key.replace("chunks/", "frames/").replace(".mp4", "_frame_2.00.jpg"),
            "recorded_at": "2026-11-05T07:35:02+09:00",
        }))

    def test_sequence_uses_video_position_for_untrusted_petcam_filename(self):
        result = run_sequence_query("언제 넘어졌어?", video_id="imported", events=[{
            "id": 1, "video_id": "imported", "start_time": 20, "end_time": 21,
            "summary": "소파가 넘어졌어요", "confidence": 0.8,
            "source_frames": [{"s3_key": "frames/imported/petcam_20261105_073500_000_frame_20.00.jpg",
                               "timestamp": 20}],
        }])
        self.assertIn("영상 20초쯤에", result["answer"])
        self.assertNotIn("7시", result["answer"])

    @patch.object(recording_service, "create_presigned_url", return_value=None)
    @patch.object(recording_service, "get_chunk_processing_statuses", return_value=[])
    @patch.object(recording_service, "get_scene_records", return_value=[])
    @patch.object(recording_service, "list_objects")
    def test_recording_list_uses_offset_unless_camera_time_is_explicit(self, list_objects, *_):
        updated = datetime(2026, 11, 5, 7, 35, tzinfo=timezone.utc)
        list_objects.return_value = [
            {"Key": "chunks/smartbench_0608/smartbench_0608_000.mp4", "LastModified": updated},
            {"Key": "chunks/camera/petcam_20261105_073500_000.mp4", "LastModified": updated,
             "Metadata": {"source_type": "petcam", "captured_at": "2026-11-05T07:35:00+09:00"}},
        ]
        chunks = recording_service.list_s3_recording_chunks()["chunks"]
        imported = next(chunk for chunk in chunks if chunk["video_id"] == "smartbench_0608")
        camera = next(chunk for chunk in chunks if chunk["video_id"] == "camera")
        self.assertEqual((imported["time_kind"], imported["time_label"], imported["recorded_at"], imported["period"]),
                         ("offset", "00:00", None, None))
        self.assertEqual((camera["time_kind"], camera["time_label"], camera["recording_date"]),
                         ("capture", "07:35:00", "2026-11-05"))

    @patch.object(gcs_uploader, "_get_bucket")
    def test_upload_marks_imported_video_and_requires_timezone_for_camera(self, get_bucket):
        blob = get_bucket.return_value.blob.return_value
        gcs_uploader.upload_video("demo.mp4", "chunks/demo/demo_000.mp4")
        self.assertEqual(blob.metadata, {"source_type": "imported"})
        with self.assertRaises(ValueError):
            gcs_uploader.upload_video("camera.mp4", "chunks/camera/camera_000.mp4", captured_at="2026-11-05T07:35:00")
        gcs_uploader.upload_video("camera.mp4", "chunks/camera/camera_000.mp4", captured_at="2026-11-05T07:35:00+09:00")
        self.assertEqual(blob.metadata["source_type"], "petcam")
        self.assertEqual(blob.metadata["captured_at"], "2026-11-05T07:35:00+09:00")


if __name__ == "__main__":
    unittest.main()
