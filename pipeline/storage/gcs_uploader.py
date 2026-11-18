"""Google Cloud Storage helpers for video chunks and extracted frames."""

import os
from pathlib import Path

from google.cloud import storage


BUCKET = os.getenv("GCS_BUCKET_NAME")


def _get_bucket():
    if not BUCKET:
        raise ValueError("GCS_BUCKET_NAME 환경변수가 설정되어 있지 않습니다.")
    return storage.Client().bucket(BUCKET)


def upload_file(local_path, object_key, extra_args=None):
    content_type = (extra_args or {}).get("ContentType")
    blob = _get_bucket().blob(object_key)
    metadata = (extra_args or {}).get("Metadata")
    if metadata:
        blob.metadata = metadata
    blob.upload_from_filename(local_path, content_type=content_type)
    print(f"GCS 업로드 성공: {object_key}")
    return object_key


def upload_video(local_path, object_key, *, captured_at=None):
    """Camera uploads must supply an ISO timestamp with a timezone.

    Imported/test videos deliberately carry no capture time, even when their
    generated filename resembles a camera recording.
    """
    metadata = {"source_type": "imported"}
    if captured_at is not None:
        from pipeline.storage.capture_time import trusted_capture_time
        metadata = {"source_type": "petcam", "captured_at": str(captured_at)}
        if trusted_capture_time(metadata) is None:
            raise ValueError("captured_at must be an ISO datetime with a timezone")
    return upload_file(local_path, object_key, extra_args={"ContentType": "video/mp4", "Metadata": metadata})


def upload_frame(local_path, video_id):
    frame_filename = os.path.basename(local_path)
    object_key = f"frames/{video_id}/{frame_filename}"
    return upload_file(
        local_path,
        object_key,
        extra_args={"ContentType": "image/jpeg"},
    )


def download_video(object_key, local_path):
    _get_bucket().blob(object_key).download_to_filename(local_path)
    print(f"GCS 다운로드 성공: {local_path}")


def download_bytes(object_key):
    return _get_bucket().blob(object_key).download_as_bytes()


def get_object_metadata(object_key):
    blob = _get_bucket().get_blob(object_key)
    if blob is None:
        raise FileNotFoundError(f"GCS 객체를 찾을 수 없습니다: {object_key}")
    return {
        "ContentLength": blob.size,
        "LastModified": blob.updated,
        "Metadata": blob.metadata or {},
    }


def download_range(object_key, start, end):
    return _get_bucket().blob(object_key).download_as_bytes(start=start, end=end)


def create_presigned_url(object_key, expires_in=3600):
    """Keep media private by streaming it through the API instead of signing URLs."""
    return None


def list_objects(prefix):
    return [
        {
            "Key": blob.name,
            "Size": blob.size,
            "LastModified": blob.updated,
            "Metadata": blob.metadata or {},
        }
        for blob in _get_bucket().list_blobs(prefix=prefix)
    ]


def object_exists(object_key):
    return _get_bucket().blob(object_key).exists()
