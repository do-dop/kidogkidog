from datetime import timezone
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from pipeline.chunking import CHUNK_SECONDS
from db.scenes import get_scene_records
from db.processed_chunks import get_chunk_processing_statuses
from pipeline.storage.gcs_uploader import create_presigned_url, list_objects
from pipeline.storage.capture_time import trusted_capture_time

from api.services.media_service import media_path_for_key, thumbnail_path_for_key


def get_chunk_index(s3_key):
    match = re.search(r"_(\d{3})\.mp4$", Path(s3_key).name)
    if not match:
        return 0

    return int(match.group(1))


def get_chunk_recorded_at(s3_key, last_modified=None, metadata=None):
    # Object modification time and generated petcam_ filenames are not evidence
    # of when an imported source video was filmed.
    return trusted_capture_time(metadata)


def format_seconds(seconds):
    minutes, remainder = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{remainder:02d}"


def get_chunk_period_from_hour(hour):
    if hour < 12:
        return "오전"
    if hour < 17:
        return "오후"
    if hour < 21:
        return "저녁"
    return "야간"


def format_timestamp(dt):
    if not dt:
        return None

    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc)

    return dt.isoformat()


def find_thumbnail_key(video_id, chunk_key):
    chunk_stem = Path(chunk_key).stem

    try:
        scenes = get_scene_records(video_id=video_id)
    except Exception:
        return None

    for scene in scenes:
        s3_key = scene.get("s3_key")

        if s3_key and f"{chunk_stem}_frame_" in s3_key:
            return s3_key

    return None


def list_s3_recording_chunks(video_id: str | None = None):
    prefix = f"chunks/{video_id}/" if video_id else "chunks/"
    objects = list_objects(prefix)
    status_by_chunk = {
        item["chunk_path"]: item
        for item in get_chunk_processing_statuses(video_id=video_id)
    }
    chunks = []

    for obj in objects:
        key = obj["Key"]

        if not key.lower().endswith(".mp4"):
            continue

        parts = Path(key).parts
        chunk_video_id = parts[1] if len(parts) >= 2 and parts[0] == "chunks" else "unknown"
        index = get_chunk_index(key)
        start_seconds = index * CHUNK_SECONDS
        recorded_at = get_chunk_recorded_at(key, obj.get("LastModified"), obj.get("Metadata"))
        if recorded_at:
            recorded_at = recorded_at.astimezone(ZoneInfo("Asia/Seoul"))
        thumbnail_key = find_thumbnail_key(chunk_video_id, key)

        chunks.append({
            "id": key,
            "s3_key": key,
            "video_id": chunk_video_id,
            "filename": Path(key).name,
            "chunk_index": index,
            "start_seconds": start_seconds,
            "duration_seconds": 60,
            "recorded_at": recorded_at.isoformat() if recorded_at else None,
            "recording_date": recorded_at.date().isoformat() if recorded_at else None,
            "time_label": recorded_at.strftime("%H:%M:%S") if recorded_at else format_seconds(start_seconds),
            "period": get_chunk_period_from_hour(recorded_at.hour) if recorded_at else None,
            "time_kind": "capture" if recorded_at else "offset",
            "size": obj.get("Size", 0),
            "last_modified": format_timestamp(obj.get("LastModified")),
            "url": create_presigned_url(key, expires_in=3600),
            "media_path": media_path_for_key(key),
            "thumbnail_s3_key": thumbnail_key,
            "thumbnail_url": create_presigned_url(thumbnail_key, expires_in=3600) if thumbnail_key else None,
            "thumbnail_media_path": media_path_for_key(thumbnail_key) if thumbnail_key else thumbnail_path_for_key(key),
            "processing": status_by_chunk.get(key, {"state": "pending"}),
        })

    chunks.sort(key=lambda item: (item["video_id"], item["chunk_index"], item["filename"]))

    return {
        "status": "ok",
        "prefix": prefix,
        "count": len(chunks),
        "chunks": chunks,
    }
