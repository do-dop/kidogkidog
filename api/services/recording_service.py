from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

from db.scenes import get_scene_records
from pipeline.s3_uploader import create_presigned_url, list_objects

from api.services.media_service import media_path_for_key, thumbnail_path_for_key


def get_chunk_index(s3_key):
    match = re.search(r"_(\d{3})\.mp4$", Path(s3_key).name)
    if not match:
        return 0

    return int(match.group(1))


def get_chunk_recorded_at(s3_key, last_modified=None):
    filename = Path(s3_key).name
    match = re.search(r"petcam_(\d{8})_(\d{6})_(\d{3})\.mp4$", filename)

    if match:
        base = datetime.strptime(
            f"{match.group(1)}{match.group(2)}",
            "%Y%m%d%H%M%S",
        )
        return base + timedelta(seconds=int(match.group(3)) * 60)

    if last_modified:
        if last_modified.tzinfo:
            return last_modified.astimezone(timezone.utc).replace(tzinfo=None)
        return last_modified

    return None


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
    chunks = []

    for obj in objects:
        key = obj["Key"]

        if not key.lower().endswith(".mp4"):
            continue

        parts = Path(key).parts
        chunk_video_id = parts[1] if len(parts) >= 2 and parts[0] == "chunks" else "unknown"
        index = get_chunk_index(key)
        start_seconds = index * 60
        recorded_at = get_chunk_recorded_at(key, obj.get("LastModified"))
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
            "time_label": recorded_at.strftime("%H:%M") if recorded_at else None,
            "period": get_chunk_period_from_hour(recorded_at.hour) if recorded_at else "오전",
            "size": obj.get("Size", 0),
            "last_modified": format_timestamp(obj.get("LastModified")),
            "url": create_presigned_url(key, expires_in=3600),
            "media_path": media_path_for_key(key),
            "thumbnail_s3_key": thumbnail_key,
            "thumbnail_url": create_presigned_url(thumbnail_key, expires_in=3600) if thumbnail_key else None,
            "thumbnail_media_path": media_path_for_key(thumbnail_key) if thumbnail_key else thumbnail_path_for_key(key),
        })

    chunks.sort(key=lambda item: (item["video_id"], item["chunk_index"], item["filename"]))

    return {
        "status": "ok",
        "prefix": prefix,
        "count": len(chunks),
        "chunks": chunks,
    }
