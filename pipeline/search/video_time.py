"""Display chunk-local evidence at its position in the complete recording."""

import re
from datetime import timedelta
from zoneinfo import ZoneInfo

from pipeline.chunking import CHUNK_SECONDS
from pipeline.storage.capture_time import trusted_capture_time


_CHUNK_FRAME = re.compile(r"_(\d{3})_frame_[0-9.]+(?:\.jpg|\.jpeg)?$")
_CHUNK_VIDEO = re.compile(r"_(\d{3})\.mp4$")
_SEOUL = ZoneInfo("Asia/Seoul")


def frame_video_seconds(frame: dict, local_seconds: float | None = None) -> float | None:
    """Return an imported frame's video-wide offset, preserving local playback metadata."""
    if local_seconds is None:
        local_seconds = frame.get("timestamp")
    if local_seconds is None:
        return None
    source = str(frame.get("s3_key") or frame.get("frame_id") or frame.get("frame_path") or "")
    match = _CHUNK_FRAME.search(source) or _CHUNK_VIDEO.search(source)
    return float(local_seconds) + (int(match.group(1)) * CHUNK_SECONDS if match else 0)


def _clock(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    minutes, remainder = divmod(whole, 60)
    return f"{minutes:02d}:{remainder:02d}"


def format_frame_time(frame: dict, local_seconds: float) -> str:
    """Prefer verified camera time; otherwise use a whole-video offset."""
    captured = trusted_capture_time({
        "source_type": frame.get("capture_time_source") or frame.get("source_type"),
        "captured_at": frame.get("recorded_at") or frame.get("captured_at"),
    })
    if captured is not None:
        frame_second = float(frame.get("timestamp") or 0)
        moment = captured + timedelta(seconds=float(local_seconds) - frame_second)
        return moment.astimezone(_SEOUL).strftime("%H:%M:%S")
    return _clock(frame_video_seconds(frame, local_seconds) or 0)


def format_event_time(event: dict, local_seconds: float) -> str:
    frames = event.get("source_frames") or []
    for frame in frames:
        if isinstance(frame, dict) and frame.get("s3_key"):
            return format_frame_time(frame, local_seconds)
    return _clock(float(local_seconds))
