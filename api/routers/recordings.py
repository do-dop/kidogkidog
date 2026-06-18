from fastapi import APIRouter, HTTPException

from api.services.recording_service import list_s3_recording_chunks
from pipeline.query_suggester import get_suggestion_behavior_events


router = APIRouter(prefix="/recordings", tags=["recordings"])


@router.get("/chunks")
def list_recording_chunks(video_id: str | None = None):
    try:
        return list_s3_recording_chunks(video_id=video_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"S3 청크 목록을 불러오지 못했습니다: {exc}",
        ) from exc


@router.get("/events")
def list_recording_events(video_id: str | None = None, limit: int = 50):
    safe_limit = max(1, min(limit, 5))

    try:
        events = get_suggestion_behavior_events(video_id=video_id, limit=safe_limit)
    except Exception as exc:
        if "no such table: behavior_events" in str(exc):
            return {
                "status": "ok",
                "count": 0,
                "events": [],
            }

        raise HTTPException(
            status_code=500,
            detail=f"행동 이벤트 메타데이터를 불러오지 못했습니다: {exc}",
        ) from exc

    return {
        "status": "ok",
        "count": len(events),
        "events": events,
    }
