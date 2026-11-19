from fastapi import APIRouter, HTTPException

from api.services.recording_service import list_s3_recording_chunks
from pipeline.search.query_suggester import get_suggestion_behavior_events
from pipeline.storage.vector_store import get_indexed_frames


router = APIRouter(prefix="/recordings", tags=["recordings"])


@router.get("/chunks")
def list_recording_chunks(video_id: str | None = None):
    try:
        return list_s3_recording_chunks(video_id=video_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"GCS 청크 목록을 불러오지 못했습니다: {exc}",
        ) from exc


@router.get("/index-status")
def list_recording_index_status(video_id: str | None = None):
    """Report actual search readiness without running question generation."""
    try:
        chunks = list_s3_recording_chunks(video_id=video_id)["chunks"]
        by_video = {}
        for chunk in chunks:
            item = by_video.setdefault(chunk["video_id"], {"chunk_count": 0, "processing_states": {}})
            item["chunk_count"] += 1
            state = chunk.get("processing", {}).get("state", "pending")
            item["processing_states"][state] = item["processing_states"].get(state, 0) + 1

        videos = []
        for current_video_id, item in sorted(by_video.items()):
            frame_count = len(get_indexed_frames(video_id=current_video_id))
            videos.append({
                "video_id": current_video_id,
                **item,
                "indexed_frame_count": frame_count,
                "ready_for_search": frame_count > 0,
            })
        return {"status": "ok", "videos": videos}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"영상 검색 인덱스 상태를 확인하지 못했습니다: {exc}") from exc


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
