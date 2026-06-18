from fastapi import APIRouter, HTTPException

from api.schemas import FrequentQueryDeleteRequest, SuggestionRequest
from db.search_history import (
    delete_user_frequent_query,
    get_user_recent_queries,
    get_user_top_queries,
)
from pipeline.query_suggester import get_suggestion_behavior_events, suggest_query_items
from pipeline.vector_store import get_indexed_frames


router = APIRouter(tags=["suggestions"])


@router.get("/suggestions")
def get_suggestions(
    user_id: str | None = None,
    video_id: str | None = None,
    limit: int = 5,
):
    safe_limit = max(1, min(limit, 10))

    try:
        question_items = suggest_query_items(
            user_id=user_id,
            video_id=video_id,
            limit=safe_limit,
        )
        questions = [item["question"] for item in question_items]
        behavior_events = get_suggestion_behavior_events(
            video_id=video_id,
            limit=5,
        )
        indexed_frame_count = len(get_indexed_frames(video_id=video_id))
        top_queries = get_user_top_queries(user_id=user_id, limit=5) if user_id else []
        recent_queries = get_user_recent_queries(user_id=user_id, limit=5) if user_id else []
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"추천 질문을 생성하지 못했습니다: {exc}",
        ) from exc

    return {
        "status": "ok",
        "video_id": video_id,
        "questions": questions,
        "question_sources": question_items,
        "behavior_events": behavior_events,
        "indexed_frame_count": indexed_frame_count,
        "top_queries": top_queries,
        "recent_queries": recent_queries,
    }


@router.post("/suggestions")
def post_suggestions(request: SuggestionRequest):
    return get_suggestions(
        user_id=request.user_id,
        video_id=request.video_id,
        limit=request.limit,
    )


@router.delete("/users/{user_id}/frequent-queries")
def delete_frequent_query(user_id: str, request: FrequentQueryDeleteRequest):
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id가 필요합니다.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query가 필요합니다.")

    try:
        deleted_count = delete_user_frequent_query(
            user_id=user_id,
            query_raw=request.query,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"자주 찾는 검색어를 삭제하지 못했습니다: {exc}",
        ) from exc

    return {
        "status": "ok",
        "deleted_count": deleted_count,
        "query": request.query,
    }
