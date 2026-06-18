import time

from fastapi import APIRouter, HTTPException

from api.schemas import QueryRequest
from db.search_history import insert_search_log, upsert_user_frequent_query


router = APIRouter(tags=["search"])


@router.post("/query")
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="query가 필요합니다."
        )

    top_k = max(1, min(request.top_k, 10))
    started_at = time.perf_counter()

    try:
        from pipeline.rag_chain import run_rag_query

        rag_result = run_rag_query(
            query=request.query,
            video_id=request.video_id,
            top_k=top_k,
            user_id=request.user_id,
            recording_date=request.recording_date,
            time_range=request.time_range.dict() if request.time_range else None,
            source_event_id=request.source_event_id,
            event_start=request.event_start,
            event_end=request.event_end,
        )
    finally:
        latency_ms = int((time.perf_counter() - started_at) * 1000)

    if request.user_id:
        result_count = len(rag_result.get("results", []))
        try:
            insert_search_log(
                user_id=request.user_id,
                query_raw=request.query,
                video_id=request.video_id,
                top_k=top_k,
                result_count=result_count,
                latency_ms=latency_ms,
            )
            upsert_user_frequent_query(
                user_id=request.user_id,
                query_raw=request.query,
            )
        except Exception as exc:
            print(f"검색 로그 저장 실패: {exc}", flush=True)

    return {
        "status": "ok",
        "query": request.query,
        "video_id": request.video_id,
        "recording_date": request.recording_date,
        "top_k": top_k,
        "answer": rag_result["answer"],
        "results": rag_result["results"],
        "evidence_items": rag_result.get("evidence_items", rag_result["results"]),
        "behavior_events": rag_result.get("behavior_events", []),
        "used_llm": rag_result["used_llm"],
    }
