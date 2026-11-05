import os
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import QueryRequest
from db.search_history import insert_search_log, upsert_user_frequent_query
from pipeline.search.result_grouping import group_results_by_video


router = APIRouter(tags=["search"])


@router.post("/query")
def query(request: QueryRequest, background_tasks: BackgroundTasks):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="query가 필요합니다."
        )

    top_k = max(1, min(request.top_k, 10))
    started_at = time.perf_counter()
    route_decision = None
    expose_route = os.getenv("QUESTION_ROUTER_EXPOSE_RESULT", "").lower()

    try:
        from pipeline.search.question_router import get_router_mode, run_shadow_classification

        if get_router_mode() == "shadow":
            if expose_route in {"1", "true", "yes"}:
                route_decision = run_shadow_classification(request.query)
            else:
                background_tasks.add_task(run_shadow_classification, request.query)
    except Exception as exc:
        # 질문 라우터는 shadow 기능이므로 어떤 실패도 기존 검색을 막지 않는다.
        print(f"질문 라우터 shadow 실행 실패: {exc}", flush=True)

    try:
        from pipeline.search.rag_chain import run_rag_query

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

    # 순서 질문의 결과는 같은 영상 안의 단계 목록이다. 영상별 한 장으로 묶으면 단계가 사라진다.
    is_sequence = rag_result.get("query_type") == "sequence"
    raw_results = rag_result.get("results", [])
    grouped_results = raw_results if is_sequence else group_results_by_video(raw_results, top_k)
    raw_evidence = rag_result.get("evidence_items", raw_results)
    grouped_evidence = (
        grouped_results if raw_evidence is raw_results
        else raw_evidence if is_sequence
        else group_results_by_video(raw_evidence, top_k)
    )

    if request.user_id:
        result_count = len(grouped_results)
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

    response = {
        "status": "ok",
        "query": request.query,
        "video_id": request.video_id,
        "recording_date": request.recording_date,
        "top_k": top_k,
        "answer": rag_result["answer"],
        "results": grouped_results,
        "evidence_items": grouped_evidence,
        "behavior_events": rag_result.get("behavior_events", []),
        "used_llm": rag_result["used_llm"],
        "query_type": rag_result.get("query_type", "search"),
        "total_event_count": rag_result.get("total_event_count"),
        "excluded_event_count": rag_result.get("excluded_event_count", 0),
        "sequence_intent": rag_result.get("sequence_intent"),
        "truncated": rag_result.get("truncated", False),
    }

    if route_decision and expose_route in {"1", "true", "yes"}:
        response["question_route"] = route_decision.to_dict()

    if rag_result.get("aggregate_result") is not None:
        response["aggregate_result"] = rag_result["aggregate_result"]

    return response
