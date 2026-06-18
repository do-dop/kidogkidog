from db.scenes import get_video_top_object_labels
from db.search_history import (
    get_global_top_queries,
    get_user_top_queries,
)
from pipeline.scene_event_extractor import extract_scene_events, summarize_scene_events


def build_prompt_context(user_id: str | None = None, video_id: str | None = None) -> dict:
    """
    사용자 검색 패턴 + 영상 객체 라벨 + 장면 후보를 하나의 context로 묶는다.

    2단계에서는 추천 질문 UI에 사용하고,
    3단계 RAG에서는 프롬프트 컨텍스트로 사용할 수 있다.
    """
    user_queries = []

    if user_id:
        user_queries = get_user_top_queries(user_id=user_id, limit=5)

    global_queries = get_global_top_queries(limit=5)
    dominant_objects = get_video_top_object_labels(video_id=video_id, limit=10)
    scene_events = extract_scene_events(video_id=video_id, limit=8)

    return {
        "user_id": user_id,
        "video_id": video_id,
        "user_frequent_queries": user_queries,
        "global_frequent_queries": global_queries,
        "dominant_objects": dominant_objects,
        "scene_events": scene_events,
        "scene_event_summary": summarize_scene_events(scene_events),
    }


def build_prompt_hint(user_id: str | None = None, video_id: str | None = None) -> str:
    """
    3단계 RAG 프롬프트에 넣기 좋은 문자열 힌트.
    """
    context = build_prompt_context(user_id=user_id, video_id=video_id)

    user_query_text = ", ".join(
        item["query"] for item in context["user_frequent_queries"]
    ) or "없음"

    global_query_text = ", ".join(
        item["query"] for item in context["global_frequent_queries"]
    ) or "없음"

    object_text = ", ".join(
        f"{item['label']}({item['count']})"
        for item in context["dominant_objects"]
    ) or "없음"

    scene_event_text = context["scene_event_summary"] or "없음"

    return (
        f"사용자 빈출 검색어: {user_query_text}\n"
        f"전체 빈출 검색어: {global_query_text}\n"
        f"영상 내 주요 객체: {object_text}\n"
        f"눈에 띄는 장면 후보: {scene_event_text}"
    )
