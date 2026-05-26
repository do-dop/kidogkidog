from db.metadata import get_global_top_queries, get_user_top_queries
from pipeline.question_generator import generate_questions_from_events
from pipeline.scene_event_extractor import extract_scene_events


DEFAULT_QUESTIONS = [
    "반려동물이 움직이거나 위치를 바꾼 장면이 있나요?",
    "영상 중간에 새롭게 등장한 객체가 있나요?",
    "반려동물이 특정 물체 근처에 머무른 장면이 있나요?",
]


def suggest_queries(
    user_id: str | None = None,
    video_id: str | None = None,
    limit: int = 6,
) -> list[str]:
    """
    추천 질문 생성.

    우선순위:
    1. 장면 후보 기반 LLM 추천 질문
    2. 사용자 빈출 검색어
    3. 전체 빈출 검색어
    4. 기본 질문
    """
    suggestions = []

    events = extract_scene_events(video_id=video_id, limit=8)

    generated_questions = generate_questions_from_events(
        events=events,
        limit=limit,
    )

    suggestions.extend(generated_questions)

    if user_id:
        user_queries = get_user_top_queries(user_id=user_id, limit=5)
        suggestions.extend(item["query"] for item in user_queries)

    global_queries = get_global_top_queries(limit=5)
    suggestions.extend(item["query"] for item in global_queries)

    suggestions.extend(DEFAULT_QUESTIONS)

    return _deduplicate_keep_order(suggestions)[:limit]


def get_suggestion_events(video_id: str | None = None, limit: int = 5) -> list[dict]:
    """
    UI에서 '왜 이런 질문이 나왔는지' 보여주기 위한 장면 후보 반환.
    """
    return extract_scene_events(video_id=video_id, limit=limit)


def _deduplicate_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        normalized = item.strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result