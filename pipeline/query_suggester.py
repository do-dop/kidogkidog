from db.metadata import (
    get_global_top_queries,
    get_user_top_queries,
    get_top_behavior_events,
)
from pipeline.question_generator import generate_questions_from_behavior_events
from pipeline.scene_event_extractor import extract_scene_events


DEFAULT_QUESTIONS = [
    "반려동물이 움직이거나 위치를 바꾼 장면이 있나요?",
    "영상 중간에 새롭게 등장한 객체가 있나요?",
    "반려동물이 특정 물체 근처에 머무른 장면이 있나요?",
]


DEFAULT_BEHAVIOR_QUESTIONS = [
    "반려동물이 가장 오래 집중한 행동은 무엇인가요?",
    "반려동물이 같은 행동을 반복한 구간이 있나요?",
    "반려동물이 특정 물체에 관심을 보인 장면이 있나요?",
    "평소와 달라 보이는 행동이 있었나요?",
    "보호자가 확인해볼 만한 특이 행동이 있나요?",
]


def suggest_queries(
    user_id: str | None = None,
    video_id: str | None = None,
    limit: int = 6,
) -> list[str]:
    """
    추천 질문 생성.

    개선 후 우선순위:
    1. 행동 이벤트가 있으면 현재 영상의 행동 이벤트 기반 질문만 사용
       - 사용자 빈출 검색어 / 전체 빈출 검색어를 섞지 않는다.
       - 그래야 고양이 영상에 강아지 질문이 섞이지 않는다.

    2. 행동 이벤트가 없을 때만 기존 장면 후보 기반 질문 사용

    3. 장면 후보도 없을 때만 사용자 빈출 / 전체 빈출 / 기본 질문 사용
    """
    behavior_events = get_top_behavior_events(
        video_id=video_id,
        limit=8,
    )

    # 1. 행동 이벤트가 있으면 행동 기반 추천질문만 사용
    if behavior_events:
        generated_questions = generate_questions_from_behavior_events(
            events=behavior_events,
            limit=limit,
        )

        generated_questions = _filter_questions_by_event_subject(
            questions=generated_questions,
            behavior_events=behavior_events,
        )

        if generated_questions:
            return _deduplicate_keep_order(generated_questions)[:limit]

        # 행동 이벤트는 있는데 질문 생성이 실패한 경우에만 generic 행동 질문 사용
        return DEFAULT_BEHAVIOR_QUESTIONS[:limit]

    # 2. 행동 이벤트가 없으면 기존 객체/장면 후보 기반 추천 질문으로 fallback
    scene_events = extract_scene_events(
        video_id=video_id,
        limit=8,
    )

    if scene_events:
        generated_questions = generate_questions_from_events(
            events=scene_events,
            limit=limit,
        )

        if generated_questions:
            return _deduplicate_keep_order(generated_questions)[:limit]

    # 3. 그래도 없을 때만 사용자 빈출 검색어 사용
    suggestions = []

    if user_id:
        user_queries = get_user_top_queries(
            user_id=user_id,
            limit=5,
        )

        suggestions.extend(item["query"] for item in user_queries)

    # 4. 전체 빈출 검색어
    global_queries = get_global_top_queries(limit=5)
    suggestions.extend(item["query"] for item in global_queries)

    # 5. 기본 질문
    suggestions.extend(DEFAULT_QUESTIONS)

    return _deduplicate_keep_order(suggestions)[:limit]


def get_suggestion_behavior_events(
    video_id: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    UI에서 '오늘 발견한 주요 행동'을 보여주기 위한 행동 이벤트 반환.

    신뢰도가 너무 낮은 이벤트는 추천 근거로 사용하지 않는다.
    """
    events = get_top_behavior_events(
        video_id=video_id,
        limit=limit * 3,
    )

    filtered_events = []

    for event in events:
        confidence = event.get("confidence")

        if confidence is None:
            confidence = 0.5

        if float(confidence) < 0.3:
            continue

        filtered_events.append(event)

    return filtered_events[:limit]


def get_suggestion_events(
    video_id: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    UI에서 '왜 이런 질문이 나왔는지' 보여주기 위한 장면 후보 반환.

    행동 이벤트가 없는 경우 fallback 설명용으로 사용한다.
    """
    return extract_scene_events(
        video_id=video_id,
        limit=limit,
    )


def has_behavior_events(video_id: str | None = None) -> bool:
    """
    특정 영상에 행동 이벤트가 존재하는지 확인한다.
    """
    behavior_events = get_top_behavior_events(
        video_id=video_id,
        limit=1,
    )

    return bool(behavior_events)


def _filter_questions_by_event_subject(
    questions: list[str],
    behavior_events: list[dict],
) -> list[str]:
    """
    현재 영상의 행동 이벤트 주체와 맞지 않는 질문을 제거한다.

    예:
    - behavior_events에 cat만 있으면 '강아지' 질문 제거
    - behavior_events에 dog만 있으면 '고양이' 질문 제거
    """
    subjects = {
        str(event.get("subject") or "").lower()
        for event in behavior_events
        if event.get("subject")
    }

    has_cat = "cat" in subjects
    has_dog = "dog" in subjects

    filtered = []

    for question in questions:
        q = question.strip()

        if not q:
            continue

        # 고양이 영상인데 강아지 질문이면 제거
        if has_cat and not has_dog:
            if "강아지" in q or "dog" in q.lower():
                continue

        # 강아지 영상인데 고양이 질문이면 제거
        if has_dog and not has_cat:
            if "고양이" in q or "cat" in q.lower():
                continue

        filtered.append(q)

    return filtered


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