from db.metadata import get_top_behavior_events
from pipeline.question_generator import generate_questions_from_behavior_events


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

    현재 방향:
    - 객체/장면 후보 기반 추천 질문은 사용하지 않는다.
    - 사용자 빈출 검색어도 섞지 않는다.
    - 현재 영상의 behavior_events 기반 질문만 만든다.
    """
    behavior_events = get_top_behavior_events(
        video_id=video_id,
        limit=8,
    )

    if behavior_events:
        generated_questions = generate_questions_from_behavior_events(
            events=behavior_events,
            limit=limit,
        )

        if generated_questions:
            return _deduplicate_keep_order(generated_questions)[:limit]

    return DEFAULT_BEHAVIOR_QUESTIONS[:limit]


def get_suggestion_behavior_events(
    video_id: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    UI에서 '오늘 발견한 주요 행동'을 보여주기 위한 행동 이벤트 반환.
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
    예전 scene_event 기반 추천 근거 함수.

    현재는 객체/장면 후보 기반 추천을 사용하지 않으므로 빈 리스트를 반환한다.
    app.py에서 이 함수를 import하고 있을 수 있어서 함수 이름만 유지한다.
    """
    return []


def has_behavior_events(video_id: str | None = None) -> bool:
    """
    특정 영상에 행동 이벤트가 존재하는지 확인한다.
    """
    behavior_events = get_top_behavior_events(
        video_id=video_id,
        limit=1,
    )

    return bool(behavior_events)


def _deduplicate_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        normalized = str(item).strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result