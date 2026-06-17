from db.metadata import get_top_behavior_events
from pipeline.question_generator import generate_questions_from_behavior_events


DEFAULT_BEHAVIOR_QUESTIONS = [
    "가장 오래 이어진 행동은 무엇인가요?",
    "같은 행동이 반복된 구간이 있나요?",
    "특정 물체 근처에 오래 머문 장면이 있나요?",
    "움직임이 많았던 장면은 언제였나요?",
    "확인해볼 만한 행동이 있었나요?",
]


def suggest_queries(
    user_id: str | None = None,
    video_id: str | None = None,
    limit: int = 3,
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
    seen_signatures = set()

    for event in events:
        confidence = event.get("confidence")

        if confidence is None:
            confidence = 0.5

        if float(confidence) < 0.3:
            continue

        signature = _behavior_event_signature(event)
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        filtered_events.append(event)

        if len(filtered_events) >= limit:
            break

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


def _behavior_event_signature(event: dict) -> str:
    text = " ".join(
        str(event.get(key) or "")
        for key in ("action", "target_object", "summary")
    ).lower().replace(" ", "")

    keyword_groups = [
        ("feeding", ["먹이", "음식", "밥", "그릇", "사료", "먹는", "섭취", "bowl", "food"]),
        ("water", ["물", "마시", "water"]),
        ("exploring", ["탐색", "살피", "주변환경"]),
        ("movement", ["이동", "움직", "돌아다니", "걷"]),
        ("person", ["사람", "보호자", "person"]),
        ("vehicle", ["차량", "자동차", "차안", "차아래", "vehicle", "car"]),
    ]

    for group_name, keywords in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return group_name

    action = str(event.get("action") or "").strip()
    target = str(event.get("target_object") or "").strip()
    if action or target:
        return f"{action}:{target}"

    return str(event.get("id"))


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
