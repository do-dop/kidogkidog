from db.behavior_events import get_top_behavior_events
from pipeline.question_generator import generate_questions_from_behavior_events
from pipeline.vector_store import get_indexed_frames

def _event_text(event: dict) -> str:
    """행동 이벤트에서 분류에 쓸 텍스트를 모은다."""
    return " ".join(
        str(event.get(key, "") or "")
        for key in [
            "subject",
            "action",
            "target_object",
            "summary",
            "evidence",
        ]
    ).lower()


def _event_category(event: dict) -> str:
    """비슷한 행동 이벤트를 하나의 카테고리로 묶는다."""
    text = _event_text(event)

    vehicle_words = ["차", "차량", "자동차", "바퀴", "vehicle", "car", "wheel"]
    under_words = ["아래", "밑", "숨", "들어", "under", "hide", "enter"]
    food_words = ["먹", "음식", "먹이", "사료", "밥", "그릇", "bowl", "food"]
    water_words = ["물", "마시", "water"]
    person_words = ["사람", "보호자", "손", "person", "human", "hand"]

    if any(word in text for word in vehicle_words) and any(word in text for word in under_words):
        return "vehicle_under"

    if any(word in text for word in vehicle_words):
        return "vehicle"

    if any(word in text for word in food_words):
        return "food"

    if any(word in text for word in water_words):
        return "water"

    if any(word in text for word in person_words):
        return "person"

    if any(word in text for word in under_words):
        return "hide_or_enter"

    return "other"


def _event_score(event: dict) -> float:
    try:
        return float(event.get("interestingness") or 0)
    except Exception:
        return 0.0


def _dedupe_events_by_category(events: list[dict], limit: int) -> list[dict]:
    """
    같은 종류 이벤트는 1개만 남긴다.
    음식 이벤트가 많이 잡혀도 추천 후보를 독점하지 않게 한다.
    """
    best_by_category: dict[str, dict] = {}

    for event in events:
        category = _event_category(event)

        if category not in best_by_category:
            best_by_category[category] = event
            continue

        if _event_score(event) > _event_score(best_by_category[category]):
            best_by_category[category] = event

    priority = [
        "vehicle_under",
        "vehicle",
        "hide_or_enter",
        "person",
        "water",
        "food",
        "other",
    ]

    result = []
    for category in priority:
        if category in best_by_category:
            result.append(best_by_category[category])

    return result[:limit]


def _question_category(question: str) -> str:
    text = question.lower()

    if any(word in text for word in ["차", "차량", "자동차", "바퀴", "밑", "아래"]):
        return "vehicle"
    if any(word in text for word in ["먹", "음식", "먹이", "사료", "밥", "그릇"]):
        return "food"
    if any(word in text for word in ["물", "마시"]):
        return "water"
    if any(word in text for word in ["사람", "보호자", "손"]):
        return "person"
    if any(word in text for word in ["숨", "들어", "머문"]):
        return "hide_or_enter"

    return "other"


def _dedupe_questions_by_category(questions: list[str], limit: int) -> list[str]:
    """
    같은 종류 질문은 1개만 남긴다.
    예: 음식 질문 3개 → 음식 질문 1개
    """
    result = []
    seen_categories = set()

    for question in questions:
        category = _question_category(question)

        if category in seen_categories:
            continue

        result.append(question)
        seen_categories.add(category)

        if len(result) >= limit:
            break

    return result


def _is_low_value_question(question: str) -> bool:
    """
    검색 결과로 이어지기 어려운 너무 넓은 추천 질문을 제외한다.
    예: 주변 물체 근처, 오래 이어진 행동처럼 구체적인 장면이 없는 질문.
    """
    text = str(question or "").strip()

    if not text:
        return True

    weak_phrases = [
        "주변 물체",
        "주변 객체",
        "주변 환경",
        "오래 이어진 행동",
        "행동이 있는 구간",
        "행동 후보",
    ]

    if any(phrase in text for phrase in weak_phrases):
        return True

    return False


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
    return [
        item["question"]
        for item in suggest_query_items(
            user_id=user_id,
            video_id=video_id,
            limit=limit,
        )
    ]


def suggest_query_items(
    user_id: str | None = None,
    video_id: str | None = None,
    limit: int = 3,
) -> list[dict]:
    """
    추천 질문과 해당 질문의 근거 behavior event metadata를 함께 반환한다.
    """
    behavior_events = get_top_behavior_events(
        video_id=video_id,
        limit=max(limit * 8, 30),
    )

    if behavior_events:
        generated_questions = generate_questions_from_behavior_events(
            events=behavior_events,
            limit=max(limit * 3, 9),
        )

        generated_questions = [
            question
            for question in _deduplicate_keep_order(generated_questions)
            if not _is_low_value_question(question)
        ]

        generated_questions = _dedupe_questions_by_category(
            generated_questions,
            limit=limit,
        )

        if generated_questions:
            return [
                _build_question_item(
                    question=question,
                    event=_best_event_for_question(question, behavior_events),
                )
                for question in generated_questions
            ]

    indexed_questions = _generate_questions_from_indexed_frames(
        video_id=video_id,
        limit=limit,
    )

    if indexed_questions:
        return [
            {
                "question": question,
                "source_event_id": None,
                "event_start": None,
                "event_end": None,
            }
            for question in indexed_questions
        ]

    return []


def get_suggestion_behavior_events(
    video_id: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    UI에서 '오늘 발견한 주요 행동'을 보여주기 위한 행동 이벤트 반환.
    같은 종류의 행동은 1개만 남겨서 음식 질문이 여러 개 반복되지 않게 한다.
    """
    events = get_top_behavior_events(
        video_id=video_id,
        limit=max(limit * 8, 30),
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

    # 여기서 카테고리별로 1개만 남김
    # 예: 음식 이벤트가 여러 개 있어도 food 1개만 남음
    deduped_events = _dedupe_events_by_category(
        filtered_events,
        limit=limit,
    )

    return deduped_events[:limit]


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


def _build_question_item(question: str, event: dict | None) -> dict:
    if not event:
        return {
            "question": question,
            "source_event_id": None,
            "event_start": None,
            "event_end": None,
        }

    return {
        "question": question,
        "source_event_id": event.get("id"),
        "event_start": event.get("start_time"),
        "event_end": event.get("end_time"),
    }


def _best_event_for_question(question: str, events: list[dict]) -> dict | None:
    if not events:
        return None

    return max(
        events,
        key=lambda event: (
            _question_event_score(question, event),
            float(event.get("interestingness") or 0.0),
            float(event.get("confidence") or 0.0),
        ),
    )


def _question_event_score(question: str, event: dict) -> float:
    question_text = str(question or "").lower().replace(" ", "")
    event_text = " ".join(
        str(event.get(key) or "")
        for key in ("subject", "action", "target_object", "summary")
    ).lower().replace(" ", "")

    evidence = event.get("evidence") or []
    if isinstance(evidence, list):
        event_text += "".join(str(item).lower().replace(" ", "") for item in evidence)

    score = 0.0
    keywords = [
        "먹", "밥", "사료", "물", "마시", "그릇", "bowl", "food", "water",
        "사람", "손", "person", "hand",
        "움직", "걷", "이동", "탐색",
        "만지", "건드", "비비", "긁",
        "가방", "공", "장난감", "소파", "침대",
    ]

    for keyword in keywords:
        if keyword in question_text and keyword in event_text:
            score += 1.0

    for token in question.replace("?", " ").replace(",", " ").split():
        normalized = token.strip().lower().replace(" ", "")
        if len(normalized) >= 2 and normalized in event_text:
            score += 0.2

    return score


def _generate_questions_from_indexed_frames(
    video_id: str | None = None,
    limit: int = 3,
) -> list[str]:
    """
    behavior_events가 아직 없는 영상도 빈 추천 영역으로 남지 않도록
    인덱싱된 프레임의 객체 라벨을 기반으로 검색 가능한 질문을 만든다.
    """
    try:
        frames = get_indexed_frames(video_id=video_id)
    except Exception as exc:
        print(f"프레임 기반 추천 질문 생성 실패: {exc}", flush=True)
        return []

    label_counts: dict[str, int] = {}

    for frame in frames:
        labels = _parse_object_labels(frame.get("object_labels"))
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    sorted_labels = [
        label
        for label, _count in sorted(
            label_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    questions = []

    for label in sorted_labels:
        label_text = _label_to_question_text(label)
        if not label_text:
            continue

        questions.extend([
            f"{label_text} 근처에 머문 장면 보여줘",
            f"{label_text}을 만지거나 살펴보는 장면 있어?",
            f"{label_text} 주변에서 움직인 장면 찾아줘",
        ])

        if len(questions) >= limit:
            break

    if len(questions) < limit and frames:
        questions.extend([
            "움직임이 잘 보이는 장면 찾아줘",
            "반려동물이 머무른 장면 보여줘",
            "주변 물체를 살펴보는 장면 있어?",
        ])

    questions = [
        question
        for question in _deduplicate_keep_order(questions)
        if not _is_low_value_question(question)
    ]

    return questions[:limit]


def _parse_object_labels(value) -> list[str]:
    if not value:
        return []

    if isinstance(value, list):
        raw_labels = value
    else:
        raw_labels = str(value).split(",")

    labels = []
    for label in raw_labels:
        normalized = str(label).strip().lower()
        if normalized:
            labels.append(normalized)

    return labels


def _label_to_question_text(label: str) -> str | None:
    label_map = {
        "dog": "강아지",
        "cat": "고양이",
        "person": "사람",
        "bowl": "그릇",
        "cup": "컵",
        "bottle": "물병",
        "sports ball": "공",
        "ball": "공",
        "backpack": "가방",
        "handbag": "가방",
        "suitcase": "가방",
        "chair": "의자",
        "couch": "소파",
        "bed": "침대",
        "dining table": "테이블",
        "toy": "장난감",
    }

    return label_map.get(label, label if len(label) <= 12 else None)
