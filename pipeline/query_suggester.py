from db.behavior_events import get_top_behavior_events
from pipeline.question_generator import generate_questions_from_behavior_events
from pipeline.vector_store import get_indexed_frames


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
        limit=8,
    )

    if behavior_events:
        generated_questions = generate_questions_from_behavior_events(
            events=behavior_events,
            limit=limit,
        )

        generated_questions = _deduplicate_keep_order(generated_questions)[:limit]

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

    return _deduplicate_keep_order(questions)[:limit]


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
