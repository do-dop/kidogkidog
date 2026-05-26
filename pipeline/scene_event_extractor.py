import json
from collections import Counter, defaultdict
from typing import Any

from db.metadata import get_scene_records


# 너무 배경처럼 자주 잡히는 객체들
BACKGROUND_LABELS = {
    "bed",
    "couch",
    "chair",
    "tv",
    "dining table",
}

# 반려동물로 볼 수 있는 라벨
PET_LABELS = {
    "dog",
    "cat",
    "bird",
}

# 한 번만 감지된 객체는 오탐일 가능성이 있으므로 주요 이벤트로 쓰지 않음
MIN_LABEL_OCCURRENCES = 2

# 객체 조합도 최소 2회 이상 반복될 때만 추천 근거로 사용
MIN_PAIR_OCCURRENCES = 2

# 장면 변화 이벤트도 너무 자주 뜨지 않게 최소 변화 점수 설정
MIN_SCENE_CHANGE_SCORE = 0.65


def _safe_parse_labels(raw_labels: Any) -> list[str]:
    """
    scenes.object_labels에 저장된 값을 list[str]로 변환한다.

    지원 형태:
    - '["dog", "bed"]'
    - 'dog,bed'
    - ["dog", "bed"]
    - None
    """
    if not raw_labels:
        return []

    if isinstance(raw_labels, list):
        return [
            str(label).strip()
            for label in raw_labels
            if str(label).strip()
        ]

    try:
        parsed = json.loads(raw_labels)

        if isinstance(parsed, list):
            return [
                str(label).strip()
                for label in parsed
                if str(label).strip()
            ]

        if isinstance(parsed, str):
            parsed = parsed.strip()
            return [parsed] if parsed else []

    except Exception:
        pass

    return [
        label.strip()
        for label in str(raw_labels).split(",")
        if label.strip()
    ]


def _label_change_score(previous_labels: set[str], current_labels: set[str]) -> float:
    """
    이전 프레임과 현재 프레임의 라벨 구성이 얼마나 달라졌는지 계산한다.

    0.0에 가까움: 거의 안 바뀜
    1.0에 가까움: 거의 완전히 바뀜
    """
    if not previous_labels and not current_labels:
        return 0.0

    union = previous_labels | current_labels
    intersection = previous_labels & current_labels

    if not union:
        return 0.0

    return 1.0 - (len(intersection) / len(union))


def extract_scene_events(video_id: str | None = None, limit: int = 8) -> list[dict]:
    """
    scenes 테이블의 object_labels와 timestamp 흐름을 보고
    '눈에 띄는 장면 후보'를 추출한다.

    중요한 점:
    - 행동을 확정하지 않는다.
    - 단일 프레임 오탐을 최대한 줄인다.
    - 특정 객체/객체 조합이 반복 감지될 때 더 높은 우선순위를 준다.
    """
    scene_records = get_scene_records(video_id=video_id)

    if not scene_records:
        return []

    events = []

    label_counter = Counter()
    label_timestamps = defaultdict(list)

    pair_counter = Counter()
    pair_timestamps = defaultdict(list)

    scene_change_candidates = []

    previous_labels = set()

    # ─────────────────────────────────────────
    # 1차 순회: 라벨 빈도, 조합 빈도, 장면 변화 후보 수집
    # ─────────────────────────────────────────
    for scene in scene_records:
        labels = _safe_parse_labels(scene.get("object_labels"))
        label_set = set(labels)

        if not label_set:
            previous_labels = label_set
            continue

        timestamp = float(scene.get("start_time", 0.0))

        # 라벨별 등장 횟수/시간 기록
        for label in sorted(label_set):
            label_counter[label] += 1
            label_timestamps[label].append(timestamp)

        # 반려동물 + 주변 객체 조합 기록
        pets = label_set & PET_LABELS
        surrounding_objects = label_set - PET_LABELS

        for pet in sorted(pets):
            for obj in sorted(surrounding_objects):
                pair = tuple(sorted([pet, obj]))
                pair_counter[pair] += 1
                pair_timestamps[pair].append(timestamp)

        # 사람 + 반려동물 조합도 의미 있는 후보로 기록
        if "person" in label_set:
            for pet in sorted(pets):
                pair = tuple(sorted(["person", pet]))
                pair_counter[pair] += 1
                pair_timestamps[pair].append(timestamp)

        # 장면 구성 변화 후보 기록
        change_score = _label_change_score(previous_labels, label_set)

        if previous_labels and change_score >= MIN_SCENE_CHANGE_SCORE:
            scene_change_candidates.append({
                "event_type": "scene_label_changed",
                "timestamp": timestamp,
                "labels": sorted(label_set),
                "score": round(change_score, 3),
                "description": (
                    f"{timestamp:.1f}초에 감지 객체 구성이 크게 바뀜 "
                    f"이전={sorted(previous_labels)}, 현재={sorted(label_set)}"
                ),
            })

        previous_labels = label_set

    # ─────────────────────────────────────────
    # 2차 처리: 반복 감지된 단일 객체 이벤트 생성
    # ─────────────────────────────────────────
    for label, count in label_counter.most_common():
        if count < MIN_LABEL_OCCURRENCES:
            continue

        first_time = label_timestamps[label][0]

        event_type = "repeated_object_detected"
        score = min(0.65 + count * 0.03, 0.9)

        if label in PET_LABELS:
            event_type = "pet_repeatedly_detected"
            score = min(0.75 + count * 0.03, 0.95)

        elif label == "person":
            event_type = "person_repeatedly_detected"
            score = min(0.7 + count * 0.03, 0.9)

        elif label in BACKGROUND_LABELS:
            event_type = "background_object_repeatedly_detected"
            score = min(0.45 + count * 0.02, 0.65)

        events.append({
            "event_type": event_type,
            "timestamp": first_time,
            "labels": [label],
            "score": round(score, 3),
            "description": (
                f"{first_time:.1f}초 이후 '{label}' 객체가 "
                f"{count}회 반복 감지됨"
            ),
        })

    # ─────────────────────────────────────────
    # 3차 처리: 반복 감지된 객체 조합 이벤트 생성
    # ─────────────────────────────────────────
    for pair, count in pair_counter.most_common():
        if count < MIN_PAIR_OCCURRENCES:
            continue

        first_time = pair_timestamps[pair][0]
        labels = list(pair)

        # 반려동물 조합은 더 중요하게 본다
        has_pet = bool(set(labels) & PET_LABELS)
        has_person = "person" in labels

        if has_person and has_pet:
            event_type = "person_pet_pair_repeated"
            score = min(0.78 + count * 0.03, 0.95)
            description = (
                f"{first_time:.1f}초 이후 {labels} 조합이 "
                f"{count}회 반복 감지됨"
            )

        elif has_pet:
            event_type = "pet_object_pair_repeated"
            score = min(0.75 + count * 0.03, 0.95)
            description = (
                f"{first_time:.1f}초 이후 반려동물과 주변 객체 {labels} 조합이 "
                f"{count}회 반복 감지됨"
            )

        else:
            event_type = "object_pair_repeated"
            score = min(0.55 + count * 0.02, 0.75)
            description = (
                f"{first_time:.1f}초 이후 {labels} 조합이 "
                f"{count}회 반복 감지됨"
            )

        events.append({
            "event_type": event_type,
            "timestamp": first_time,
            "labels": labels,
            "score": round(score, 3),
            "description": description,
        })

    # ─────────────────────────────────────────
    # 4차 처리: 장면 변화 이벤트 추가
    # 단, 장면 변화는 오탐 영향을 받을 수 있으므로 최대 3개만 사용
    # ─────────────────────────────────────────
    scene_change_candidates = sorted(
        scene_change_candidates,
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )

    events.extend(scene_change_candidates[:3])

    # ─────────────────────────────────────────
    # 5차 처리: 점수 보정
    # ─────────────────────────────────────────
    for event in events:
        label_set = set(event.get("labels", []))

        # 배경 객체만 있는 이벤트는 추천 우선순위를 낮춘다
        if label_set and label_set.issubset(BACKGROUND_LABELS):
            event["score"] = min(float(event.get("score", 0.5)), 0.5)

        # 반려동물이 포함된 이벤트는 조금 더 중요하게 본다
        if label_set & PET_LABELS:
            event["score"] = min(float(event.get("score", 0.5)) + 0.05, 0.98)

    # ─────────────────────────────────────────
    # 6차 처리: 정렬 및 중복 제거
    # ─────────────────────────────────────────
    events = _deduplicate_events(events)

    events = sorted(
        events,
        key=lambda item: (
            float(item.get("score", 0.0)),
            -(item.get("timestamp") or 0),
        ),
        reverse=True,
    )

    return events[:limit]


def summarize_scene_events(events: list[dict], max_items: int = 5) -> str:
    """
    UI 또는 prompt context에 넣을 장면 후보 요약 문자열 생성
    """
    if not events:
        return "눈에 띄는 장면 후보가 아직 없습니다."

    summaries = []

    for event in events[:max_items]:
        summaries.append(event["description"])

    return " / ".join(summaries)


def format_events_for_prompt(events: list[dict]) -> str:
    """
    LLM에게 넘기기 좋은 텍스트 형태로 장면 후보를 변환한다.
    """
    if not events:
        return "- 장면 후보 없음"

    lines = []

    for idx, event in enumerate(events, start=1):
        timestamp = event.get("timestamp")
        timestamp_text = f"{timestamp:.1f}초" if timestamp is not None else "여러 구간"
        labels = ", ".join(event.get("labels", [])) or "없음"

        lines.append(
            f"{idx}. 시간: {timestamp_text}\n"
            f"   유형: {event.get('event_type')}\n"
            f"   감지 객체: {labels}\n"
            f"   설명: {event.get('description')}"
        )

    return "\n".join(lines)


def _deduplicate_events(events: list[dict]) -> list[dict]:
    """
    같은 유형, 같은 시간대, 같은 라벨 조합의 이벤트 중복 제거
    """
    seen = set()
    result = []

    for event in events:
        timestamp = event.get("timestamp")

        if timestamp is not None:
            timestamp_key = round(float(timestamp), 1)
        else:
            timestamp_key = None

        key = (
            event.get("event_type"),
            timestamp_key,
            tuple(sorted(event.get("labels", []))),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(event)

    return result