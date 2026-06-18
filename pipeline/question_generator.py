import json
import os
import re
from typing import Any

from pipeline.behavior_event_extractor import format_behavior_events_for_prompt


DEFAULT_GENERATED_QUESTIONS = [
    "반려동물이 특정 위치 근처에 머무른 장면이 있나요?",
    "영상 중간에 새롭게 등장한 객체나 사람이 있나요?",
    "반려동물이 움직이거나 위치를 바꾼 장면이 있나요?",
    "반려동물이 주변 물체에 관심을 보인 장면이 있나요?",
    "처음과 달라진 장면이 있었나요?",
]


DEFAULT_BEHAVIOR_QUESTIONS = [
    "반려동물이 가장 오래 집중한 행동은 무엇인가요?",
    "반려동물이 같은 행동을 반복한 구간이 있나요?",
    "반려동물이 특정 물체에 관심을 보인 장면이 있나요?",
    "움직임이 많았던 장면은 언제였나요?",
    "보호자가 확인해볼 만한 장면이 있나요?",
]

def generate_questions_from_behavior_events(
    events: list[dict],
    limit: int = 5,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """
    행동 이벤트를 LLM에게 전달하여 행동 중심 추천 질문을 생성한다.

    동물 종류는 코드에서 dog/cat처럼 고정하지 않는다.
    behavior_events에 기록된 subject/summary/action을 보고 LLM이 자연스럽게 판단한다.
    OPENAI_API_KEY가 없거나 호출 실패 시에는 동물명을 특정하지 않고 "반려동물" 기반 fallback 질문을 반환한다.
    """
    if not events:
        return DEFAULT_BEHAVIOR_QUESTIONS[:limit]

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return _fallback_questions_from_behavior_events(events, limit=limit)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = _build_behavior_question_generation_prompt(
            events=events,
            limit=limit,
        )

        response = client.responses.create(
            model=model,
            input=prompt,
        )

        output_text = response.output_text.strip()
        questions = _parse_questions(output_text)

        if questions:
            return questions[:limit]

    except Exception as exc:
        print(f"행동 기반 추천 질문 LLM 생성 실패: {exc}", flush=True)

    return _fallback_questions_from_behavior_events(events, limit=limit)


def generate_questions_from_events(
    events: list[dict],
    limit: int = 5,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """
    장면 후보를 LLM에게 전달하여 추천 질문을 생성한다.

    OPENAI_API_KEY가 없거나 호출 실패 시 fallback 질문을 반환한다.

    이 함수는 기존 객체/장면 후보 기반 추천질문용으로 유지한다.
    행동 이벤트가 없을 때 fallback으로 사용할 수 있다.
    """
    if not events:
        return DEFAULT_GENERATED_QUESTIONS[:limit]

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return _fallback_questions_from_events(events, limit=limit)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = _build_question_generation_prompt(events=events, limit=limit)

        response = client.responses.create(
            model=model,
            input=prompt,
        )

        output_text = response.output_text.strip()
        questions = _parse_questions(output_text)

        if questions:
            return questions[:limit]

    except Exception as exc:
        print(f"추천 질문 LLM 생성 실패: {exc}", flush=True)

    return _fallback_questions_from_events(events, limit=limit)


def _build_behavior_question_generation_prompt(events: list[dict], limit: int) -> str:
    event_text = format_behavior_events_for_prompt(events)

    return f"""
너는 반려동물 영상 검색 서비스의 추천 질문 생성 도우미다.

아래는 영상에서 분석된 반려동물 행동 이벤트이다.
사용자가 실제로 눌러서 영상에서 확인해볼 만한 추천 질문을 만들어라.

핵심 원칙:
- 영상에서 눈으로 직접 확인 가능한 장면만 질문으로 만들어라.
- 감정 진단, 원인 추정, 건강 상태 단정 질문은 만들지 마라.
- "왜", "이유", "목적", "불안", "안정", "편안", "호기심", "스트레스" 같은 표현은 쓰지 마라.
- 다만 영상으로 확인 가능한 범위의 "관심을 보인 장면", "반응을 확인할 수 있는 장면", "확인해볼 만한 장면" 표현은 사용할 수 있다.
- 질문은 짧고 명확하게 작성한다.
- 질문은 예/아니오로 답할 수 있거나, 특정 시점을 찾을 수 있는 형태로 만든다.
- 행동 이벤트에 나온 행동, 장소, 대상만 사용한다.
- 행동 이벤트에 없는 상황을 상상해서 질문하지 마라.
- 한 질문에는 하나의 행동만 담는다.
- 객체 이름만 묻는 질문은 만들지 마라.
- 동물 종류가 명확하지 않으면 "반려동물"이라고 표현한다.
- "탐색"이라는 단어를 사용한 질문은 최대 1개만 만들어라.
- 모든 행동 이벤트가 탐색처럼 보이더라도 질문을 전부 탐색 중심으로 만들지 마라.
- 서로 비슷한 질문을 반복하지 마라.
- 질문은 최대 {limit}개만 생성한다.
- JSON 배열만 출력한다.
- markdown 코드블록은 쓰지 마라.

좋은 질문의 형태:
- "~한 장면이 있어?"
- "~에 관심을 보인 장면이 있어?"
- "~의 반응을 확인할 수 있어?"
- "~에 오래 머문 구간은 언제야?"
- "~에 들어간 장면이 있어?"
- "~에서 나온 장면이 있어?"
- "~을 먹은 장면이 있어?"
- "~을 마신 장면이 있어?"
- "~위에 올라간 장면이 있어?"
- "~아래에 들어간 장면이 있어?"
- "~한 장면은 언제였어?"
- "~한 행동이 반복된 장면이 있어?"

행동 이벤트:
{event_text}

좋은 출력 예시:
[
  "물병에 관심을 보인 장면이 있어?",
  "밥그릇 근처에 오래 머문 구간은 언제야?",
  "차 안에 들어간 장면이 있어?",
  "소파 아래에 들어간 장면이 있어?",
  "침대 위에서 움직인 장면이 있어?",
  "사람이 등장한 뒤 반려동물의 반응을 확인할 수 있어?"
]

나쁜 출력 예시:
[
  "불안해 보이는 장면이 있어?",
  "안정된 모습이었어?",
  "왜 그곳에 들어갔어?",
  "무엇 때문에 관심을 보였어?",
  "사람에게 왜 반응했어?",
  "탐색하는 동안 편안해 보였어?",
  "반려동물이 보이나요?",
  "그릇이 있나요?"
]
""".strip()


def _parse_questions(output_text: str) -> list[str]:
    if not output_text:
        return []

    cleaned_text = output_text.strip()

    cleaned_text = cleaned_text.removeprefix("```json").removeprefix("```").strip()
    cleaned_text = cleaned_text.removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, list):
            return _clean_questions(parsed)
    except json.JSONDecodeError:
        pass

    lines = cleaned_text.splitlines()
    questions = []

    for line in lines:
        cleaned = line.strip()
        cleaned = cleaned.lstrip("-•0123456789. ")
        cleaned = cleaned.strip('"').strip("'").strip()

        if cleaned:
            questions.append(cleaned)

    return _clean_questions(questions)


def _clean_questions(items: list[Any]) -> list[str]:
    result = []
    seen = set()

    for item in items:
        question = str(item).strip()

        if not question:
            continue

        question = question.replace("\n", " ").strip()

        if not question.endswith("?"):
            if question.endswith("다"):
                question = question[:-1] + "나요?"
            else:
                question = question + "?"

        if question in seen:
            continue

        seen.add(question)
        result.append(question)

    return result


def _fallback_questions_from_behavior_events(events: list[dict], limit: int = 5) -> list[str]:
    """
    LLM 호출이 불가능할 때 사용하는 행동 이벤트 기반 fallback 질문 생성.

    모델 없이도 예전 추천질문처럼 사용자가 눌러보고 싶은 문장으로 만든다.
    단, 감정/원인을 단정하지 않고 영상에서 확인 가능한 관심, 반응, 반복, 체류 구간만 묻는다.
    """
    questions = []

    for event in events:
        subject = _subject_text_for_fallback(event.get("subject"))
        subject_with_particle = _subject_with_particle(subject)
        action = event.get("action") or ""
        target = event.get("target_object") or event.get("target")
        summary = event.get("summary") or ""
        duration = _safe_float(event.get("duration"))
        repeat_count = int(event.get("repeat_count") or 0)

        target_text = _target_to_korean(target)
        action_text = _action_to_question_text(action)

        if target_text:
            questions.append(f"{subject_with_particle} {target_text}에 관심을 보인 장면이 있나요?")
            questions.append(f"{target_text} 근처에 오래 머문 구간은 언제인가요?")

        if repeat_count >= 3:
            if target_text:
                questions.append(f"{subject_with_particle} {target_text} 주변에서 비슷한 행동을 반복한 장면이 있나요?")
            else:
                questions.append(f"{subject_with_particle} 같은 행동을 반복한 구간이 있나요?")

        if duration >= 10:
            if target_text:
                questions.append(f"{subject_with_particle} {target_text} 주변에 오래 머문 장면이 있나요?")
            else:
                questions.append(f"{subject_with_particle} 한 행동을 오래 지속한 구간이 있나요?")

        if "사람" in action or "person" in str(target).lower() or "사람" in summary:
            questions.append(f"사람이 등장한 뒤 {subject}의 반응을 확인할 수 있나요?")

        if action_text:
            questions.append(f"{subject_with_particle} {action_text} 장면을 확인할 수 있나요?")

    questions.extend(DEFAULT_BEHAVIOR_QUESTIONS)

    return _deduplicate_keep_order(questions)[:limit]


def _fallback_questions_from_events(events: list[dict], limit: int = 5) -> list[str]:
    """
    LLM 호출이 불가능할 때 사용하는 최소 fallback.
    행동을 확정하지 않고 장면 후보 기반 질문만 만든다.
    """
    questions = []

    for event in events:
        event_type = event.get("event_type")
        labels = event.get("labels", [])
        timestamp = event.get("timestamp")

        label_text = ", ".join(labels)

        if event_type == "new_object_appeared":
            questions.append(f"영상 중간에 새롭게 등장한 객체({label_text})가 있나요?")

        elif event_type in {"person_detected", "person_repeatedly_detected"}:
            questions.append("사람이 등장한 뒤 반려동물의 반응을 확인할 수 있나요?")

        elif event_type in {
            "pet_object_co_occurrence",
            "pet_object_pair_repeated",
            "repeated_pet_object_pair",
        }:
            questions.append(f"반려동물이 주변 객체({label_text})에 관심을 보인 장면이 있나요?")

        elif event_type == "person_pet_pair_repeated":
            questions.append("사람과 반려동물이 함께 보이는 구간에서 반려동물의 반응을 확인할 수 있나요?")

        elif event_type == "pet_repeatedly_detected":
            questions.append("반려동물이 반복적으로 감지된 구간에서 어떤 움직임이 있었나요?")

        elif event_type == "scene_label_changed":
            if timestamp is not None:
                questions.append(f"{float(timestamp):.1f}초 근처에서 장면 변화가 있었나요?")
            else:
                questions.append("영상 중간에 장면 구성이 크게 바뀐 구간이 있나요?")

        elif event_type in {"repeated_object_detected", "object_pair_repeated"}:
            questions.append(f"영상에서 {label_text} 조합이 반복해서 보이는 장면이 있나요?")

    questions.extend(DEFAULT_GENERATED_QUESTIONS)

    return _deduplicate_keep_order(questions)[:limit]


def _subject_text_for_fallback(subject: str | None) -> str:
    """
    fallback 질문에서 사용할 주체명.

    동물 종류를 코드 리스트로 판단하지 않는다.
    - subject를 특정 동물명으로 번역하지 않고 "반려동물"이라고 표현한다.
    """
    if not subject:
        return "반려동물"

    subject_text = str(subject).strip()

    if not subject_text:
        return "반려동물"

    # 영어 라벨은 코드에서 동물명을 확정하지 않는다.
    if subject_text.isascii():
        return "반려동물"

    return subject_text


def _subject_with_particle(subject: str) -> str:
    if not subject:
        return "반려동물이"

    last_char = subject[-1]
    if not ("가" <= last_char <= "힣"):
        return f"{subject}이"

    has_final_consonant = (ord(last_char) - ord("가")) % 28 != 0
    return f"{subject}{'이' if has_final_consonant else '가'}"


def _action_to_question_text(action: str | None) -> str | None:
    if not action:
        return None

    text = str(action).strip()
    if not text:
        return None

    text = re.sub(r"^반려동물(이|가)\s*", "", text)
    text = re.sub(r"^동물(이|가)\s*", "", text)
    text = text.replace("것으로 보임", "").replace("있는 것으로 보임", "")
    text = text.replace("것으로 추정됨", "").replace("있는 것으로 추정됨", "")
    text = text.strip(" .")

    if not text:
        return None

    return text


def _target_to_korean(target: str | None) -> str | None:
    if not target:
        return None

    target = str(target).strip()

    mapping = {
        "bowl": "그릇",
        "cup": "컵",
        "bottle": "물병",
        "bed": "침대",
        "couch": "소파",
        "chair": "의자",
        "dining table": "테이블",
        "sports ball": "공",
        "remote": "리모컨",
        "person": "사람",
    }

    return mapping.get(target.lower(), target)

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _deduplicate_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        item = item.strip()

        if not item:
            continue

        if item in seen:
            continue

        seen.add(item)
        result.append(item)

    return result
