import json
import os
from typing import Any

from pipeline.scene_event_extractor import format_events_for_prompt


DEFAULT_GENERATED_QUESTIONS = [
    "반려동물이 특정 위치 근처에 머무른 장면이 있나요?",
    "영상 중간에 새롭게 등장한 객체나 사람이 있나요?",
    "반려동물이 움직이거나 위치를 바꾼 장면이 있나요?",
    "반려동물이 주변 물체에 관심을 보인 장면이 있나요?",
    "처음과 달라진 장면이 있었나요?",
]


def generate_questions_from_events(
    events: list[dict],
    limit: int = 5,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """
    장면 후보를 LLM에게 전달하여 추천 질문을 생성한다.

    OPENAI_API_KEY가 없거나 호출 실패 시 fallback 질문을 반환한다.
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


def _build_question_generation_prompt(events: list[dict], limit: int) -> str:
    event_text = format_events_for_prompt(events)

    return f"""
너는 반려동물 영상 검색 서비스의 추천 질문 생성 도우미다.

아래는 영상에서 자동 추출된 장면 후보이다.
중요한 점:
- 행동을 확정하지 마라.
- "강아지가 잤다", "고양이가 탈출했다"처럼 단정하지 마라.
- 사용자가 영상에서 확인해볼 만한 질문을 만들어라.
- 질문은 한국어로 작성한다.
- 질문은 "~있나요?", "~보이나요?", "~확인할 수 있나요?"처럼 조심스러운 표현을 사용한다.
- 영상에서 확인 가능한 시각 정보에 관한 질문만 만든다.
- 최대 {limit}개만 생성한다.
- JSON 배열만 출력한다.

장면 후보:
{event_text}

출력 예시:
[
  "강아지가 특정 물체 근처에 머무른 장면이 있나요?",
  "사람이 등장한 뒤 반려동물이 반응한 장면이 있나요?"
]
""".strip()


def _parse_questions(output_text: str) -> list[str]:
    if not output_text:
        return []

    try:
        parsed = json.loads(output_text)
        if isinstance(parsed, list):
            return _clean_questions(parsed)
    except json.JSONDecodeError:
        pass

    lines = output_text.splitlines()
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

        if not question.endswith("?") and not question.endswith("나요"):
            if question.endswith("다"):
                question = question[:-1] + "나요?"
            else:
                question = question + "?"

        if question in seen:
            continue

        seen.add(question)
        result.append(question)

    return result


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

        elif event_type == "person_detected":
            questions.append("사람이 등장한 뒤 반려동물의 반응을 확인할 수 있나요?")

        elif event_type == "pet_object_co_occurrence":
            questions.append(f"반려동물이 주변 객체({label_text}) 근처에 머무른 장면이 있나요?")

        elif event_type == "scene_label_changed":
            if timestamp is not None:
                questions.append(f"{float(timestamp):.1f}초 근처에서 장면 변화가 있었나요?")
            else:
                questions.append("영상 중간에 장면 구성이 크게 바뀐 구간이 있나요?")

        elif event_type == "repeated_pet_object_pair":
            questions.append(f"영상에서 {label_text} 조합이 반복해서 보이는 장면이 있나요?")

    questions.extend(DEFAULT_GENERATED_QUESTIONS)

    return _deduplicate_keep_order(questions)[:limit]


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