import json
import os
import re
from typing import Any

from pipeline.behavior_event_extractor import format_behavior_events_for_prompt


DEFAULT_GENERATED_QUESTIONS = [
    "반려동물이 특정 위치 근처에 머무른 장면이 있나요?",
    "영상 중간에 새롭게 등장한 객체나 사람이 있나요?",
    "반려동물이 움직이거나 위치를 바꾼 장면이 있나요?",
    "사람이 함께 나온 장면이 있나요?",
    "처음과 달라진 장면이 있었나요?",
]


# 근거 이벤트가 없으면 query_suggester의 기존 라벨 보충에 맡긴다.
DEFAULT_BEHAVIOR_QUESTIONS = []

_DURATION_PATTERN = re.compile(r"오래|오랜|한동안|계속|지속|줄곧|내내|장시간|끊임|꾸준|종일")
_REPEAT_PATTERN = re.compile(r"반복|여러\s*번|자주|거듭|되풀이|빈번|수차례|몇\s*번|두\s*번|[2-9]\s*(?:번|회)")
_INTERPRETATION_PATTERN = re.compile(
    r"기다|기대|관심|호기심|불안|편안|행복|슬퍼|슬프|기쁨|즐거|좋아|싫어|집중|"
    r"원하|원한|원하는|하려|[가-힣]+려는|[가-힣]+려고|같아|같은가|같나요|같은지|"
    r"감정|기분|의도|목적|이유|왜|스트레스|안정|흥분|긴장|지루|심심|외로|무서|두려"
)


def _question_rejection_reason(question: str, event: dict) -> str | None:
    """문장은 고치지 않고 부적절한 후보 전체를 제외한다."""
    if _DURATION_PATTERN.search(question):
        return "지속시간 강조 표현 금지"
    if _INTERPRETATION_PATTERN.search(question):
        return "감정/의도 해석"
    count = _safe_float(event.get("repeat_count"))
    if _REPEAT_PATTERN.search(question) and not (2 <= count < float("inf")):
        return "repeat_count >= 2 근거 없음"
    return None


def _validate_behavior_questions(
    items: list, events: list[dict], limit: int, *, with_sources: bool = False,
) -> list:
    # 기존 최종 연결 규칙도 확인한다. 생성 근거와 최종 연결 이벤트 양쪽에서
    # 반복 근거가 있어야 하고, 최종 연결 이벤트당 하나만 남긴다.
    if not events or limit <= 0:
        return []

    from pipeline.query_suggester import _best_event_for_question

    result, seen_events, seen_questions = [], set(), set()
    for item in items:
        if not isinstance(item, dict):
            continue
        index, question = item.get("event_index"), item.get("question")
        if type(index) is not int or not 1 <= index <= len(events):
            continue
        if not isinstance(question, str) or not question.strip():
            continue
        question = " ".join(question.split())
        if not question.endswith("?"):
            question += "?"
        source = events[index - 1]
        linked = _best_event_for_question(question, events) or source
        if _question_rejection_reason(question, source) or _question_rejection_reason(question, linked):
            continue
        source_key = source.get("id") if source.get("id") is not None else id(source)
        linked_key = linked.get("id") if linked.get("id") is not None else id(linked)
        normalized = re.sub(r"\s|[?!.,]", "", question)
        if source_key in seen_events or linked_key in seen_events or normalized in seen_questions:
            continue
        # 문장 검증/중복 규칙은 유지하고, 통과한 후보의 원본 근거만 함께 전달한다.
        result.append({
            "question": question,
            "source_event_id": source.get("id"),
            "original_event_id": source.get("id"),
            "event_start": source.get("start_time"),
            "event_end": source.get("end_time"),
        } if with_sources else question)
        seen_events.update((source_key, linked_key))
        seen_questions.add(normalized)
        if len(result) >= limit:
            break
    return result


def _parse_behavior_questions(output_text: str) -> list:
    text = output_text.strip().removeprefix("```json").removeprefix("```").strip()
    text = text.removesuffix("```").strip()
    try:
        items = json.loads(text)
    except (ValueError, TypeError):
        return []
    return items if isinstance(items, list) else []


def generate_questions_from_behavior_events(
    events: list[dict], limit: int = 5, model: str = "gpt-4o-mini",
) -> list[str]:
    """기존 문자열 API는 유지한다. 추천 연결에는 구조화된 item API를 사용한다."""
    return [item["question"] for item in generate_question_items_from_behavior_events(
        events, limit=limit, model=model,
    )]


def generate_question_items_from_behavior_events(
    events: list[dict],
    limit: int = 5,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    행동 이벤트를 LLM에게 전달하여 행동 중심 추천 질문을 생성한다.

    동물 종류는 코드에서 dog/cat처럼 고정하지 않는다.
    behavior_events에 기록된 subject/summary/action을 보고 LLM이 자연스럽게 판단한다.
    OPENAI_API_KEY가 없거나 호출 실패 시에는 관찰 가능한 절에 기반한 fallback 질문을 반환한다.
    """
    if not events or limit <= 0:
        return []

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return _behavior_fallback_items(events, limit)

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
        # 정상 응답에서 탈락한 후보를 억지로 채우지 않는다.
        items = _validate_behavior_questions(
            _parse_behavior_questions(output_text), events, limit, with_sources=True,
        )
        return [dict(item, question_source="LLM") for item in items]

    except Exception as exc:
        print(f"행동 기반 추천 질문 LLM 생성 실패: {exc}", flush=True)

    return _behavior_fallback_items(events, limit)


def _behavior_fallback_items(events: list[dict], limit: int) -> list[dict]:
    items = _fallback_questions_from_behavior_events(events, limit, with_sources=True)
    return [dict(item, question_source="behavior fallback") for item in items]


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
영상에서 직접 보고 맞다/아니다를 확인 가능한 사실만 짧게 질문한다.
우선순위: 1. 먹기, 마시기, 접근, 이동, 들어가기, 나오기, 뛰기, 눕기, 앉기
2. 그릇 근처, 소파 아래, 문 앞, 사람 옆, 급식기 근처 등 명시된 위치
3. 사람이나 특정 물체의 등장 여부.
이벤트에 없는 행동, 대상, 위치를 만들지 않는다. 한 질문에는 하나의 사실만 담는다.
기다림, 관심, 호기심, 불안, 편안, 좋아함, 싫어함, 집중, 행복,
원하는 것 같다, ~하려는 것 같다 등 감정/의도/원인 해석은 질문하지 않는다.
summary/action에 해석이 있어도 명시된 관찰 사실만 사용하고, 없으면 제외한다.
예: '자동 급식기 앞에서 배식을 기다리는 것으로 보임'
→ '자동 급식기 근처에 있는 장면이 있어?'
예: '주변을 돌아다니며 호기심을 보임' → '주변을 돌아다닌 장면이 있어?'
시선 방향이 명시적으로 관찰된 경우에만 '급식기 쪽을 바라본 장면이 있어?'를 쓴다.
'집중함'만으로 시선 방향을 추론하지 않는다.
'오래', '오랜 시간', '한동안', '계속', '지속', '내내' 등 지속시간 강조는
이번에는 duration 값에 관계없이 사용하지 않는다. 이벤트 길이는 행동 지속시간 보장이 아니다.
'반복', '여러 번', '자주'는 해당 이벤트 repeat_count >= 2일 때만 허용한다.
프레임 감지 횟수만 있거나 실제 행동 반복이 불명확하면 반복 표현을 쓰지 않는다.
각 이벤트당 질문은 최대 1개. 서로 다른 이벤트의 행동/대상을 우선한다.
서로 다른 이벤트라도 같은 물 마시기 등 의미가 같은 질문은 하나만 만든다.
최대 {limit}개이며 근거가 부족하면 더 적게 또는 빈 배열을 반환한다.
좋은 문장: '사료 그릇에 접근한 장면이 있어?', '물을 마신 장면이 있어?',
'소파 아래에 들어간 적이 있어?', '사람이 함께 나온 장면이 있어?',
'다른 공간으로 이동한 장면이 있어?' (각 사실이 이벤트에 있을 때만).
반드시 아래 JSON 배열 형식으로만 출력한다. event_index는 아래 목록의 1부터 시작하는 번호다.
[{{"event_index": 1, "question": "관찰 가능한 사실을 묻는 질문?"}}]
행동 이벤트:
{event_text}
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


def _fallback_questions_from_behavior_events(
    events: list[dict], limit: int = 5, *, with_sources: bool = False,
) -> list:
    """관찰 가능한 절만 선택한다. 자유 형식 action을 그대로 문장에 붙이지 않는다."""
    candidates = []
    for index, event in enumerate(events, 1):
        text = " ".join(str(event.get(key) or "") for key in ("action", "summary"))
        # 부정/희망/추정이 붙은 서술을 확정 행동으로 바꾸지 않는다.
        if re.search(r"않|못|없|싶|추정", text):
            continue
        question = None
        # 완성된 관찰 절만 매칭한다. '마시려는', '들어가고 싶은' 등은 제외한다.
        patterns = [
            (r"(?:물을?\s*마시(?:는|고|며|다)|물을?\s*마심)", "물을 마신 장면이 있어?"),
            (r"사료\s*그릇에\s*(?:접근(?:하여|한|함)|다가(?:가|간))", "사료 그릇에 접근한 장면이 있어?"),
            (r"(?:먹이를?\s*먹(?:는|고|음)|사료를?\s*먹(?:는|고|음))", "먹이를 먹은 장면이 있어?"),
            (r"소파\s*아래에?\s*들어(?:간|감|가는)", "소파 아래에 들어간 장면이 있어?"),
            (r"장애물을?\s*넘(?:어서는|는|은|었)", "장애물을 넘은 장면이 있어?"),
            (r"돌아다니(?:며|는|고)|돌아다닌", "주변을 돌아다닌 장면이 있어?"),
            (r"이동(?:하는|한|함)", "이동한 장면이 있어?"),
            (r"움직임을 보인|움직이는", "움직인 장면이 있어?"),
            (r"사람이\s*(?:함께\s*)?(?:나온|등장|보이)", "사람이 함께 나온 장면이 있어?"),
            (r"자동\s*급식기\s*(?:앞|주변|근처)", "자동 급식기 근처에 있는 장면이 있어?"),
        ]
        for pattern, candidate in patterns:
            if re.search(pattern, text):
                question = candidate
                break
        if question:
            candidates.append({"event_index": index, "question": question})
    return _validate_behavior_questions(candidates, events, limit, with_sources=with_sources)


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
