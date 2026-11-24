"""Offer concrete follow-ups when a question omits its action or referent."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from pipeline.search.aggregate_query import ACTION_PATTERNS, event_matches_action


_ACTION_CLUES = {
    "drink": r"마시|음수|물(?!속|안|에)",
    "eat": r"먹|밥|사료|간식|음식",
    "gate_climb": r"울타리|펜스|우리|넘|기어",
    "water_enter": r"수영장|분수|물속|입수",
    "front_paw_raise": r"앞\s*발|발을\s*들|발\s*듦",
    "sofa_up": r"소파|쇼파",
    "head_lift": r"고개|머리\s*들",
    "camera_approach": r"카메라|렌즈",
}

_COUNT_QUESTIONS = {
    "drink": "물을 몇 번 마셨어?",
    "eat": "음식을 몇 번 먹었어?",
    "gate_climb": "울타리에 몇 번 올라갔어?",
    "water_enter": "물속에 몇 번 들어갔어?",
    "front_paw_raise": "앞발을 몇 번 들었어?",
    "sofa_up": "소파에 몇 번 올라갔어?",
    "head_lift": "고개를 몇 번 들었어?",
    "camera_approach": "카메라에 몇 번 가까이 다가갔어?",
}

_SCENE_QUESTIONS = {
    "drink": "물을 마시는 장면 보여줘",
    "eat": "음식을 먹는 장면 보여줘",
    "gate_climb": "울타리에 올라가는 장면 보여줘",
    "water_enter": "물속에 들어가는 장면 보여줘",
    "front_paw_raise": "앞발을 들어 올리는 장면 보여줘",
    "sofa_up": "소파에 올라가는 장면 보여줘",
    "head_lift": "고개를 드는 장면 보여줘",
    "camera_approach": "카메라에 가까이 다가가는 장면 보여줘",
}

_VAGUE_REFERENCE = re.compile(
    r"그거|이거|저거|그\s*장면|이\s*장면|저\s*장면|그\s*행동|이\s*행동|"
    r"그랬|그때|그\s*다음|무슨\s*행동|어떤\s*행동|뭐\s*했",
    re.I,
)
_OVERVIEW_REQUEST = re.compile(r"오늘|어제|하루|요약|전체|타임라인")
_SPECIFIC_CONDITION = re.compile(
    r"점프|뛰|달리|걷|먹|마시|올라|내려|넘|들어가|나가|목욕|잠|자고|놀|"
    r"접근|다가|짖|핥|떨|떨어|두\s*마리|몇\s*마리|빨간|파란|색깔",
    re.I,
)


def _target_for(action: str, events: list[dict[str, Any]]) -> str | None:
    targets = Counter(
        str(event.get("target_object") or "").strip()
        for event in events if event_matches_action(event, action)
    )
    for target, _ in targets.most_common():
        if target and len(target) <= 12 and target not in {"없음", "반려동물", "앞발", "물"}:
            return target
    return None


def _option(action: str, events: list[dict[str, Any]]) -> dict[str, str]:
    target = _target_for(action, events)
    question = _COUNT_QUESTIONS[action]
    if action == "front_paw_raise" and target:
        question = f"{target} 앞에서 앞발을 몇 번 들었어?"
    elif action == "water_enter" and target:
        question = f"{target} 물속에 몇 번 들어갔어?"
    return {"label": question, "query": question}


def build_count_clarification(query: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Only suggest an action named in the question; events supply its context."""
    ranked = []
    for action in ACTION_PATTERNS:
        clue = bool(re.search(_ACTION_CLUES[action], query, re.I))
        if not clue:
            continue
        evidence_count = sum(event_matches_action(event, action) for event in events)
        ranked.append((evidence_count, action))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    options = [_option(action, events) for _, action in ranked[:3]]
    if len(options) == 1:
        action = ranked[0][1]
        target = _target_for(action, events)
        if action == "front_paw_raise" and target:
            prompt = f"{target} 앞에서 앞발을 든 횟수를 물으셨나요?"
        else:
            prompt = f"{ACTION_PATTERNS[action][1]} 횟수를 물으셨나요?"
    elif options:
        prompt = "어떤 행동의 횟수를 물으셨나요? 아래 질문을 선택하거나 직접 고쳐 입력해 주세요."
    else:
        prompt = "어떤 행동의 횟수를 세어드릴까요? 행동과 세는 기준을 알려 주세요."
    return {"question": prompt, "options": options}


def needs_scene_clarification(query: str) -> bool:
    """Recognize missing referents, without treating every failed search as vague."""
    return bool(
        _VAGUE_REFERENCE.search(query)
        and not _OVERVIEW_REQUEST.search(query)
        and not _SPECIFIC_CONDITION.search(query)
        and not any(re.search(pattern, query, re.I) for pattern, _ in ACTION_PATTERNS.values())
    )


def build_scene_clarification(query: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Suggest recorded actions as questions, without claiming they match the user's intent."""
    clues = re.sub(_VAGUE_REFERENCE, " ", query)
    clues = re.sub(r"보여\s*줘|찾아\s*줘|있어|했어|해줘|강아지|고양이|반려동물", " ", clues)
    clue_terms = [part for part in re.findall(r"[가-힣a-zA-Z]{2,}", clues) if len(part) >= 2]
    if not clue_terms:
        return {"question": "무엇을 보여드릴까요? 찾고 싶은 행동이나 장면을 알려 주세요.", "options": []}
    candidates: dict[str, tuple[int, str]] = {}
    for event in events:
        description = " ".join(str(event.get(key) or "") for key in ("action", "target_object", "summary"))
        if not description.strip():
            continue
        action = next((name for name in ACTION_PATTERNS if event_matches_action(event, name)), None)
        if action:
            option = _SCENE_QUESTIONS[action]
            target = _target_for(action, events)
            if action == "front_paw_raise" and target:
                option = f"{target} 앞에서 앞발을 들어 올리는 장면 보여줘"
            elif action == "water_enter" and target:
                option = f"{target} 물속에 들어가는 장면 보여줘"
            key = action
        else:
            action_text = re.sub(r"[.!?]+$", "", str(event.get("action") or "").strip())
            if not action_text or len(action_text) > 70:
                continue
            option = f"{action_text} 장면 보여줘"
            key = action_text
        score = sum(2 for part in clue_terms if part in description)
        previous = candidates.get(key)
        candidates[key] = (score + (previous[0] if previous else 0) + 1, option)
    ranked = sorted(candidates.values(), key=lambda item: -item[0])[:3]
    options = [{"label": text, "query": text} for _, text in ranked]
    if len(options) == 1:
        scene = options[0]["query"].removesuffix(" 장면 보여줘")
        prompt = f"{scene} 장면을 찾으셨나요?"
    elif options:
        prompt = "찾고 싶은 장면이 아래 중 하나인가요? 선택하거나 직접 설명해 주세요."
    else:
        prompt = "어떤 장면을 찾고 싶으신가요? 행동이나 장소를 조금 더 알려 주세요."
    return {"question": prompt, "options": options}
