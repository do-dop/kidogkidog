"""Offer concrete follow-up questions when a count request has no clear action."""

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
    "front_paw_raise": r"앞발|발을\s*들|발\s*듦",
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
    """Rank supported actions using the user's words and this video's events."""
    ranked = []
    for action in ACTION_PATTERNS:
        clue = bool(re.search(_ACTION_CLUES[action], query, re.I))
        evidence_count = sum(event_matches_action(event, action) for event in events)
        if clue or evidence_count:
            ranked.append((int(clue), evidence_count, action))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    options = [_option(action, events) for _, _, action in ranked[:3]]
    if len(options) == 1:
        action = ranked[0][2]
        target = _target_for(action, events)
        if action == "front_paw_raise" and target:
            prompt = f"{target} 앞에서 앞발을 든 횟수를 물으셨나요?"
        else:
            prompt = f"{ACTION_PATTERNS[action][1]} 횟수를 물으셨나요?"
    else:
        prompt = "어떤 행동의 횟수를 물으셨나요? 아래 질문을 선택하거나 직접 고쳐 입력해 주세요."
    return {"question": prompt, "options": options}
