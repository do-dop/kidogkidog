"""Count distinct, evidence-backed behavior episodes across a complete timeline."""

import re
from typing import Any


ACTION_PATTERNS = {
    "drink": (r"마시|마심|마셨|마셨어|마신|마셔|음수|물\s*먹|drink|drank", "물 마시기"),
    "eat": (r"(?:밥|사료|먹이|간식|음식).{0,8}먹|먹었|먹는|먹고|먹음|식사|섭취|eat|eating", "먹기"),
    "gate_climb": (
        r"(?:울타리|펜스|gate).{0,14}(?:올라|올랐|기어|타고|climb)|"
        r"(?:올라|올랐|기어|타고|climb).{0,14}(?:울타리|펜스|gate)",
        "울타리 올라타기",
    ),
    "water_enter": (
        r"(?:분수|수영장|물속|물\s*안).{0,16}(?:들어|들어갔|입수)|"
        r"(?:들어|들어갔|입수).{0,16}(?:분수|수영장|물속|물\s*안)",
        "물속 들어가기",
    ),
    "front_paw_raise": (r"앞발.{0,16}(?:들|올리|올렸)|(?:들|올리|올렸).{0,16}앞발", "앞발 들기"),
    "sofa_up": (
        r"(?:소파|쇼파|sofa).{0,16}(?:올라|올랐|올리|점프|타고)|"
        r"(?:올라|올랐|올리|점프|타고).{0,16}(?:소파|쇼파|sofa)",
        "소파 올라가기",
    ),
    "head_lift": (r"고개.{0,8}(?:들|올리|들어)", "고개 들기"),
}
COUNT_PATTERN = re.compile(r"몇\s*(?:번|회)|횟수", re.I)
UNCERTAIN_PATTERN = re.compile(r"마시지\s*않|먹지\s*않|마시는지|먹는지|근처|주변|앞에|옆에", re.I)


def parse_count_action(query: str) -> tuple[str, str] | None:
    if not COUNT_PATTERN.search(query):
        return None
    # 식사 중 고개 들기 질문은 식사 횟수와 단위가 다르다.
    if re.search(r"고개.{0,8}(?:들|올리|들어)", query):
        return ("head_lift", "고개 들기")
    matches = [
        (action, label)
        for action, (pattern, label) in ACTION_PATTERNS.items()
        if re.search(pattern, query, re.I)
    ]
    return matches[0] if len(matches) == 1 else None


def is_scene_event(event: dict[str, Any]) -> bool:
    """소파가 넘어지는 것처럼 반려동물이 아닌 물건이 주체인 장면 변화 이벤트(주체가 곧 대상 물건)."""
    subject = str(event.get("subject") or "").strip()
    return bool(subject) and subject == str(event.get("target_object") or "").strip()


def event_matches_action(event: dict[str, Any], action: str) -> bool:
    # 반려동물의 행동 횟수를 세는 질문이므로 물건의 변화는 세지 않는다.
    if is_scene_event(event):
        return False
    description = " ".join(str(event.get(key) or "") for key in ("action", "summary", "target_object"))
    # 검수 메모에 적힌 부정 표현을 행동의 양성 근거로 세지 않는다.
    # 예: "사료를 먹는 장면이 아님"은 '먹기' 이벤트가 아니다.
    if re.search(r"(?:장면|행동|모습)?(?:이|은|는)?\s*(?:아님|아니|확인되지\s*않|보이지\s*않|없음)", description, re.I):
        return False
    if action == "gate_climb":
        if re.search(r"올라|올랐|기어|타고|넘|climb", description, re.I) is None:
            return False
        if re.search(r"울타리|펜스|우리(?:를|에|위)|gate|fence", description, re.I) is None:
            return False
        if re.search(r"못\s*올라", description, re.I):
            return False
        if re.search(r"시도|하려|려고", description, re.I):
            evidence = " ".join(str(item) for item in (event.get("evidence") or []))
            return bool(re.search(r"발.{0,6}걸치|손.{0,6}뻗|몸.{0,6}올라|점프", evidence, re.I))
        return True
    if action == "water_enter":
        return bool(re.search(ACTION_PATTERNS[action][0], description, re.I))
    if action == "front_paw_raise":
        return bool(re.search(ACTION_PATTERNS[action][0], description, re.I))
    if UNCERTAIN_PATTERN.search(description):
        return False
    return bool(re.search(ACTION_PATTERNS[action][0], description, re.I))


def count_episodes(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping records of the same episode, independently per video."""
    valid = []
    for event in events:
        try:
            start = float(event["start_time"])
            end = float(event["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end < start or not event.get("video_id"):
            continue
        valid.append((str(event["video_id"]), start, end, event))

    valid.sort(key=lambda item: (item[0], item[1], item[2]))
    episodes: list[dict[str, Any]] = []
    for video_id, start, end, event in valid:
        if episodes and episodes[-1]["video_id"] == video_id and start <= episodes[-1]["end_time"]:
            episode = episodes[-1]
            episode["end_time"] = max(episode["end_time"], end)
            episode["event_ids"].append(event.get("id"))
            episode["events"].append(event)
        else:
            episodes.append({
                "video_id": video_id,
                "start_time": start,
                "end_time": end,
                "event_ids": [event.get("id")],
                "events": [event],
            })
    return episodes
