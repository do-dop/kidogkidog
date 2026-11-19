"""Short questions derived from explicit, searchable behavior events.

The rules use the recorded action, not a video's display title.  Each result
keeps its event ID so the search can play the source interval directly.
"""

import re


_UNCONFIRMED = re.compile(r"않|못|없|싶|추정|하려|려고|것으로 보")
_RULES = (
    (re.compile(r"목줄이.*떨어"), "목줄이 떨어진 장면이 있어?"),
    (re.compile(r"매달린 물체를 향해 앞발을 올"), "매달린 물체를 향해 앞발을 올린 장면이 있어?"),
    (re.compile(r"나뭇가지를 향해 고개를 들어 올"), "나뭇가지를 향해 고개를 든 장면이 있어?"),
    (re.compile(r"나무 위에서.*몸을 움직"), "나무 위에서 몸을 움직인 장면이 있어?"),
    (re.compile(r"장난감을 향해 몸을 숙"), "장난감을 향해 몸을 숙인 장면이 있어?"),
    (re.compile(r"장난감을 바라"), "장난감을 바라본 장면이 있어?"),
    (re.compile(r"소파 위로 올라"), "소파 위로 올라간 장면이 있어?"),
    (re.compile(r"뒤를 바라보며 서"), "뒤를 바라보며 서 있는 장면이 있어?"),
    (re.compile(r"(?:문|현관) 쪽으로 (?:이동|다가)"), "문 쪽으로 이동한 장면이 있어?"),
    (re.compile(r"소파 쪽으로 (?:이동|다가)"), "소파 쪽으로 이동한 장면이 있어?"),
    (re.compile(r"카메라 쪽으로 다가"), "카메라 쪽으로 다가온 장면이 있어?"),
)


def grounded_event_question_items(events: list[dict], limit: int) -> list[dict]:
    """Select distinct observed actions, retaining their exact source event."""
    if limit <= 0:
        return []

    result = []
    seen_questions = set()
    for event in events:
        action = str(event.get("action") or "").strip()
        if event.get("id") is None or not action or _UNCONFIRMED.search(action):
            continue
        for pattern, question in _RULES:
            if not pattern.search(action) or question in seen_questions:
                continue
            result.append({
                "question": question,
                "source_event_id": event.get("id"),
                "original_event_id": event.get("id"),
                "event_start": event.get("start_time"),
                "event_end": event.get("end_time"),
                "question_source": "observed action",
            })
            seen_questions.add(question)
            break
        if len(result) >= limit:
            break
    return result
