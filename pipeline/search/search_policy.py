"""Shared, deterministic limits and contradiction checks for scene search."""

import re
from collections import deque


def diverse_video_candidates(rows, limit):
    """Reserve one ranked candidate per video before taking another occurrence."""
    groups = {}
    for index, row in enumerate(rows):
        key = row.get("video_id") or ("unknown", index)
        groups.setdefault(key, deque()).append(row)
    selected = []
    while groups and len(selected) < limit:
        for key in list(groups):
            selected.append(groups[key].popleft())
            if not groups[key]:
                del groups[key]
            if len(selected) >= limit:
                break
    return selected


# Only reject an explicit opposite action; an unknown action is left to the
# semantic matcher. Generic eating does not establish nursing, or vice versa.
_OPPOSITES = (
    (r"물.{0,5}마|마시|마셔|음수", r"사료|밥.{0,3}먹|먹이.{0,3}먹|젖.{0,3}먹"),
    (r"사료|밥.{0,3}먹", r"물.{0,5}마|마시|마셔|음수|젖.{0,3}먹"),
    (r"젖.{0,5}먹|수유", r"사료|밥.{0,3}먹|물.{0,5}마|마시"),
    (r"들어가|들어간|들어갔|진입", r"나오|나온|나왔|나가|나간|나감"),
    (r"나오|나온|나왔|나가|나간|나감", r"들어가|들어간|들어갔|진입"),
    (r"넘어지|넘어진|넘어졌|쓰러|뒤집|젖혀", r"올라가|올라간|올라갔|올라탐|올라타"),
)


def contradicts_action(query, event):
    description = " ".join(str(event.get(key) or "") for key in ("action", "summary"))
    for requested, opposite in _OPPOSITES:
        if (re.search(requested, query) and not re.search(opposite, query)
                and re.search(opposite, description) and not re.search(requested, description)):
            return True
    return False
