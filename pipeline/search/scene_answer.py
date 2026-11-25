"""Short Korean answers based only on returned scene evidence."""

import re
import math

from pipeline.search.video_time import frame_video_seconds


_ACTIONS = (
    (r"젖.{0,6}먹|수유", "젖 먹는"),
    (r"마시|마심|마셔|음수|물.{0,3}먹", "물 마시는"),
    (r"(?:사료|음식|먹이|밥|간식).{0,12}(?:먹|섭취)|식사", "밥 먹는"),
    (r"목욕|씻고|씻는", "목욕하는"),
    (r"잠을 자|잠자|낮잠", "잠자는"),
    (r"몸.{0,5}(?:털|터는)", "몸을 터는"),
)


def _action(row):
    description = str(row.get("summary") or "")
    return next((label for pattern, label in _ACTIONS if re.search(pattern, description)), None)


def build_scene_answer(results):
    if not results:
        return "찾으시는 장면을 아직 확인하지 못했어요. 다른 행동이나 시간대로 찾아보세요."
    first = results[0]
    action = _action(first)
    same_action = action is not None and all(_action(row) == action for row in results)
    descriptions = {str(row.get("summary") or "").strip().rstrip(".") for row in results}
    if same_action:
        subject = f"{action} 장면"
    elif len(descriptions) == 1 and next(iter(descriptions)):
        subject = f"‘{next(iter(descriptions))}’ 장면"
    else:
        subject = "관련 장면"
    partial = any(row.get("match_level") in {"related", "related_event"} for row in results)
    video_count = len({row.get("video_id") for row in results if row.get("video_id")})
    if video_count > 1:
        answer = f"{subject}을 영상 {video_count}개에서 찾았어요. 아래에서 보고 싶은 장면을 골라 보세요."
    else:
        timestamp = first.get("representative_timestamp")
        if timestamp is None:
            timestamp = first.get("timestamp") or 0
        whole = max(0, math.floor(frame_video_seconds(first, float(timestamp)) or 0))
        minute, second = divmod(whole, 60)
        moment = f"{minute:02d}:{second:02d}"
        location = f"영상 {moment}쯤부터"
        answer = f"{subject}을 찾았어요. {location} 확인해 보세요."
    if partial:
        answer += " 다만 질문의 세부 조건까지는 확인하지 못했어요. 비슷한 장면으로 참고해 주세요."
    return answer
