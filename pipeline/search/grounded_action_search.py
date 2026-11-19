"""Resolve common free-form action requests from timestamped behavior events.

This path requires both the requested object/place and an observed action.  It
does not turn a video title or a detected object into proof of an action.
"""

import re

from db.behavior_events import get_behavior_events_for_aggregation
from pipeline.search.aggregate_query import count_episodes
from pipeline.search.event_text_search import event_as_search_result


_UNCERTAIN = re.compile(r"않|못|없|싶|추정|하려|려고")
_INTENTS = (
    ("카메라 접근", re.compile(r"카메라"), re.compile(r"가까|다가|접근|얼굴"),
     re.compile(r"카메라"), re.compile(r"다가|접근|가까")),
    ("문 쪽 이동", re.compile(r"문|현관"), re.compile(r"이동|다가|접근|걸어|갔|가(?:는|니|나요|봤|서)"),
     re.compile(r"문|현관"), re.compile(r"이동|다가|접근|걸어")),
    ("목줄 떨어짐", re.compile(r"목줄"), re.compile(r"떨어|늘어|내려"),
     re.compile(r"목줄"), re.compile(r"떨어|늘어|내려")),
    ("장난감 바라보기", re.compile(r"장난감"), re.compile(r"바라|쳐다|봐|보는"),
     re.compile(r"장난감"), re.compile(r"바라|쳐다")),
    ("장난감 쪽으로 숙이기", re.compile(r"장난감"), re.compile(r"숙|고개.{0,4}내"),
     re.compile(r"장난감"), re.compile(r"숙|고개.{0,4}내")),
    ("소파 쪽 이동", re.compile(r"소파|쇼파"), re.compile(r"이동|다가|접근|걸어|갔|가(?:는|니|나요|봤|서)"),
     re.compile(r"소파|쇼파"), re.compile(r"이동|다가|접근|걸어")),
)


def find_grounded_action_scenes(query, video_id, top_k, recording_date=None, time_range=None):
    """Return playable episodes for an explicitly recorded action, if any."""
    if not video_id or recording_date or time_range:
        return None
    matching_intents = [item for item in _INTENTS if item[1].search(query) and item[2].search(query)]
    if len(matching_intents) != 1:
        return None
    label, _, _, target_pattern, action_pattern = matching_intents[0]

    try:
        events = get_behavior_events_for_aggregation(video_id=video_id)
    except Exception as exc:
        print(f"행동 이벤트 장면 조회 실패: {exc}", flush=True)
        return None
    matched = [
        event for event in events
        if target_pattern.search(str(event.get("target_object") or "") + " " + str(event.get("action") or ""))
        and action_pattern.search(str(event.get("action") or ""))
        and not _UNCERTAIN.search(str(event.get("action") or ""))
        and event.get("source_frames")
    ]
    if not matched:
        return None

    results = []
    episodes = count_episodes(matched)
    for episode in episodes:
        for event in episode["events"]:
            result = event_as_search_result(event, float(event.get("confidence") or 0.5))
            if result and result.get("s3_key"):
                result["playback_start"] = max(0.0, episode["start_time"] - 0.5)
                result["playback_end"] = episode["end_time"] + 0.5
                result["match_level"] = "behavior_event"
                results.append(result)
                break
        if len(results) >= max(1, top_k):
            break
    if not results:
        return None

    positions = ", ".join(f"{int(item['timestamp'] // 60):02d}:{int(item['timestamp'] % 60):02d}" for item in results)
    return {
        "query": query,
        "video_id": video_id,
        "answer": f"{label} 행동이 {positions} 근처에 기록됐어요. 아래 영상에서 확인해 주세요.",
        "results": results,
        "evidence_items": results,
        "behavior_events": matched,
        "used_llm": False,
        "match_level": "behavior_event",
    }
