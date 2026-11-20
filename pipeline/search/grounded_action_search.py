"""Resolve common free-form action requests from timestamped behavior events.

This path requires both the requested object/place and an observed action.  It
does not turn a video title or a detected object into proof of an action.
"""

import re

from db.behavior_events import get_behavior_events_for_aggregation
from pipeline.search.aggregate_query import count_episodes
from pipeline.search.event_text_search import event_as_search_result
from pipeline.search.video_time import format_frame_time, frame_video_seconds


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

# These actions are present in event text but are not count-action categories.
# Match the object and action together so a generic pet frame cannot prove them.
_PRECISE_ACTIONS = (
    ("수유", re.compile(r"젖.{0,12}먹|수유"), re.compile(r"젖.{0,12}먹|수유")),
    ("사료 먹기", re.compile(r"(?:사료|밥그릇|그릇).{0,12}먹|먹.{0,12}(?:사료|밥그릇|그릇)"),
     re.compile(r"(?:사료|밥그릇|그릇).{0,12}먹|먹.{0,12}(?:사료|밥그릇|그릇)")),
    ("목욕", re.compile(r"목욕|욕조.{0,8}씻|씻.{0,8}욕조"), re.compile(r"목욕|욕조.{0,8}씻|씻.{0,8}욕조")),
    ("잠자기", re.compile(r"잠|수면|자고|자는"), re.compile(r"잠|수면|자고|자는")),
    ("수건 놀이", re.compile(r"(?:수건|천).{0,12}(?:물|놀)|(?:물|놀).{0,12}(?:수건|천)"),
     re.compile(r"(?:수건|천).{0,12}(?:물|놀)|(?:물|놀).{0,12}(?:수건|천)")),
)


def _source_chunk(event):
    for frame in event.get("source_frames") or []:
        if isinstance(frame, dict) and frame.get("s3_key"):
            return str(frame["s3_key"]).split("_frame_", 1)[0]
    return str(event.get("video_id") or "")


def find_precise_action_scenes(query, video_id, top_k, source_event=None,
                               recording_date=None, time_range=None):
    """Find explicit actions across chunks, keeping episode times chunk-local for playback."""
    if not video_id or recording_date or time_range or not re.search(r"장면|구간|보여|찾아|있어|언제", query):
        return None
    # Before/after questions need sequence anchoring, not an action-only result.
    from pipeline.search.sequence_query import is_sequence_query
    if is_sequence_query(query):
        return None
    intents = [item for item in _PRECISE_ACTIONS if item[1].search(query)]
    if len(intents) != 1:
        return None
    label, _, event_pattern = intents[0]
    if source_event is not None:
        source_text = " ".join(str(source_event.get(key) or "") for key in ("action", "summary", "target_object"))
        if not event_pattern.search(source_text):
            return None

    try:
        events = get_behavior_events_for_aggregation(video_id=video_id)
    except Exception as exc:
        print(f"행동 장면 조회 실패: {exc}", flush=True)
        return None
    matched = []
    for event in events:
        description = " ".join(str(event.get(key) or "") for key in ("action", "summary", "target_object"))
        if event_pattern.search(description) and not _UNCERTAIN.search(description) and event.get("source_frames"):
            matched.append(event)
    if not matched:
        return {
            "query": query,
            "video_id": video_id,
            "answer": f"{label} 장면은 저장된 행동 기록에서 확인하지 못했어요. 원본 영상을 확인해 주세요.",
            "results": [],
            "evidence_items": [],
            "behavior_events": [],
            "used_llm": False,
            "match_level": "unverified_action",
        }

    matched.sort(key=lambda event: (
        frame_video_seconds((event.get("source_frames") or [{}])[0], event.get("start_time") or 0) or 0,
        float(event.get("end_time") or 0),
    ))
    episodes = []
    for event in matched:
        try:
            start = float(event["start_time"])
            end = float(event["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        chunk = _source_chunk(event)
        if episodes and episodes[-1]["chunk"] == chunk and start <= episodes[-1]["end"] + 3:
            episodes[-1]["end"] = max(end, episodes[-1]["end"])
            episodes[-1]["events"].append(event)
        else:
            episodes.append({"chunk": chunk, "start": start, "end": end, "events": [event]})

    results = []
    for episode in episodes:
        event = episode["events"][0]
        result = event_as_search_result(event, float(event.get("confidence") or 0.5))
        if not result or not result.get("s3_key"):
            continue
        result["playback_start"] = max(0.0, episode["start"] - 1)
        result["playback_end"] = episode["end"] + 1
        result["match_level"] = "behavior_event"
        results.append(result)
        if len(results) >= max(1, top_k):
            break
    if not results:
        return None
    positions = ", ".join(format_frame_time(item, float(item["timestamp"])) for item in results[:3])
    if len(results) > 3:
        positions += " 등"
    return {
        "query": query,
        "video_id": video_id,
        "answer": f"{label} 장면은 {positions} 근처에서 확인할 수 있어요.",
        "results": results,
        "evidence_items": results,
        "behavior_events": matched,
        "used_llm": False,
        "match_level": "behavior_event",
    }


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

    positions = ", ".join(format_frame_time(item, float(item["timestamp"])) for item in results)
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
