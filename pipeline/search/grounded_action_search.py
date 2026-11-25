"""Resolve common free-form action requests from timestamped behavior events.

This path requires both the requested object/place and an observed action.  It
does not turn a video title or a detected object into proof of an action.
"""

import json
import os
import re

from db.behavior_events import get_behavior_events_for_aggregation
from pipeline.search.aggregate_query import count_episodes
from pipeline.search.event_text_search import event_as_search_result
from pipeline.search.video_time import format_frame_time, frame_video_seconds
from pipeline.storage.capture_time import trusted_capture_time
from pipeline.search.search_policy import contradicts_action, diverse_video_candidates
from pipeline.search.color_grounding import asks_about_color, asks_about_colored_action, without_color_words


_UNCERTAIN = re.compile(r"않|못|없|싶|추정|하려|려고")
_COMPOUND_ACTION = re.compile(r"(?P<context>.+)(?:하고|(?<!하)고)\s+(?P<focus>[^?]{2,}\??)$")
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

def _source_chunk(event):
    for frame in event.get("source_frames") or []:
        if isinstance(frame, dict) and frame.get("s3_key"):
            return str(frame["s3_key"]).split("_frame_", 1)[0]
    return str(event.get("video_id") or "")


def _supports_explicit_constraints(query, event):
    """Reject contradictions the model might overlook in species or direction."""
    if contradicts_action(query, event):
        return False
    description = " ".join(str(event.get(key) or "") for key in ("subject", "action", "summary"))
    labels = set()
    for frame in event.get("source_frames") or []:
        raw = frame.get("object_labels") or []
        labels.update(str(item).lower() for item in (raw if isinstance(raw, list) else str(raw).split(",")))
    for korean, label, opposite in (("고양이", "cat", "강아지"), ("강아지", "dog", "고양이")):
        if korean in query and (opposite in description or (labels and label not in labels)):
            return False
    for direction in ("안으로", "밖으로"):
        if direction not in query:
            continue
        if direction not in description:
            return False
        # Leaving the camera frame does not prove leaving a door or pool.
        place = re.search(r"([가-힣]{1,12})\s+" + direction, query)
        if place and place.group(1) not in description:
            return False
    return True


def _event_in_datetime(event, recording_date, time_range):
    if not recording_date and not time_range:
        return True
    start_hour, end_hour = None, None
    if time_range:
        try:
            start_hour = int(time_range.get("start_hour"))
            end_hour = int(time_range.get("end_hour"))
        except (TypeError, ValueError):
            pass
    if start_hour is not None and start_hour <= 0 and end_hour >= 24:
        start_hour = end_hour = None
    if not recording_date and start_hour is None:
        return True
    for frame in event.get("source_frames") or []:
        captured = trusted_capture_time({
            "source_type": frame.get("capture_time_source") or frame.get("source_type"),
            "captured_at": frame.get("recorded_at") or frame.get("captured_at"),
        }) if isinstance(frame, dict) else None
        if captured is None:
            continue
        if recording_date and captured.date().isoformat() != recording_date:
            continue
        if start_hour is None:
            return True
        hour = captured.hour + captured.minute / 60 + captured.second / 3600
        if (start_hour <= end_hour and start_hour <= hour < end_hour) or (
            start_hour > end_hour and (hour >= start_hour or hour < end_hour)
        ):
            return True
    return False


def _compound_action_parts(query):
    """Separate the earlier context from the action the user asks to see."""
    match = _COMPOUND_ACTION.search(query.strip())
    return (match.group("context"), match.group("focus")) if match else (None, query)


def _preceded_by_context(event, context_events):
    try:
        start = float(event["start_time"])
    except (KeyError, TypeError, ValueError):
        return False
    for previous in context_events:
        try:
            if (previous.get("video_id") == event.get("video_id")
                    and float(previous["end_time"]) <= start
                    and previous.get("id") != event.get("id")):
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _select_matching_events(query, events):
    """Ask one general matcher to compare the requested action with recorded events.

    The model only returns row numbers. Times, descriptions and playback frames
    always come from the database, never from generated text.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    rows = [
        {"row": index, "subject": event.get("subject"), "action": event.get("action"),
         "target": event.get("target_object"), "summary": event.get("summary")}
        for index, event in enumerate(events)
    ]
    prompt = (
        "영상 질문과 관찰된 행동 이벤트를 비교하세요. 행동과 질문의 세부 조건이 모두 명확히 맞으면 rows, "
        "핵심 행동은 같지만 주체의 종류·관계 같은 세부 조건이 기록에 없어 확정할 수 없으면 related_rows에 넣으세요. "
        "질문이 'A하고 B했어?'처럼 여러 행동을 말하면 B가 답으로 찾는 핵심 행동이고 A는 선행 맥락입니다. "
        "선행 행동 A의 row는 답의 rows나 related_rows에 넣지 말고, B의 row만 고르세요. "
        "선행 맥락이 실제로 B보다 앞에 있는지도 시간 순서로 확인하세요. "
        "행동의 대상과 방향을 구분하세요(예: 들어가기/나오기, 젖 먹기/사료 먹기). "
        "동물의 존재만으로 행동을 입증하지 마세요. 추정·부정·의도 표현은 확정 근거가 아닙니다. "
        "서로 다른 행동·대상·방향은 related_rows에도 넣지 마세요. 반복된 같은 행동은 해당 row를 모두 고르세요. "
        "질문이 단순 객체 존재/일반 이동이거나 핵심 행동 일치가 없으면 두 배열을 모두 비우세요. "
        "오직 JSON {\"rows\":[정수,...],\"related_rows\":[정수,...]} 형식으로 답하세요.\n"
        f"질문: {query}\n이벤트: {json.dumps(rows, ensure_ascii=False)}"
    )
    response = OpenAI().responses.create(
        model=os.getenv("ACTION_EVENT_MATCH_MODEL", "gpt-4o-mini"),
        input=prompt,
        text={"format": {"type": "json_object"}},
        temperature=0,
    )
    parsed = json.loads(response.output_text)
    def selected_rows(key):
        values = parsed.get(key, [])
        if not isinstance(values, list):
            return []
        indices = {row for row in values if type(row) is int and 0 <= row < len(events)}
        return [event for index, event in enumerate(events) if index in indices]
    exact = selected_rows("rows")
    exact_ids = {event.get("id") for event in exact}
    related = [event for event in selected_rows("related_rows") if event.get("id") not in exact_ids]
    return {"exact": exact, "related": related}


def find_action_event_scenes(query, video_id, top_k, source_event=None,
                             recording_date=None, time_range=None):
    """Find any recorded action across chunks, with event-backed playback times."""
    if asks_about_color(query) and not asks_about_colored_action(query):
        return None
    asks_whether_action_happened = bool(re.search(r"(?:았|었|했)(?:어|니|나요|습니까)[?\s]*$", query))
    if not re.search(r"장면|구간|보여|찾아|있어|언제", query) and not asks_about_colored_action(query) and not asks_whether_action_happened:
        return None
    from pipeline.search.object_presence import _presence_subject
    if _presence_subject(query):
        return None
    # Before/after questions need sequence anchoring, not an action-only result.
    from pipeline.search.sequence_query import is_sequence_query
    if is_sequence_query(query):
        return None
    try:
        events = get_behavior_events_for_aggregation(video_id=video_id)
    except Exception as exc:
        print(f"행동 장면 조회 실패: {exc}", flush=True)
        return None
    candidates = [event for event in events if event.get("source_frames")
                  and _event_in_datetime(event, recording_date, time_range)
                  and not _UNCERTAIN.search(
        " ".join(str(event.get(key) or "") for key in ("action", "summary", "target_object"))
    )]
    if not candidates:
        return None
    # A missing color in the stored event must not hide a matching action.
    # The actual color is checked against the image before the result is shown.
    action_query = without_color_words(query) if asks_about_color(query) else query
    context_query, action_query = _compound_action_parts(action_query)
    context_candidates = candidates
    from pipeline.search.action_event_index import semantic_event_candidates
    ranked = semantic_event_candidates(action_query, candidates, video_id=video_id)
    if ranked is not None:
        if video_id:
            candidates = ranked
        else:
            # Semantic top-N can be filled by one long video. Reserve a text-
            # ranked lead from other videos before the final meaning check.
            from pipeline.search.event_text_search import rank_events_by_text
            ranked_ids = {event.get("id") for event in ranked}
            remaining = [event for event in rank_events_by_text(action_query, candidates)
                         if event.get("id") not in ranked_ids]
            candidates = diverse_video_candidates(ranked + remaining, 120)
    elif len(candidates) > 120:
        # Existing recordings without a complete index keep the prior path.
        from pipeline.search.event_text_search import rank_events_by_text
        ranked = rank_events_by_text(action_query, candidates)
        candidates = ranked[:120] if video_id else diverse_video_candidates(ranked, 120)
    try:
        selection = _select_matching_events(action_query, candidates)
    except Exception as exc:
        print(f"행동 이벤트 의미 비교 실패: {exc}", flush=True)
        return None
    if selection is None:
        return None
    # Older callers and tests may still provide a plain list of exact matches.
    exact = selection.get("exact", []) if isinstance(selection, dict) else selection
    related = selection.get("related", []) if isinstance(selection, dict) else []
    matched = [event for event in exact if _supports_explicit_constraints(query, event)]
    uncertain = not matched
    if uncertain:
        matched = [event for event in related if _supports_explicit_constraints(query, event)]
    if matched and context_query:
        context_ranked = semantic_event_candidates(context_query, context_candidates, video_id=video_id)
        if not context_ranked:
            from pipeline.search.event_text_search import rank_events_by_text
            context_ranked = (rank_events_by_text(context_query, context_candidates)[:120]
                              if len(context_candidates) > 120 else context_candidates)
        try:
            context_selection = _select_matching_events(context_query, context_ranked)
            context_events = context_selection.get("exact", []) if isinstance(context_selection, dict) else (context_selection or [])
        except Exception as exc:
            print(f"선행 행동 의미 비교 실패: {exc}", flush=True)
            context_events = []
        matched = [event for event in matched if _preceded_by_context(event, context_events)]
    if not matched:
        return {
            "query": query,
            "video_id": video_id,
            "answer": "질문한 행동은 저장된 행동 기록에서 확인하지 못했어요. 원본 영상을 확인해 주세요.",
            "results": [],
            "evidence_items": [],
            "behavior_events": [],
            "used_llm": False,
            "match_level": "unverified_action",
        }

    semantic_order = {id(event): index for index, event in enumerate(matched)}
    # Merge in chronological order even when the global candidates were ranked
    # by relevance. Otherwise a later episode can swallow an earlier one.
    matched.sort(key=lambda event: (
            str(event.get("video_id") or ""), _source_chunk(event),
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

    if not video_id:
        episodes.sort(key=lambda episode: min(semantic_order[id(event)] for event in episode["events"]))
        selected_videos = set()
        for episode in episodes:
            candidate_video = episode["events"][0].get("video_id")
            if candidate_video not in selected_videos and len(selected_videos) < max(1, top_k):
                selected_videos.add(candidate_video)
        episodes = [episode for episode in episodes if episode["events"][0].get("video_id") in selected_videos]

    results = []
    scenes_per_video = {}
    for episode in episodes:
        event = episode["events"][0]
        if not video_id and scenes_per_video.get(event.get("video_id"), 0) >= max(1, top_k):
            continue
        result = event_as_search_result(event, float(event.get("confidence") or 0.5))
        if not result or not result.get("s3_key"):
            continue
        # In "A then B" questions, seek to B's onset. Its thumbnail may be
        # sampled later, and the earlier A must never become a selectable hit.
        if context_query:
            result["timestamp"] = episode["start"]
            result["representative_timestamp"] = episode["start"]
        result["playback_start"] = max(0.0, episode["start"] - 1)
        result["playback_end"] = episode["end"] + 1
        result["match_level"] = "related_event" if uncertain else "behavior_event"
        results.append(result)
        scenes_per_video[event.get("video_id")] = scenes_per_video.get(event.get("video_id"), 0) + 1
        if video_id and len(results) >= max(1, top_k):
            break
    if not results:
        return None
    positions = ", ".join(
        (f"{item['video_id']} " if not video_id else "") + format_frame_time(item, float(item["timestamp"]))
        for item in results[:3]
    )
    if len(results) > 3:
        positions += " 등"
    return {
        "query": query,
        "video_id": video_id,
        "answer": (
            f"관련 행동 기록은 {positions} 근처에 있지만 질문의 세부 조건까지는 확인하지 못했어요."
            if uncertain else f"질문한 행동은 {positions} 근처에서 확인할 수 있어요."
        ),
        "results": results,
        "evidence_items": results,
        "behavior_events": matched,
        "used_llm": False,
        "match_level": "related_event" if uncertain else "behavior_event",
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
