"""Verify an explicitly requested object color in the actual evidence frames."""

from __future__ import annotations

import base64
import io
import json
import os
import re


_COLOR_WORDS = re.compile(
    r"(?:파란|파랑|푸른|청색|빨간|빨강|붉은|적색|초록|녹색|노란|노랑|"
    r"검은|검정|하얀|흰색|분홍|보라|주황|갈색|회색|"
    r"\b(?:blue|red|green|yellow|black|white|pink|purple|orange|brown|gray|grey)\b)(?:색)?",
    re.IGNORECASE,
)
_MOVEMENT_WORDS = re.compile(
    r"접근|다가|가까|이동|걸어|달려|뛰어|들어|나가|올라|내려|향해|갔|가는|"
    r"approach|walk|run|enter|leave|jump|move",
    re.IGNORECASE,
)
_ACTION_WORDS = re.compile(
    r"접근|다가|가까|이동|걸어|달려|뛰어|들어|나가|올라|내려|향해|갔|가는|"
    r"먹|마시|물고|놀|바라|쳐다|건드|밀|뛰|앉|자고|잠|씻|목욕|"
    r"approach|walk|run|enter|leave|jump|move|eat|drink|play|look|touch|sleep",
    re.IGNORECASE,
)
_COLOR_FAMILIES = {
    "blue": re.compile(r"파란|파랑|푸른|청색|\bblue\b", re.I),
    "red": re.compile(r"빨간|빨강|붉은|적색|\bred\b", re.I),
    "green": re.compile(r"초록|녹색|\bgreen\b", re.I),
    "yellow": re.compile(r"노란|노랑|\byellow\b", re.I),
    "black": re.compile(r"검은|검정|\bblack\b", re.I),
    "white": re.compile(r"하얀|흰색|\bwhite\b", re.I),
    "pink": re.compile(r"분홍|\bpink\b", re.I),
    "purple": re.compile(r"보라|\bpurple\b", re.I),
    "orange": re.compile(r"주황|\borange\b", re.I),
    "brown": re.compile(r"갈색|\bbrown\b", re.I),
    "gray": re.compile(r"회색|\bgr[ae]y\b", re.I),
}
MAX_COLOR_CANDIDATES = 8
MAX_EXPLICIT_COLOR_EVENTS = 4
MAX_EVENT_FRAMES = 3


def asks_about_color(query: str) -> bool:
    return bool(_COLOR_WORDS.search(query or ""))


def asks_about_colored_action(query: str) -> bool:
    return asks_about_color(query) and bool(_ACTION_WORDS.search(query))


def without_color_words(query: str) -> str:
    """Find the action and object first; inspect their color in the image."""
    return re.sub(r"\s+", " ", _COLOR_WORDS.sub("", query or "")).strip()


def include_explicit_color_events(query: str, payload: dict, video_id: str | None,
                                  recording_date=None, time_range=None) -> dict:
    """Bring color-labeled events into the shortlist before visual verification."""
    requested = {name for name, pattern in _COLOR_FAMILIES.items() if pattern.search(query)}
    if len(requested) != 1 or recording_date or time_range:
        return payload
    color = next(iter(requested))
    try:
        from db.behavior_events import get_behavior_events_for_aggregation
        from pipeline.search.event_text_search import event_as_search_result, rank_events_by_text

        events = get_behavior_events_for_aggregation(video_id=video_id)
        matching = [event for event in events
                    if _COLOR_FAMILIES[color].search(" ".join(str(event.get(key) or "")
                                                        for key in ("target_object", "action")))]
        ranked = rank_events_by_text(query, matching)[:MAX_EXPLICIT_COLOR_EVENTS]
        added = [(event, event_as_search_result(event, 1.0)) for event in ranked]
        added = [(event, result) for event, result in added if result and result.get("s3_key")]
    except Exception as exc:
        print(f"색상 명시 이벤트 조회 실패: {exc}", flush=True)
        return payload
    if not added:
        return payload
    payload = dict(payload)
    existing = payload.get("evidence_items") or payload.get("results") or []
    seen = {result.get("source_event_id") for _, result in added}
    payload["results"] = [result for _, result in added] + [
        result for result in existing if result.get("source_event_id") not in seen
    ]
    payload["evidence_items"] = payload["results"]
    payload["behavior_events"] = [event for event, _ in added] + [
        event for event in payload.get("behavior_events") or [] if event.get("id") not in seen
    ]
    return payload


def _recorded_color_conflicts(query: str, event: dict | None) -> bool:
    if not event:
        return False
    requested = {name for name, pattern in _COLOR_FAMILIES.items() if pattern.search(query)}
    if len(requested) != 1:
        return False
    # These fields describe the event's action target. A color found only in
    # unrelated background evidence is deliberately ignored here.
    target = " ".join(str(event.get(key) or "") for key in ("target_object", "action"))
    recorded = {name for name, pattern in _COLOR_FAMILIES.items() if pattern.search(target)}
    return bool(recorded and requested.isdisjoint(recorded))


def _evidence_frames(result: dict, event: dict | None) -> list[dict]:
    frames = [frame for frame in (event or {}).get("source_frames") or []
              if isinstance(frame, dict) and frame.get("s3_key")]
    if not frames:
        frames = [frame for frame in result.get("step_frames") or []
                  if isinstance(frame, dict) and frame.get("s3_key")]
    if not frames and result.get("s3_key"):
        frames = [result]
    if len(frames) <= MAX_EVENT_FRAMES:
        return frames
    return [frames[0], frames[len(frames) // 2], frames[-1]]


def _image_data(key: str) -> str:
    from PIL import Image
    from pipeline.storage.gcs_uploader import download_bytes

    with Image.open(io.BytesIO(download_bytes(key))) as image:
        image = image.convert("RGB")
        image.thumbnail((768, 768))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def verify_result_color(query: str, result: dict, event: dict | None = None) -> bool:
    """Fail closed: only a visible color on the requested object is a match."""
    if not os.getenv("OPENAI_API_KEY"):
        return False
    if _recorded_color_conflicts(query, event):
        return False
    frames = _evidence_frames(result, event)
    if not frames:
        return False
    if _MOVEMENT_WORDS.search(query) and event is None and len(frames) < 2:
        return False
    try:
        from openai import OpenAI

        content = [{
            "type": "input_text",
            "text": (
                "Answer whether these evidence frames support the Korean video question. "
                "The requested color must visibly belong to the requested object, not merely appear elsewhere "
                "in the frame. Distinguish blue from red. If the color is unclear or the object is absent, "
                "answer false. The stored action must also match the action in the question; a matching color "
                "alone is insufficient. The stored action below may support movement, but never use its wording "
                "as proof of color. When the question asks about movement and no stored action is supplied, "
                "a single still image cannot prove it: answer false. Do not infer movement from proximity. "
                "Return only JSON {\"match\":true|false}.\n"
                f"Question: {query}\n"
                f"Stored action: {event.get('action') if event else 'none'}\n"
                f"Stored target: {event.get('target_object') if event else 'none'}"
            ),
        }]
        for frame in frames:
            content.append({"type": "input_image", "image_url": _image_data(frame["s3_key"]), "detail": "high"})
        response = OpenAI(timeout=45.0, max_retries=0).responses.create(
            model=os.getenv("COLOR_VERIFY_MODEL", "gpt-4o-mini"),
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_object"}},
            temperature=0,
        )
        return json.loads(response.output_text).get("match") is True
    except Exception as exc:
        print(f"색상 근거 재검증 실패: {exc}", flush=True)
        return False


def verify_color_results(query: str, payload: dict, top_k: int) -> dict:
    """Keep only visually confirmed color results; never reuse an unverified answer."""
    if not asks_about_color(query):
        return payload
    events = {event.get("id"): event for event in payload.get("behavior_events") or []}
    confirmed = []
    indexed_by_video = {}
    for result in (payload.get("evidence_items") or payload.get("results") or [])[:MAX_COLOR_CANDIDATES]:
        event = events.get(result.get("source_event_id"))
        verification_result = result
        if _MOVEMENT_WORDS.search(query) and event is None:
            video_id = result.get("video_id")
            if video_id not in indexed_by_video:
                try:
                    from pipeline.storage.vector_store import get_indexed_frames
                    indexed_by_video[video_id] = get_indexed_frames(video_id=video_id)
                except Exception:
                    indexed_by_video[video_id] = []
            nearby = _nearby_frames(result, indexed_by_video[video_id])
            if len(nearby) < 2:
                continue
            verification_result = {**result, "step_frames": nearby}
        if verify_result_color(query, verification_result, event):
            confirmed.append(result)
            if len(confirmed) >= max(1, top_k):
                break

    payload = dict(payload)
    # Any aggregate count was computed before the color check and is not a
    # color-qualified count. Do not let the UI display it as one.
    payload.pop("aggregate_result", None)
    if payload.get("query_type") == "aggregate":
        payload["query_type"] = "search"
    payload["results"] = confirmed
    payload["evidence_items"] = confirmed
    selected_ids = {item.get("source_event_id") for item in confirmed}
    payload["behavior_events"] = [event for event in events.values() if event.get("id") in selected_ids]
    payload["used_llm"] = bool(confirmed)
    if confirmed:
        from pipeline.search.video_time import format_frame_time

        moments = ", ".join(format_frame_time(item, float(item.get("timestamp") or 0))
                            for item in confirmed[:3])
        payload["answer"] = f"질문한 색과 대상이 보이는 장면은 {moments} 근처에서 확인할 수 있어요."
        payload["match_level"] = "verified_color"
    else:
        payload["answer"] = "질문한 색의 물체와 행동이 함께 확인되는 장면을 찾지 못했어요."
        payload["match_level"] = "unverified_color"
    return payload


def _nearby_frames(result: dict, indexed: list[dict]) -> list[dict]:
    """Take before/current/after images from the same video chunk."""
    key = str(result.get("s3_key") or "")
    if not key:
        return []
    chunk = key.split("_frame_", 1)[0]
    center = float(result.get("timestamp") or 0)
    nearby = []
    for frame in indexed:
        frame_key = str(frame.get("s3_key") or "")
        if not frame_key or frame_key.split("_frame_", 1)[0] != chunk:
            continue
        try:
            second = float(frame["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(second - center) <= 5:
            nearby.append((second, frame))
    nearby.sort(key=lambda item: item[0])
    before = [item for item in nearby if item[0] < center]
    after = [item for item in nearby if item[0] > center]
    chosen = [before[-1][1]] if before else []
    chosen += [next((item[1] for item in nearby if item[1].get("s3_key") == key), result)]
    if after:
        chosen.append(after[0][1])
    return chosen
