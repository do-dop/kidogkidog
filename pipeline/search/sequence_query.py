"""행동 이벤트를 시간순으로 연결하는 시퀀스 질문 경로."""

import re
from datetime import datetime, timedelta
from typing import Any

MAX_EVENTS = 500
_SEQUENCE_WORDS = ("순서", "시간순", "차례", "흐름", "타임라인", "어떤 행동", "뭐했", "무엇을 했", "그다음", "그 다음", "다음에", "이후", "나서", "먼저")
_FRAME_KEY = re.compile(r"petcam_(\d{8})_(\d{6})_(\d{3})_frame_([0-9.]+)\.jpg$")


def is_sequence_query(query: str) -> bool:
    compact = re.sub(r"\s+", "", query.lower())
    if any(word in compact for word in ("몇번", "몇회", "횟수", "몇개", "얼마나자주")):
        return False
    return any(re.sub(r"\s+", "", word) in compact for word in _SEQUENCE_WORDS)


def _recorded_at(event: dict[str, Any]) -> datetime | None:
    for frame in event.get("source_frames") or []:
        if not isinstance(frame, dict):
            continue
        match = _FRAME_KEY.search(str(frame.get("s3_key") or ""))
        if not match:
            continue
        chunk_start = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        chunk_start += timedelta(minutes=int(match.group(3)))
        event_start = float(event.get("start_time") or 0)
        return chunk_start + timedelta(seconds=event_start)
    return None


def _in_filter(recorded_at: datetime | None, recording_date: str | None, time_range: dict | None) -> bool:
    if recording_date and (recorded_at is None or recorded_at.date().isoformat() != recording_date):
        return False
    if not time_range:
        return True
    start = time_range.get("start_hour")
    end = time_range.get("end_hour")
    if start is None or end is None or (start <= 0 and end >= 24):
        return True
    if recorded_at is None:
        return False
    hour = recorded_at.hour + recorded_at.minute / 60
    return start <= hour < end if start <= end else hour >= start or hour < end


def _description(event: dict[str, Any]) -> str:
    return str(event.get("summary") or event.get("action") or "움직임 감지").strip()


def run_sequence_query(
    query: str,
    video_id: str | None = None,
    top_k: int = 5,
    recording_date: str | None = None,
    time_range: dict | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict:
    """전체 후보를 시간순으로 정렬하고 근거 프레임을 함께 돌려준다."""
    if events is None:
        from db.behavior_events import get_behavior_events_for_aggregation
        events = get_behavior_events_for_aggregation(video_id=video_id)[:MAX_EVENTS + 1]
    truncated = len(events) > MAX_EVENTS
    events = events[:MAX_EVENTS]

    rows = []
    for event in events:
        recorded_at = _recorded_at(event)
        if _in_filter(recorded_at, recording_date, time_range):
            rows.append((recorded_at, event))

    rows.sort(key=lambda row: (
        row[0] is None,
        row[0] or datetime.max,
        str(row[1].get("video_id") or ""),
        float(row[1].get("start_time") or 0),
        int(row[1].get("id") or 0),
    ))
    selected = rows[:top_k]
    results = []
    for recorded_at, event in selected:
        frames = event.get("source_frames") or []
        frame = next((item for item in frames if isinstance(item, dict)), {})
        timestamp = float(frame.get("timestamp") or event.get("start_time") or 0)
        results.append({
            "frame_id": frame.get("frame_id") or f"event-{event.get('id')}",
            "frame_path": frame.get("frame_path"),
            "s3_key": frame.get("s3_key"),
            "timestamp": timestamp,
            "video_id": event.get("video_id"),
            "object_labels": frame.get("object_labels") or "",
            "score": float(event.get("confidence") or 0),
            "source_event_id": event.get("id"),
            "event_start": event.get("start_time"),
            "event_end": event.get("end_time"),
            "representative_timestamp": timestamp,
            "playback_start": max(timestamp - 3, 0),
            "playback_end": timestamp + 5,
        })

    if not rows:
        answer = "선택한 영상·날짜·시간에 분석된 행동 이벤트가 없어 순서를 확인할 수 없습니다."
    else:
        lines = []
        for index, (recorded_at, event) in enumerate(selected, 1):
            time_text = recorded_at.strftime("%m/%d %H:%M:%S") if recorded_at else f"영상 {event.get('start_time')}초"
            lines.append(f"{index}. {time_text} — {_description(event)}")
        scope = f"기록된 행동 이벤트 {len(rows)}개 중 앞의 {len(selected)}개를 시간순으로 보여드릴게요."
        if truncated:
            scope += " 조회 한도에 도달해 이후 이벤트는 포함되지 않았을 수 있습니다."
        answer = scope + "\n" + "\n".join(lines)

    return {
        "query": query,
        "query_type": "sequence",
        "video_id": video_id,
        "answer": answer,
        "results": results,
        "evidence_items": results,
        "behavior_events": [event for _, event in selected],
        "used_llm": False,
        "total_event_count": len(rows),
        "truncated": truncated,
    }
