"""Conservative text matching for timestamped scene descriptions.

Visual frame retrieval remains the fallback. An event is promoted only when
its description covers enough of a specific query and clearly beats the next
event; generic questions should keep the existing frame ranking.
"""

from __future__ import annotations

from collections import Counter
from math import log
import re
from typing import Any


_QUERY_ENDINGS = re.compile(r"(?:장면\s*)?(?:보여\s*줘|보여\s*주세요|찾아\s*줘|찾아\s*주세요)[.!?\s]*$")


def _grams(value: str) -> set[str]:
    value = re.sub(r"[^가-힣a-z0-9]", "", value.lower())
    return {
        value[index:index + length]
        for length in (2, 3)
        for index in range(max(0, len(value) - length + 1))
    }


def _description(event: dict[str, Any]) -> str:
    evidence = event.get("evidence") or []
    if isinstance(evidence, list):
        evidence = " ".join(str(item) for item in evidence)
    return " ".join(str(part or "") for part in (
        event.get("subject"), event.get("action"), event.get("summary"), evidence,
    ))


def best_specific_event(query: str, events: list[dict[str, Any]]) -> tuple[dict[str, Any], float] | None:
    """Return a distinct text match, or None when evidence is ambiguous."""
    query_grams = _grams(_QUERY_ENDINGS.sub("", query.strip()))
    if len(query_grams) < 5 or not events:
        return None

    event_grams = [_grams(_description(event)) for event in events]
    document_frequency = Counter(gram for grams in event_grams for gram in grams)
    corpus_size = len(event_grams)

    def weight(gram: str) -> float:
        return log((corpus_size + 1) / (document_frequency[gram] + 1)) + 1

    denominator = sum(weight(gram) for gram in query_grams)
    ranked = sorted(
        (
            (sum(weight(gram) for gram in query_grams & grams) / denominator, index)
            for index, grams in enumerate(event_grams)
        ),
        reverse=True,
    )
    best_score, best_index = ranked[0]
    next_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score < 0.35 or best_score - next_score < 0.07:
        return None
    return events[best_index], best_score


def rank_events_by_text(query: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shortlist long timelines by weighted text overlap without fixing action names."""
    query_grams = _grams(_QUERY_ENDINGS.sub("", query.strip()))
    event_grams = [_grams(_description(event)) for event in events]
    frequency = Counter(gram for grams in event_grams for gram in grams)
    size = len(events)
    return [event for _, event in sorted(
        ((sum(log((size + 1) / (frequency[gram] + 1)) + 1
              for gram in query_grams & grams), event) for event, grams in zip(events, event_grams)),
        key=lambda item: item[0], reverse=True,
    )]


def event_as_search_result(event: dict[str, Any], score: float) -> dict[str, Any] | None:
    try:
        start = float(event["start_time"])
        end = float(event["end_time"])
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start or not event.get("video_id"):
        return None

    frames = [frame for frame in event.get("source_frames") or [] if isinstance(frame, dict)]
    center = (start + end) / 2
    def distance(frame: dict[str, Any]) -> float:
        try:
            timestamp = float(frame["timestamp"])
        except (KeyError, TypeError, ValueError):
            return float("inf")
        return abs(timestamp - center)

    source = min(frames, key=distance) if frames else {}
    return {
        "video_id": event["video_id"],
        "frame_id": source.get("frame_id") or f"event-{event.get('id', 'match')}",
        "frame_path": source.get("frame_path"),
        "s3_key": source.get("s3_key"),
        "timestamp": center,
        "representative_timestamp": center,
        "scene_start": start,
        "scene_end": end,
        "playback_start": max(0.0, start - 0.5),
        "playback_end": end + 0.5,
        "object_labels": source.get("object_labels") or "",
        "summary": event.get("summary") or event.get("action"),
        "source_event_id": event.get("id"),
        "score": score,
    }
