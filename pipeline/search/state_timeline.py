"""Experimental per-frame state timeline for repeated actions in one visible animal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.search.video_aggregate import (
    sample_frames,
    timeline_windows,
    video_duration,
)


def classify_state_frames(video_path: Path, intent: dict[str, str], ask_json: Any,
                          step: float = 0.25) -> dict[str, Any]:
    """Classify each sampled frame as before, after, or unknown.

    Overlapping window endpoint labels must agree; a disagreement becomes
    unknown rather than silently choosing the most convenient state.
    """
    windows = []
    votes: dict[float, list[dict[str, Any]]] = {}
    for window in timeline_windows(video_duration(video_path), context_seconds=0):
        frames = sample_frames(video_path, window["start"], window["end"], step)
        prompt = (
            "Classify EACH supplied image independently for the SAME target animal. "
            "Return JSON {\"labels\":[{\"index\":integer,\"state\":\"before|after|unknown\","
            "\"subject_visible\":boolean,\"reason\":string}]}. Return exactly "
            "one label for each zero-based image index; never invent an index. "
            "before means that animal visibly satisfies before_state in THIS frame. "
            "after means it visibly satisfies after_state in THIS frame. "
            "unknown means occlusion, unclear paws, or no visible target animal. "
            "Do not infer a frame's state from another timestamp, and do not infer "
            "that a transition happened between frames. If another animal appears, "
            "label the target uncertain instead of switching identity. "
            f"Intent: {json.dumps(intent, ensure_ascii=False)}. "
            f"Frame times by index: {[frame['time'] for frame in frames]}."
        )
        response = ask_json(prompt, frames)
        raw = response.get("labels", []) if isinstance(response, dict) else []
        by_index = {row.get("index"): row for row in raw if isinstance(row, dict)}
        windows.append({"window": window, "frame_times": [f["time"] for f in frames],
                        "response": response})
        for index, frame in enumerate(frames):
            row = by_index.get(index, {})
            state = row.get("state")
            if state not in ("before", "after") or row.get("subject_visible") is not True:
                state = "unknown"
            votes.setdefault(frame["time"], []).append({
                "state": state,
                "reason": str(row.get("reason") or ""),
                "window_index": len(windows) - 1,
            })

    labels = []
    for time, observations in sorted(votes.items()):
        states = {row["state"] for row in observations}
        labels.append({"time": time,
                       "state": states.pop() if len(states) == 1 else "unknown",
                       "observations": observations})
    return {"step": step, "labels": labels, "windows": windows}


def transitions_from_states(labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Count a before→after edge only after a visible before frame.

    An unknown interval never resets an after episode. This reducer assumes
    one visible target animal; multi-animal input needs a separate tracker.
    """
    events = []
    unarmed_after = []
    last_before = None
    armed = False
    in_after = False
    for row in sorted(labels, key=lambda item: item["time"]):
        time, state = row["time"], row["state"]
        if state == "before":
            last_before = time
            armed = True
            in_after = False
        elif state == "after":
            if armed:
                events.append({"before_time": last_before, "after_time": time,
                               "reset_time": last_before if events else None})
                armed = False
            elif not in_after:
                unarmed_after.append(time)
            in_after = True
    return {"events": events, "unarmed_after_times": unarmed_after}
