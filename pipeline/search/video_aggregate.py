"""Question-directed, evidence-backed counting from a complete local video.

This is an offline evaluation path. It does not write events to the database or
replace the deployed /query route. The vision model proposes observations;
the reducer never turns missing before/after or reset evidence into a count.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any


def parse_count_intent(query: str, ask_json: Any) -> dict[str, str]:
    """Translate one count question into a reusable state-transition definition."""
    result = ask_json(
        "Read the Korean counting question. Return JSON with keys subject, action, "
        "target, before_state, after_state, exclusion. Define a transition of ONE "
        "individual animal, even when the question uses a plural subject. The before "
        "and after states must describe that SAME animal's body location or pose; do "
        "not describe the total number of animals in the scene. Example: 'dogs climb "
        "onto sofa' => before_state='one dog's whole body on floor', "
        "after_state='that same dog's whole body on sofa seat', exclusion='near sofa "
        "or front paws only'. For a fence climb, front paws and upper body "
        "supported on the fence can qualify; do not require all four paws on top "
        "unless the question explicitly says to cross or stand on top. For a pool "
        "entry, body entering water qualifies; do not require full submersion. "
        "For front-paw raising, use resting paws down -> paws lifted toward the "
        "target, even if hind paws remain on the ground when the question allows it. "
        "Use observable states, not video-specific rules. "
        "A question asking how many times one action CATEGORY repeats is valid. "
        "Reject only questions with no visually observable action category or "
        "multiple unrelated requested categories. Question: " + query,
        [],
    )
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError(f"지원하지 않는 횟수 질문: {query}")
    keys = ("subject", "action", "target", "before_state", "after_state", "exclusion")
    if any(not isinstance(result.get(key), str) or not result[key].strip() for key in keys):
        raise ValueError("질문 해석에 상태 전환 정보가 빠졌습니다")
    return {key: result[key].strip() for key in keys}


def video_duration(video_path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or fps <= 0 or count <= 0:
            raise ValueError(f"영상을 읽을 수 없습니다: {video_path}")
        return count / fps
    finally:
        cap.release()


def sample_frames(video_path: Path, start: float, end: float, step: float) -> list[dict[str, Any]]:
    """Read timestamped JPEGs from the original video, including both endpoints."""
    import cv2

    if start < 0 or end < start or step <= 0:
        raise ValueError("잘못된 프레임 구간")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    frames = []
    try:
        times = [start + i * step for i in range(int(math.floor((end - start) / step)) + 1)]
        if not times or end - times[-1] > 0.1 * step:
            times.append(end)
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            if max(height, width) > 1024:
                scale = 1024 / max(height, width)
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                frames.append({"time": round(t, 3), "jpeg": jpg.tobytes()})
    finally:
        cap.release()
    return frames


def timeline_windows(duration: float, core_seconds: float = 4.0, context_seconds: float = 0.75) -> list[dict[str, float]]:
    if duration <= 0 or core_seconds <= 0:
        raise ValueError("잘못된 영상 길이 또는 시간창")
    windows = []
    for i in range(math.ceil(duration / core_seconds)):
        core_start = i * core_seconds
        core_end = min(duration, (i + 1) * core_seconds)
        windows.append({
            "core_start": core_start,
            "core_end": core_end,
            "start": max(0.0, core_start - context_seconds),
            "end": min(duration, core_end + context_seconds),
        })
    return windows


def _finite_time(value: Any, low: float, high: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and low <= number <= high else None


def collect_candidates(video_path: Path, intent: dict[str, str], ask_json: Any,
                       coarse_step: float = 0.5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan every part of the video; retain all plausible transitions for review."""
    duration = video_duration(video_path)
    windows = timeline_windows(duration)
    candidates = []
    scan_log = []
    for window in windows:
        frames = sample_frames(video_path, window["start"], window["end"], coarse_step)
        prompt = (
            "These are timestamped frames from one video window. Find ALL possible "
            "transitions for this intent, including uncertain ones. Return JSON "
            "{\"candidates\":[{\"time\":number,\"subject\":string,\"reason\":string}]}. "
            "One candidate per individual, per possible entry into after_state. "
            "Inspect the whole frame carefully; small or dark animals may blend into "
            "furniture or the edge of the image. "
            "Continue searching after the first animal's entry: another animal may "
            "enter the same target later in this window. "
            "Require a frame showing the individual in after_state or visibly midway "
            "through the transition. Reject mere approach, movement nearby, speculation "
            "that an action might happen, and continued occupancy. If only approach "
            "is visible, return no candidate. Return at most five candidates per "
            "window. Use appearance in subject, e.g. black dog or white dog. "
            "A brief action may fall between sampled frames; only include it if both "
            "surrounding frames support a transition. Do not count. "
            f"Intent: {json.dumps(intent, ensure_ascii=False)}. "
            f"Only report candidate times in [{window['core_start']:.2f}, {window['core_end']:.2f}) "
            "except the last window, which may include its endpoint."
        )
        response = ask_json(prompt, frames)
        raw = response.get("candidates", []) if isinstance(response, dict) else []
        if not isinstance(raw, list):
            raise ValueError("후보 응답이 목록이 아닙니다")
        scan_log.append({"window": window, "response": response, "frame_times": [f["time"] for f in frames]})
        for row in raw:
            if not isinstance(row, dict):
                continue
            # The model can correctly spot an event in an overlapping context
            # frame even when it ignores the requested core-only instruction.
            # Keep it, then deduplicate identical candidate proposals below.
            t = _finite_time(row.get("time"), window["start"], window["end"])
            if t is not None:
                candidates.append({"time": t, "subject": str(row.get("subject") or "unknown"),
                                   "reason": str(row.get("reason") or ""), "window": len(scan_log) - 1})
    unique = {}
    for row in candidates:
        unique.setdefault((round(row["time"], 2), row["subject"].strip().lower()), row)
    return sorted(unique.values(), key=lambda row: row["time"]), scan_log


def verify_candidates(video_path: Path, intent: dict[str, str], candidates: list[dict[str, Any]],
                      ask_json: Any, fine_step: float = 0.25) -> list[dict[str, Any]]:
    """Ask for before/after/reset evidence in dense frames near each candidate."""
    duration = video_duration(video_path)
    verified = []
    for candidate in candidates:
        start = max(0.0, candidate["time"] - 1.25)
        end = min(duration, candidate["time"] + 1.5)
        frames = sample_frames(video_path, start, end, fine_step)
        prompt = (
            "Verify ONE possible state transition from these timestamped video frames. "
            "Return JSON with subject_id (stable appearance-based description; use "
            "'unknown' only if two animals cannot be distinguished; use e.g. "
            "'black dog' or 'white dog' if visibly different), start_time, end_time, "
            "before_time, after_time, reset_time, status, reason. Times must be numeric "
            "or null, within supplied frames. status is confirmed, rejected, or uncertain. "
            "A confirmed transition requires the SAME animal visibly in before_state "
            "at before_time, then in after_state at after_time. Inspect the ENTIRE "
            "provided clip: the candidate timestamp is approximate and the action "
            "may finish later. Treat after_state literally. If it requires the whole "
            "body on a target, front paws on it with rear paws still on the floor "
            "does NOT qualify. Name the earliest frame that proves full completion. "
            "If no animal reaches "
            "after_state, or only approaches, say rejected. If a transition may have "
            "happened but is hidden by a cut/occlusion, boundary, or identity confusion, "
            "say uncertain. Do not infer body location from a scene-level animal count. "
            "reset_time is a visible return to before_state after a preceding target-state "
            "episode and before this transition; null if not visible here. Do not claim "
            "a reset from time spacing alone. For a first entry, before_time suffices. "
            f"Intent: {json.dumps(intent, ensure_ascii=False)}. "
            f"Candidate: {json.dumps(candidate, ensure_ascii=False)}."
        )
        response = ask_json(prompt, frames)
        verified.append({"candidate": candidate, "frame_times": [f["time"] for f in frames],
                         "observation": response, "clip": [start, end]})
    return verified


def check_claimed_evidence(video_path: Path, intent: dict[str, str],
                           verified: list[dict[str, Any]], ask_json: Any) -> list[dict[str, Any]]:
    """Independently challenge every claimed transition using its exact keyframes."""
    duration = video_duration(video_path)
    for item in verified:
        observation = item.get("observation") or {}
        if observation.get("status") != "confirmed":
            continue
        low, high = item["clip"]
        before = _finite_time(observation.get("before_time"), low, high)
        after = _finite_time(observation.get("after_time"), low, high)
        if before is None or after is None or before >= after:
            observation["status"] = "uncertain"
            observation["evidence_check_reason"] = "전후 근거 시각이 유효하지 않음"
            continue
        frames = sample_frames(video_path, before, min(duration, after), after - before)
        prompt = (
            "Independently challenge a proposed event using ONLY these two keyframes. "
            "Return JSON with status (confirmed/rejected/uncertain), "
            "before_state_visible (boolean), after_state_visible (boolean), "
            "same_subject_visible (boolean), subject_id, reason. "
            "Confirmed requires the SAME individual visibly in before_state at the "
            "first frame and in after_state at the second. If the first frame is "
            "already after_state, or the second fails after_state, reject. If identity "
            "cannot be matched, say uncertain. Do not trust the earlier model claim; "
            "inspect the pixels. For whole-body targets, front paws alone do not count. "
            f"Intent: {json.dumps(intent, ensure_ascii=False)}. "
            f"Claimed subject: {observation.get('subject_id')}. "
            f"Times: {before:.3f}s, {after:.3f}s."
        )
        check = ask_json(prompt, frames)
        item["evidence_check"] = check
        if (not isinstance(check, dict) or check.get("status") != "confirmed"
                or check.get("before_state_visible") is not True
                or check.get("after_state_visible") is not True
                or check.get("same_subject_visible") is not True):
            observation["status"] = "rejected" if isinstance(check, dict) and check.get("status") == "rejected" else "uncertain"
            observation["evidence_check_reason"] = check.get("reason") if isinstance(check, dict) else "근거 재검증 실패"
    return verified


def repair_failed_claims(video_path: Path, intent: dict[str, str],
                         verified: list[dict[str, Any]], ask_json: Any) -> list[dict[str, Any]]:
    """One wider-window retry when the chosen evidence times, not the event, failed."""
    duration = video_duration(video_path)
    for item in verified:
        failed_check = item.get("evidence_check")
        if (not failed_check or item.get("repair")
                or item.get("observation", {}).get("status") == "confirmed"):
            continue
        center = item["candidate"]["time"]
        start, end = max(0.0, center - 2.5), min(duration, center + 2.5)
        frames = sample_frames(video_path, start, end, 0.25)
        prompt = (
            "A previous candidate's claimed before/after KEYFRAMES failed an "
            "independent visual check. Search this wider clip for the actual "
            "transition of ONE animal near the candidate. Return JSON with "
            "subject_id, start_time, end_time, before_time, after_time, reset_time, "
            "status (confirmed/rejected/uncertain), reason. All times numeric or null. "
            "Confirmed requires the same animal wholly in before_state at a visible "
            "earlier frame and wholly in after_state at a later frame. Do not repeat "
            "the invalid pair. If no such pair is visible, reject or leave uncertain. "
            f"Intent: {json.dumps(intent, ensure_ascii=False)}. "
            f"Candidate time: {center:.3f}s. Failed check: "
            f"{json.dumps(failed_check, ensure_ascii=False)}."
        )
        repair = ask_json(prompt, frames)
        item["repair"] = repair
        if not isinstance(repair, dict) or repair.get("status") != "confirmed":
            continue
        before = _finite_time(repair.get("before_time"), start, end)
        after = _finite_time(repair.get("after_time"), start, end)
        if before is None or after is None or before >= after:
            continue
        pair = sample_frames(video_path, before, after, after - before)
        check_prompt = (
            "Independently challenge a proposed event using ONLY these two keyframes. "
            "Return JSON with status (confirmed/rejected/uncertain), "
            "before_state_visible (boolean), after_state_visible (boolean), "
            "same_subject_visible (boolean), subject_id, reason. Confirm only if "
            "the same animal is visibly in before_state then after_state. If the "
            "first frame is already after_state, reject. For whole-body targets, "
            "front paws alone do not count. Do not trust the earlier claim. "
            f"Intent: {json.dumps(intent, ensure_ascii=False)}. "
            f"Claimed subject: {repair.get('subject_id')}. Times: {before:.3f}s, {after:.3f}s."
        )
        check = ask_json(check_prompt, pair)
        item["repair_evidence_check"] = check
        if (isinstance(check, dict) and check.get("status") == "confirmed"
                and check.get("before_state_visible") is True
                and check.get("after_state_visible") is True
                and check.get("same_subject_visible") is True):
            item["observation"] = repair
            item["clip"] = [start, end]
    return verified


def resolve_subject_tracks(video_path: Path, verified: list[dict[str, Any]],
                           ask_json: Any) -> list[dict[str, Any]]:
    """Resolve model descriptions to stable individuals across candidate windows."""
    relevant = [(i, item) for i, item in enumerate(verified)
                if item.get("observation", {}).get("status") != "rejected"]
    if len(relevant) == 1:
        relevant[0][1]["observation"]["track_id"] = "animal_1"
        return verified
    if not relevant:
        return verified
    frames = []
    descriptions = []
    for index, item in relevant:
        observation = item["observation"]
        t = observation.get("after_time") or item["candidate"]["time"]
        at = sample_frames(video_path, float(t), float(t), 0.25)
        if at:
            frames.extend(at)
            descriptions.append({"index": index, "time": float(t),
                                 "description": observation.get("subject_id"),
                                 "candidate_time": item["candidate"]["time"]})
    if not descriptions:
        return verified
    prompt = (
        "These images are candidate snapshots from ONE video, in the exact order "
        "listed below. Assign a stable track_id for the target animal of each "
        "candidate. The same animal may have different text descriptions across "
        "windows (e.g. black cat vs black-and-white cat); do not rely on names "
        "alone. Different animals may look similar; use spatial and temporal "
        "context where visible. If identity cannot be resolved from these "
        "snapshots, use track_id='unknown'. Return JSON "
        "{\"assignments\":[{\"index\":integer,\"track_id\":string,\"certain\":boolean,\"reason\":string}]}. "
        "Use animal_1, animal_2, etc. only for distinct visible animals. "
        f"Candidates in image order: {json.dumps(descriptions, ensure_ascii=False)}"
    )
    response = ask_json(prompt, frames)
    assignments = response.get("assignments", []) if isinstance(response, dict) else []
    by_index = {row.get("index"): row for row in assignments if isinstance(row, dict)}
    for index, item in relevant:
        assignment = by_index.get(index, {})
        track = str(assignment.get("track_id") or "unknown").strip().lower()
        if assignment.get("certain") is not True or track == "unknown":
            track = "unknown"
        item["track_resolution"] = assignment
        item["observation"]["track_id"] = track
    return verified


def reduce_transitions(verified: list[dict[str, Any]]) -> dict[str, Any]:
    """Count only proven state entries; never merge distinct animals by time gap."""
    confirmed, uncertain, rejected, duplicates = [], [], [], []
    last_by_subject: dict[str, dict[str, Any]] = {}
    for item in sorted(verified, key=lambda x: x["candidate"]["time"]):
        row = item.get("observation") or {}
        if not isinstance(row, dict):
            row = {}
        start, end = item["clip"]
        before = _finite_time(row.get("before_time"), start, end)
        after = _finite_time(row.get("after_time"), start, end)
        subject = str(row.get("track_id", row.get("subject_id")) or "unknown").strip().lower()
        status = row.get("status")
        record = {"candidate_time": item["candidate"]["time"], "subject_id": subject,
                  "start_time": _finite_time(row.get("start_time"), start, end),
                  "end_time": _finite_time(row.get("end_time"), start, end),
                  "before_time": before, "after_time": after,
                  "reset_time": _finite_time(row.get("reset_time"), start, end),
                  "reason": str(row.get("reason") or ""), "model_status": status,
                  "model_subject_id": row.get("subject_id")}
        if status == "rejected":
            rejected.append(record)
            continue
        if status != "confirmed" or subject in ("", "unknown", "ambiguous") or before is None or after is None or before >= after:
            record["review_reason"] = "전후 상태 또는 개체를 확인하지 못함"
            uncertain.append(record)
            continue
        previous = last_by_subject.get(subject)
        if previous:
            if record["start_time"] is not None and previous["end_time"] is not None and record["start_time"] <= previous["end_time"]:
                record["review_reason"] = "같은 개체의 기존 구간과 겹침"
                duplicates.append(record)
                continue
            reset = record["reset_time"]
            if reset is None or previous["end_time"] is None or not previous["end_time"] < reset < after:
                record["review_reason"] = "같은 개체의 새로운 진입 전 이탈/리셋 근거 없음"
                uncertain.append(record)
                continue
        confirmed.append(record)
        last_by_subject[subject] = record
    return {"confirmed_count": len(confirmed), "uncertain_count": len(uncertain),
            "confirmed": confirmed, "uncertain": uncertain, "rejected": rejected,
            "duplicates": duplicates}


class OpenAIJsonVision:
    """Small Responses API adapter; caller controls model and persisted outputs."""

    def __init__(self, model: str = "gpt-4o", client: Any = None):
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client
        self.model = model

    def __call__(self, prompt: str, frames: list[dict[str, Any]]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for frame in frames:
            content.append({"type": "input_text", "text": f"t={frame['time']:.3f}s"})
            encoded = base64.b64encode(frame["jpeg"]).decode("ascii")
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}", "detail": "high"})
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_object"}},
        )
        return json.loads(response.output_text)


def analyze_video_count(video_path: Path, query: str, ask_json: Any,
                        coarse_step: float = 0.5, fine_step: float = 0.25,
                        resolve_tracks: bool = True) -> dict[str, Any]:
    intent = parse_count_intent(query, ask_json)
    candidates, scan_log = collect_candidates(video_path, intent, ask_json, coarse_step)
    verified = verify_candidates(video_path, intent, candidates, ask_json, fine_step)
    verified = check_claimed_evidence(video_path, intent, verified, ask_json)
    verified = repair_failed_claims(video_path, intent, verified, ask_json)
    track_status = "skipped"
    track_error = None
    if resolve_tracks:
        try:
            verified = resolve_subject_tracks(video_path, verified, ask_json)
            track_status = "model_resolved"
        except Exception as exc:
            # Preserve all expensive scan and evidence responses. A failed ID
            # resolution must be visible, never silently presented as exact.
            track_status = "unresolved_error"
            track_error = f"{type(exc).__name__}: {exc}"
    result = reduce_transitions(verified)
    result.update({"video": str(video_path), "query": query, "intent": intent,
                   "candidates": candidates, "scan_log": scan_log, "verifications": verified,
                   "models": getattr(ask_json, "model", None),
                   "track_status": track_status, "track_error": track_error,
                   "coverage_note": "All timeline windows scanned at coarse_step; brief actions may still be missed."})
    return result
