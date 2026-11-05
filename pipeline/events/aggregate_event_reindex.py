"""Turn a reviewed full-video count result into searchable behavior events."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pipeline.search.aggregate_query import parse_count_action


ACTION_EVENT_TEXT = {
    "gate_climb": ("강아지가 울타리에 올라탐", "울타리"),
    "drink": ("반려동물이 물을 마심", "물"),
    "eat": ("반려동물이 먹이를 먹음", "먹이"),
}


def events_from_count_result(
    result: dict[str, Any], video_id: str, scenes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Only confirmed, independently checked transitions become DB events."""
    if Path(str(result.get("video") or "")).stem != video_id:
        raise ValueError("검증 결과의 영상 ID가 재인덱싱 대상과 다릅니다")
    parsed = parse_count_action(str(result.get("query") or ""))
    if parsed is None or parsed[0] not in ACTION_EVENT_TEXT:
        raise ValueError("현재 검색 경로에서 지원하지 않는 횟수 질문입니다")
    if result.get("track_status") != "model_resolved":
        raise ValueError("개체와 재진입 검증이 완료된 결과만 저장할 수 있습니다")
    confirmed = result.get("confirmed")
    if not isinstance(confirmed, list) or result.get("confirmed_count") != len(confirmed):
        raise ValueError("확정 횟수와 근거 사건 목록이 일치하지 않습니다")
    if not confirmed:
        raise ValueError("확정된 행동이 없어 기존 기록을 교체할 수 없습니다")
    if not scenes:
        raise ValueError("결과에 연결할 인덱싱 프레임이 없습니다")

    action, target = ACTION_EVENT_TEXT[parsed[0]]
    events = []
    for item in confirmed:
        if item.get("model_status") != "confirmed":
            raise ValueError("미확정 후보를 확정 이벤트로 저장할 수 없습니다")
        try:
            start = float(item["start_time"])
            end = float(item["end_time"])
            before = float(item["before_time"])
            after = float(item["after_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("행동 전후 근거 시각이 빠졌습니다") from exc
        if not all(map(math.isfinite, (start, end, before, after))) or not (0 <= start <= end and 0 <= before < after):
            raise ValueError("행동 전후 근거 시각이 올바르지 않습니다")
        nearest = min(scenes, key=lambda scene: abs(float(scene["start_time"]) - after))
        if abs(float(nearest["start_time"]) - after) > 2.5:
            raise ValueError("확정 행동 근처에 인덱싱 프레임이 없습니다")
        key = nearest.get("s3_key")
        if not key:
            raise ValueError("근거 프레임의 저장 경로가 없습니다")
        frame = {
            "s3_key": key,
            "timestamp": float(nearest["start_time"]),
            "frame_id": f"{video_id}__{Path(key).stem}",
            "object_labels": [],
        }
        events.append({
            "video_id": video_id,
            "start_time": min(start, before),
            "end_time": max(end, after),
            "subject": item.get("model_subject_id") or item.get("subject_id") or "반려동물",
            "action": action,
            "target_object": target,
            "summary": f"같은 개체의 행동 전후 상태가 {before:g}초와 {after:g}초에 확인됨",
            "evidence": [
                f"행동 전 상태 {before:g}초, 행동 후 상태 {after:g}초",
                f"후보 재검증: {item.get('reason') or '확인됨'}",
            ],
            "source_frames": [frame],
        })
    return events
