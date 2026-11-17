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
    "water_enter": ("반려동물이 물속으로 들어감", "물"),
    "front_paw_raise": ("반려동물이 두 앞발을 들어 올림", "앞발"),
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
    point_transition = parsed[0] in {"water_enter", "front_paw_raise"}
    query = str(result.get("query") or "")
    if parsed[0] == "water_enter":
        target = "분수" if "분수" in query else "수영장" if "수영장" in query else "물"
    elif parsed[0] == "front_paw_raise" and "유리문" in query:
        target = "유리문"
        action = "반려동물이 유리문 앞에서 두 앞발을 들어 올림"
    events = []
    for item in confirmed:
        if item.get("model_status") != "confirmed":
            raise ValueError("미확정 후보를 확정 이벤트로 저장할 수 없습니다")
        try:
            before = float(item["before_time"])
            after = float(item["after_time"])
            start = float(item["start_time"] if item.get("start_time") is not None else before)
            end = float(item["end_time"] if item.get("end_time") is not None else after)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("행동 전후 근거 시각이 빠졌습니다") from exc
        if not all(map(math.isfinite, (start, end, before, after))) or not (0 <= before < after and 0 <= start <= end):
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
            # 빠른 반복은 검토 창이 서로 겹쳐도 각각 별도 진입이다.
            "start_time": after if point_transition else min(start, before),
            "end_time": after + 0.1 if point_transition else max(end, after),
            "subject": item.get("model_subject_id") or item.get("subject_id") or "반려동물",
            "action": action,
            "target_object": target,
            "summary": (
                f"반려동물이 {after:g}초에 {target} 물속으로 들어갔어요."
                if parsed[0] == "water_enter" and target != "물" else
                f"반려동물이 {after:g}초에 물속으로 들어갔어요."
                if parsed[0] == "water_enter" else
                f"반려동물이 {after:g}초에 {target} 앞에서 두 앞발을 들어 올렸어요."
                if parsed[0] == "front_paw_raise" and target == "유리문" else
                f"반려동물이 {after:g}초에 두 앞발을 들어 올렸어요."
                if parsed[0] == "front_paw_raise" else
                f"같은 개체의 행동 전후 상태가 {before:g}초와 {after:g}초에 확인됨"
            ),
            "evidence": [
                f"행동 전 상태 {before:g}초, 행동 후 상태 {after:g}초",
                f"후보 재검증: {item.get('reason') or '확인됨'}",
            ],
            "source_frames": [frame],
        })
    return events
