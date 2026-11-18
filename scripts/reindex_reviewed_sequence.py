"""Replace one video's stale sequence events with a reviewed timeline.

Dry run by default. Applying writes a backup before the database transaction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from db.behavior_events import get_behavior_events_for_aggregation, insert_behavior_events  # noqa: E402
from db.connection import connect  # noqa: E402
from pipeline.storage.gcs_uploader import object_exists  # noqa: E402


def build_events(review: dict, scenes: list[dict]) -> list[dict]:
    video_id = str(review.get("video_id") or "")
    source = str(review.get("review_source") or "")
    items = review.get("events")
    if not video_id or not source or not isinstance(items, list) or not items:
        raise ValueError("검수 영상, 출처, 사건 목록이 필요합니다")
    frame_by_time = {float(scene["start_time"]): scene["s3_key"] for scene in scenes}
    events = []
    last_end = -1.0
    for item in items:
        start, end, frame_time = (float(item[key]) for key in ("start_time", "end_time", "frame_time"))
        if not all(math.isfinite(t) for t in (start, end, frame_time)) or not (last_end <= start <= end):
            raise ValueError("검수 사건의 시각이 올바른 순서가 아닙니다")
        key = frame_by_time.get(frame_time) or item.get("frame_key")
        if not key:
            raise ValueError(f"{frame_time:g}초의 저장 프레임이 없습니다")
        if not str(key).startswith(f"frames/{video_id}/") or not str(key).lower().endswith(".jpg"):
            raise ValueError("근거 프레임이 검수 영상의 GCS JPEG가 아닙니다")
        if abs(frame_time - start) > 2.0:
            raise ValueError("재생 근거 프레임이 행동 시각에서 너무 멉니다")
        action, summary = str(item.get("action") or ""), str(item.get("summary") or "")
        if not action or not summary:
            raise ValueError("검수 사건에 행동과 설명이 필요합니다")
        events.append({
            "video_id": video_id,
            "start_time": start,
            "end_time": end,
            "subject": item.get("subject") or review.get("subject") or "반려동물",
            "action": action,
            "target_object": item.get("target_object"),
            "summary": summary,
            "evidence": [source, f"원본 영상 {start:g}~{end:g}초와 저장 프레임 {frame_time:g}초 대조"],
            "source_frames": [{
                "s3_key": key,
                "timestamp": frame_time,
                "frame_id": f"{video_id}__{Path(key).stem}",
                "object_labels": [],
            }],
        })
        last_end = end
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    video_id = str(review.get("video_id") or "")
    expected_ids = set(review.get("replace_event_ids") or [])
    if not video_id or not expected_ids:
        raise ValueError("영상 ID와 교체할 기존 이벤트 ID가 필요합니다")
    conn = connect(dict_rows=True)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT start_time, s3_key FROM scenes WHERE video_id=? ORDER BY start_time", (video_id,))
        events = build_events(review, cursor.fetchall())
        existing = get_behavior_events_for_aggregation(video_id=video_id)
        if {event["id"] for event in existing} != expected_ids:
            raise ValueError("현재 DB 이벤트가 검수 당시와 다릅니다. 적용을 중단합니다")
        print(json.dumps({
            "video_id": video_id,
            "old_event_ids": sorted(expected_ids),
            "new_events": [[event["start_time"], event["end_time"], event["action"]] for event in events],
            "apply": args.apply,
        }, ensure_ascii=False))
        if not args.apply:
            return
        for event in events:
            key = event["source_frames"][0]["s3_key"]
            if not object_exists(key):
                raise FileNotFoundError(f"GCS 근거 프레임이 없습니다: {key}")
        if args.backup.exists():
            raise FileExistsError(f"기존 백업을 덮어쓰지 않습니다: {args.backup}")
        args.backup.write_text(json.dumps({
            "video_id": video_id,
            "review": str(args.review),
            "replaced_events": existing,
        }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        try:
            for event_id in sorted(expected_ids):
                cursor.execute("DELETE FROM behavior_events WHERE id=? AND video_id=?", (event_id, video_id))
                if cursor.rowcount != 1:
                    raise RuntimeError("기존 이벤트가 변경돼 재인덱싱을 중단합니다")
            inserted_ids = insert_behavior_events(events, conn=conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        print(f"저장 완료: {inserted_ids}; 백업 {args.backup}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
