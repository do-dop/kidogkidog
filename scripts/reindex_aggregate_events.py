"""Replace one video's old count-action events with reviewed full-video evidence.

Example:
  python scripts/reindex_aggregate_events.py --video-id smartbench_0188 \
    --result notebooks/smarthome-bench-evidence/new-pipeline/ablation-cross-video/smartbench_0188/union-result.json \
    --backup /tmp/smartbench_0188-behavior-backup.json --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from db.behavior_events import get_behavior_events_for_aggregation, insert_behavior_events  # noqa: E402
from db.connection import connect  # noqa: E402
from pipeline.events.aggregate_event_reindex import events_from_count_result  # noqa: E402
from pipeline.search.aggregate_query import parse_count_action  # noqa: E402


def _related_old_event(event: dict, action: str) -> bool:
    text = " ".join(str(event.get(key) or "") for key in ("action", "summary", "target_object"))
    if action == "gate_climb":
        return bool(re.search(r"울타리|펜스|gate|fence", text, re.I)
                    and re.search(r"올라|올랐|타고|넘|기어|climb", text, re.I))
    if action == "drink":
        return bool(re.search(r"마시|마셨|음수|drink", text, re.I))
    if action == "eat":
        return bool(re.search(r"먹|식사|섭취|eat", text, re.I))
    raise ValueError(f"지원하지 않는 행동: {action}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="생략하면 변경 계획만 표시")
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    parsed = parse_count_action(str(result.get("query") or ""))
    if parsed is None:
        raise ValueError("검증 결과에 지원되는 횟수 질문이 없습니다")
    action = parsed[0]

    conn = connect(dict_rows=True)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, start_time, s3_key FROM scenes WHERE video_id=? ORDER BY start_time",
            (args.video_id,),
        )
        scenes = cursor.fetchall()
        new_events = events_from_count_result(result, args.video_id, scenes)
        old_events = get_behavior_events_for_aggregation(video_id=args.video_id)
        replaced = [event for event in old_events if _related_old_event(event, action)]
        if not replaced:
            raise ValueError("교체할 기존 행동 기록이 없습니다. 대상 영상과 DB를 확인하세요")
        print(json.dumps({
            "video_id": args.video_id,
            "old_event_ids": [event["id"] for event in replaced],
            "new_intervals": [[event["start_time"], event["end_time"]] for event in new_events],
            "uncertain_candidates_not_saved": result.get("uncertain_count", 0),
            "apply": args.apply,
        }, ensure_ascii=False))
        if not args.apply:
            return
        if args.backup.exists():
            raise FileExistsError(f"기존 백업을 덮어쓰지 않습니다: {args.backup}")
        args.backup.write_text(json.dumps({
            "video_id": args.video_id,
            "source_result": str(args.result),
            "replaced_events": replaced,
        }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        try:
            for event in replaced:
                cursor.execute("DELETE FROM behavior_events WHERE id=? AND video_id=?", (event["id"], args.video_id))
                if cursor.rowcount != 1:
                    raise RuntimeError("기존 행동 기록이 변경되어 재인덱싱을 중단했습니다")
            insert_behavior_events(new_events, conn=conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        print(f"저장 완료: 확정 행동 {len(new_events)}개; 백업 {args.backup}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
