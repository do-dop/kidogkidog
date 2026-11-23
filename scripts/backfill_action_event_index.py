"""Backfill the event index after deployment.

Usage: python -m scripts.backfill_action_event_index --video-id smartbench_0608
       python -m scripts.backfill_action_event_index --all
"""

import argparse
import os

from db.behavior_events import get_behavior_events_for_aggregation
from pipeline.search.action_event_index import index_behavior_events


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-id", help="Index all events for one video")
    group.add_argument("--all", action="store_true", help="Index all stored behavior events")
    parser.add_argument("--dry-run", action="store_true", help="Only print event counts")
    args = parser.parse_args()

    events = get_behavior_events_for_aggregation(video_id=args.video_id)
    print(f"대상 행동 이벤트: {len(events)}개")
    if args.dry_run:
        return
    if events and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY가 없어 행동 이벤트 임베딩을 만들 수 없습니다.")
    indexed = index_behavior_events(events)
    print(f"의미 색인 완료: {indexed}개")


if __name__ == "__main__":
    main()
