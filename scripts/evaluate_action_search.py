"""Classify annotated action QA cases as extraction or search failures.

The input is a JSON list of {video_id, query, action_visible,
expected_event_ids}. Human review of the source video supplies action_visible
and expected_event_ids; this tool never treats model output as ground truth.
"""

import argparse
import json
from pathlib import Path

from db.behavior_events import get_behavior_events_for_aggregation
from pipeline.search.rag_chain import run_rag_query


def evaluate_case(case):
    video_id = case["video_id"]
    query = case["query"]
    visible = bool(case["action_visible"])
    expected = {int(value) for value in case.get("expected_event_ids") or []}
    stored = {int(row["id"]) for row in get_behavior_events_for_aggregation(video_id=video_id)}
    answer = run_rag_query(query, video_id=video_id)
    found = {int(row["source_event_id"]) for row in answer.get("results") or []
             if row.get("source_event_id") is not None}
    if not visible:
        outcome = "false_positive" if answer.get("results") and answer.get("match_level") not in {"related", "unverified_action"} else "correct_absence"
    elif not expected or not expected.intersection(stored):
        outcome = "extraction_failure"
    elif expected.intersection(found):
        outcome = "success"
    else:
        outcome = "search_failure"
    return {"video_id": video_id, "query": query, "outcome": outcome,
            "expected_event_ids": sorted(expected), "result_event_ids": sorted(found),
            "answer": answer.get("answer")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, help="Human-reviewed JSON case list")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if not isinstance(cases, list):
        parser.error("cases must be a JSON list")
    for case in cases:
        print(json.dumps(evaluate_case(case), ensure_ascii=False))


if __name__ == "__main__":
    main()
