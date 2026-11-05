"""Audit where each reviewed 0936 event exits two saved sampling runs.

Read-only for model outputs: this script makes no API calls and does not edit
either run. It writes a compact, reproducible comparison JSON for the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "notebooks/smarthome-bench-evidence/new-pipeline"
BASELINE = EVIDENCE / "smartbench_0936-result.json"
STEP025 = EVIDENCE / "ablation-0936-step025/smartbench_0936-result.json"
OUTPUT = EVIDENCE / "ablation-0936-step025/comparison.json"
MATCH_TOLERANCE_SECONDS = 0.25


def event_candidates(result: dict, event: dict) -> list[dict]:
    """Find proposals inside a reviewed interval, allowing small label error."""
    lower = event["start"] - MATCH_TOLERANCE_SECONDS
    upper = event["end"] + MATCH_TOLERANCE_SECONDS
    return [candidate for candidate in result["candidates"]
            if lower <= candidate["time"] <= upper]


def audit_event(result: dict, event: dict) -> dict:
    candidates = event_candidates(result, event)
    if not candidates:
        return {"stage": "candidate_missed", "candidate_times": [],
                "details": [], "verified_confirmed": False}

    stages = ("confirmed", "uncertain", "rejected", "duplicates")
    decisions = {
        stage: {item["candidate_time"]: item for item in result[stage]}
        for stage in stages
    }
    verifications = {item["candidate"]["time"]: item
                     for item in result["verifications"]}
    details = []
    for candidate in candidates:
        time = candidate["time"]
        verification = verifications.get(time, {})
        observation = verification.get("observation") or {}
        decision = next((stage for stage in stages if time in decisions[stage]), "missing_decision")
        decision_row = decisions.get(decision, {}).get(time, {})
        details.append({
            "time": time,
            "verification_status": observation.get("status"),
            "final_decision": decision,
            "review_reason": decision_row.get("review_reason"),
            "evidence_check_reason": (
                verification.get("repair_evidence_check")
                or verification.get("evidence_check") or {}
            ).get("reason"),
            "track_id": observation.get("track_id"),
        })

    if any(detail["final_decision"] == "confirmed" for detail in details):
        stage = "confirmed"
    elif any(detail["final_decision"] == "uncertain" for detail in details):
        stage = "final_uncertain"
    elif any(detail["final_decision"] == "rejected" for detail in details):
        stage = "verification_rejected"
    else:
        stage = "duplicate_or_unresolved"
    return {
        "stage": stage,
        "candidate_times": [item["time"] for item in candidates],
        "details": details,
        "verified_confirmed": any(detail["verification_status"] == "confirmed"
                                  for detail in details),
    }


def main() -> None:
    labels = json.loads((EVIDENCE / "validation-labels.json").read_text(encoding="utf-8"))
    reference = next(case for case in labels["cases"]
                     if case["video_id"] == "smartbench_0936")
    runs = {
        "0.5": json.loads(BASELINE.read_text(encoding="utf-8")),
        "0.25": json.loads(STEP025.read_text(encoding="utf-8")),
    }
    rows = []
    for index, event in enumerate(reference["events"], start=1):
        rows.append({
            "event_number": index,
            "reference_start": event["start"],
            "reference_end": event["end"],
            "runs": {step: audit_event(result, event) for step, result in runs.items()},
        })
    metrics = {}
    for step, result in runs.items():
        audits = [row["runs"][step] for row in rows]
        metrics[step] = {
            "scan_frames": sum(len(log["frame_times"]) for log in result["scan_log"]),
            "candidate_proposals": len(result["candidates"]),
            "reference_events_with_candidate": sum(bool(row["candidate_times"])
                                                   for row in audits),
            "reference_events_verified_confirmed": sum(row["verified_confirmed"]
                                                       for row in audits),
            "reference_events_final_confirmed": sum(row["stage"] == "confirmed"
                                                    for row in audits),
            "pipeline_confirmed_count": result["confirmed_count"],
            "pipeline_uncertain_count": result["uncertain_count"],
        }
    report = {
        "video_id": "smartbench_0936",
        "reference_kind": reference["reference_kind"],
        "reference_total": reference["confirmed_total"],
        "matching_rule": "Candidate timestamp within reviewed interval ±0.25s; this is a time-only audit, not independent visual proof.",
        "metrics": metrics,
        "union_candidate_coverage": sum(any(row["runs"][step]["candidate_times"]
                                            for step in runs) for row in rows),
        "events": rows,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    for step, metric in metrics.items():
        print(step, metric)
    print("union candidate coverage", report["union_candidate_coverage"], "/", len(rows))
    print(OUTPUT)


if __name__ == "__main__":
    main()
