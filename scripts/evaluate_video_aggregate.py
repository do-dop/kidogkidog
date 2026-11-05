"""Compare saved full-video count outputs with independent event-time labels.

This is offline: no API, database, or video processing is required.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "notebooks/smarthome-bench-evidence/new-pipeline"


def match_events(predictions: list[dict], references: list[dict], tolerance: float = 0.75) -> dict:
    """One-to-one time matching; duplicate hits on one reference are false positives."""
    pairs = []
    for pi, prediction in enumerate(predictions):
        time = prediction.get("after_time")
        if time is None:
            time = prediction.get("candidate_time")
        if time is None:
            continue
        for ri, reference in enumerate(references):
            if reference["start"] - tolerance <= time <= reference["end"] + tolerance:
                # The transition begins near the reference start; a later hit
                # inside one long episode is more likely a duplicate.
                pairs.append((abs(time - reference["start"]), pi, ri))
    used_predictions, used_references = set(), set()
    matches = []
    for _, pi, ri in sorted(pairs):
        if pi not in used_predictions and ri not in used_references:
            used_predictions.add(pi)
            used_references.add(ri)
            matches.append({"prediction_index": pi, "reference_index": ri})
    return {
        "matches": matches,
        "unmatched_prediction_indices": [i for i in range(len(predictions)) if i not in used_predictions],
        "missed_reference_indices": [i for i in range(len(references)) if i not in used_references],
    }


def evaluate_case(label: dict, result: dict) -> dict:
    references = label["events"]
    predictions = result["confirmed"]
    matched = match_events(predictions, references) if references else None
    return {
        "video_id": label["video_id"],
        "reference_kind": label["reference_kind"],
        "reference_confirmed": label["confirmed_total"],
        "reference_uncertain": label["uncertain_total"],
        "pipeline_confirmed": result["confirmed_count"],
        "pipeline_uncertain_candidates": result["uncertain_count"],
        "pipeline_rejected": len(result["rejected"]),
        "pipeline_duplicates": len(result["duplicates"]),
        "track_status": result.get("track_status", "not_run_api_quota_exhausted"),
        "time_only_matches": matched,
        "warning": "Time match does not prove animal identity; unresolved model aliases can make a matching total misleading.",
    }


def main() -> None:
    labels = json.loads((EVIDENCE / "validation-labels.json").read_text(encoding="utf-8"))
    cases = []
    for label in labels["cases"]:
        path = EVIDENCE / f"{label['video_id']}-result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        cases.append(evaluate_case(label, result))
    summary = {"method": labels["note"], "cases": cases,
               "pipeline_note": "Cross-window identity resolution and between-event reset checks were rerun. Model judgments and provisional human labels still require event-level review."}
    output = EVIDENCE / "evaluation-summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("video              reference  pipeline  unresolved  time-matched  false  missed")
    for case in cases:
        match = case["time_only_matches"]
        matched = len(match["matches"]) if match else "n/a"
        false = len(match["unmatched_prediction_indices"]) if match else "n/a"
        missed = len(match["missed_reference_indices"]) if match else "n/a"
        print(f"{case['video_id']:<18} {case['reference_confirmed']:>9} {case['pipeline_confirmed']:>9} "
              f"{case['pipeline_uncertain_candidates']:>11} {str(matched):>13} {str(false):>6} {str(missed):>7}")
    print(output)


if __name__ == "__main__":
    main()
