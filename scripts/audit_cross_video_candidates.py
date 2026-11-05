"""Audit saved cross-video aggregate candidates without API or video access."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_video_aggregate import match_events  # noqa: E402
from pipeline.search.video_aggregate import reduce_transitions  # noqa: E402


EVIDENCE = ROOT / "notebooks/smarthome-bench-evidence/new-pipeline"
ABLATION = EVIDENCE / "ablation-cross-video"
OUTPUT = ABLATION / "offline-audit.json"
IDS = ("0188", "0518", "0642")


def describe_candidates(candidates: list[dict], references: list[dict]) -> dict:
    predictions = [{"candidate_time": candidate["time"]} for candidate in candidates]
    matches = match_events(predictions, references)
    return {
        "proposal_count": len(candidates),
        "reference_events_with_candidate": len(matches["matches"]),
        "unmatched_proposals": len(matches["unmatched_prediction_indices"]),
        "matched_times": [{"reference_index": row["reference_index"],
                           "candidate_time": candidates[row["prediction_index"]]["time"]}
                          for row in matches["matches"]],
        "candidate_times": [candidate["time"] for candidate in candidates],
    }


def main() -> None:
    labels = json.loads((EVIDENCE / "validation-labels.json").read_text(encoding="utf-8"))
    by_id = {case["video_id"]: case for case in labels["cases"]}
    cases = []
    for video_id in IDS:
        label = by_id[f"smartbench_{video_id}"]
        folder = ABLATION / f"smartbench_{video_id}"
        baseline = json.loads((EVIDENCE / f"smartbench_{video_id}-result.json").read_text(encoding="utf-8"))
        baseline.update(reduce_transitions(baseline["verifications"]))
        final_match = match_events(baseline["confirmed"], label["events"])
        case = {
            "video_id": label["video_id"],
            "reference_kind": label["reference_kind"],
            "reference_count": label["confirmed_total"],
            "step050_scan": describe_candidates(baseline["candidates"], label["events"]),
            "step050_confirmed": baseline["confirmed_count"],
            "step050_uncertain": baseline["uncertain_count"],
            "step050_time_matches": len(final_match["matches"]),
            "step025_status": "not_run_api_quota_exhausted",
        }
        scan_only = folder / "step025-scan-only.json"
        if scan_only.exists():
            partial = json.loads(scan_only.read_text(encoding="utf-8"))
            if partial["intent"] != baseline["intent"]:
                raise ValueError(f"질문 해석이 다른 탐색 결과: {video_id}")
            case["step025_status"] = "scan_only_api_quota_exhausted"
            case["step025_scan"] = describe_candidates(partial["candidates"], label["events"])
        cases.append(case)
    report = {
        "method": "One-to-one candidate-time matching within reviewed interval ±0.75s. It is a recall aid, not proof of a physical transition or animal identity.",
        "quota_note": "OpenAI returned 429 credit_balance_exhausted during 0642 verification. No new 0.25s scan for 0188/0518 or final 0.25s/union count is available.",
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print(case["video_id"], "0.5 candidate coverage",
              f"{case['step050_scan']['reference_events_with_candidate']}/{case['reference_count']}",
              "confirmed", case["step050_confirmed"],
              "0.25", case["step025_status"])
    print(OUTPUT)


if __name__ == "__main__":
    main()
