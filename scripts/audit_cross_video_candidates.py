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
            "step025_status": "not_run",
            "union_status": "not_run",
        }
        finer_path = folder / "step025" / f"smartbench_{video_id}-result.json"
        if finer_path.exists():
            finer = json.loads(finer_path.read_text(encoding="utf-8"))
            if finer["intent"] != baseline["intent"]:
                raise ValueError(f"질문 해석이 다른 탐색 결과: {video_id}")
            finer.update(reduce_transitions(finer["verifications"]))
            finer_match = match_events(finer["confirmed"], label["events"])
            case.update({
                "step025_status": "complete",
                "step025_scan": describe_candidates(finer["candidates"], label["events"]),
                "step025_confirmed": finer["confirmed_count"],
                "step025_uncertain": finer["uncertain_count"],
                "step025_time_matches": len(finer_match["matches"]),
            })
        else:
            scan_only = folder / "step025-scan-only.json"
            if scan_only.exists():
                partial = json.loads(scan_only.read_text(encoding="utf-8"))
                if partial["intent"] != baseline["intent"]:
                    raise ValueError(f"질문 해석이 다른 탐색 결과: {video_id}")
                case["step025_status"] = "scan_only_api_quota_exhausted"
                case["step025_scan"] = describe_candidates(partial["candidates"], label["events"])
        union_path = folder / "union-result.json"
        if union_path.exists():
            union = json.loads(union_path.read_text(encoding="utf-8"))
            if union["intent"] != baseline["intent"]:
                raise ValueError(f"질문 해석이 다른 합집합 결과: {video_id}")
            union.update(reduce_transitions(union["verifications"]))
            union_match = match_events(union["confirmed"], label["events"])
            case.update({
                "union_status": "complete",
                "union_candidate_proposals": len(union["candidates"]),
                "union_confirmed": union["confirmed_count"],
                "union_uncertain": union["uncertain_count"],
                "union_time_matches": len(union_match["matches"]),
            })
        if video_id == "0642":
            identity_path = folder / "identity-check-cropped.json"
            if identity_path.exists():
                case["cropped_identity_check"] = json.loads(identity_path.read_text(encoding="utf-8"))
        cases.append(case)
    report = {
        "method": "One-to-one candidate-time matching within reviewed interval ±0.75s. It is a recall aid, not proof of a physical transition or animal identity.",
        "run_note": "All three 0.25s and union runs completed after API credits were replenished. The 0642 cropped identity check used a manually selected region and is separate from automatic union counts.",
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print(case["video_id"], "0.5 candidate coverage",
              f"{case['step050_scan']['reference_events_with_candidate']}/{case['reference_count']}",
              "counts", case["step050_confirmed"],
              case.get("step025_confirmed"), case.get("union_confirmed"))
    print(OUTPUT)


if __name__ == "__main__":
    main()
