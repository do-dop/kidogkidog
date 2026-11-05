"""Validate the union of two fixed-intent scan candidate lists for one dog.

This is an offline experiment that reads saved scans, calls the vision model
for fresh evidence checks, and writes a separate result. The original runs
are never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pipeline.search.video_aggregate import (  # noqa: E402
    check_claimed_evidence,
    reduce_transitions,
    repair_failed_claims,
    resolve_subject_tracks,
    verify_candidates,
    verify_resets,
)
from scripts.validate_video_aggregate import CachedVision  # noqa: E402


EVIDENCE = PROJECT / "notebooks/smarthome-bench-evidence/new-pipeline"
FIXED = EVIDENCE / "ablation-0936-fixed"
RUNS = {"0.5": FIXED / "step050/smartbench_0936-result.json",
        "0.25": FIXED / "step025/smartbench_0936-result.json"}
INTENT = EVIDENCE / "smartbench_0936-fixed-intent.json"
VIDEO = PROJECT / "data/videos/smartbench_0936.mp4"


def merge_candidates(runs: dict[str, dict], tolerance: float = 0.35) -> list[dict]:
    """Cluster nearby proposals for this single-animal video only."""
    proposals = sorted(
        ({"time": row["time"], "source": step, "subject": row["subject"]}
         for step, result in runs.items() for row in result["candidates"]),
        key=lambda row: row["time"],
    )
    clusters: list[list[dict]] = []
    for proposal in proposals:
        if clusters and proposal["time"] - clusters[-1][-1]["time"] <= tolerance:
            clusters[-1].append(proposal)
        else:
            clusters.append([proposal])
    return [{
        "time": round(sum(row["time"] for row in cluster) / len(cluster), 3),
        "subject": "dog at glass door",
        "reason": "candidate proposed by saved fixed-intent scan",
        "source_proposals": cluster,
    } for cluster in clusters]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FIXED / "union-result.json")
    parser.add_argument("--cache-dir", type=Path, default=EVIDENCE / "api-cache")
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)
    intent = json.loads(INTENT.read_text(encoding="utf-8"))
    runs = {step: json.loads(path.read_text(encoding="utf-8"))
            for step, path in RUNS.items()}
    if any(result["intent"] != intent for result in runs.values()):
        raise ValueError("입력 실행의 질문 해석이 동일하지 않습니다")
    candidates = merge_candidates(runs)
    print("Union candidates:", [row["time"] for row in candidates], flush=True)
    vision = CachedVision(args.cache_dir, args.model, args.model)
    verified = verify_candidates(VIDEO, intent, candidates, vision)
    verified = check_claimed_evidence(VIDEO, intent, verified, vision)
    verified = repair_failed_claims(VIDEO, intent, verified, vision)
    verified = resolve_subject_tracks(VIDEO, verified, vision)
    verified = verify_resets(VIDEO, intent, verified, vision)
    result = reduce_transitions(verified)
    result.update({
        "video": str(VIDEO.relative_to(PROJECT)), "query": runs["0.5"]["query"], "intent": intent,
        "intent_source": str(INTENT.relative_to(PROJECT)),
        "candidate_sources": {step: str(path.relative_to(PROJECT))
                              for step, path in RUNS.items()},
        "candidates": candidates, "verifications": verified,
        "model": args.model, "track_status": "model_resolved",
        "scope_note": "Temporal clustering assumes one target dog; not a multi-animal identity solution.",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"confirmed={result['confirmed_count']} uncertain={result['uncertain_count']} "
          f"rejected={len(result['rejected'])} duplicates={len(result['duplicates'])}", flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
