"""Combine verified candidates from fixed-intent 0.5s and 0.25s runs.

Individual candidate evidence is reused. Only cross-run animal identity and
between-event reset checks may call the model again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pipeline.search.video_aggregate import (  # noqa: E402
    reduce_transitions,
    resolve_subject_tracks,
    verify_resets,
)
from scripts.validate_video_aggregate import CachedVision  # noqa: E402


EVIDENCE = PROJECT / "notebooks/smarthome-bench-evidence/new-pipeline"
ABLATION = EVIDENCE / "ablation-cross-video"
SUPPORTED_IDS = ("0188", "0518", "0642")


def combine_verified(baseline: dict, finer: dict) -> list[dict]:
    """Keep every candidate: close timestamps can belong to different dogs."""
    combined = []
    for source, result in (("0.5", baseline), ("0.25", finer)):
        for item in result["verifications"]:
            row = dict(item)
            row["candidate"] = dict(item["candidate"])
            row["candidate"]["source_step"] = source
            row["observation"] = dict(item.get("observation") or {})
            combined.append(row)
    return sorted(combined, key=lambda item: item["candidate"]["time"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", choices=SUPPORTED_IDS, required=True)
    parser.add_argument("--cache-dir", type=Path, default=EVIDENCE / "api-cache")
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    vision = CachedVision(args.cache_dir, args.model, args.model)
    for video_id in args.ids:
        folder = ABLATION / f"smartbench_{video_id}"
        baseline_path = EVIDENCE / f"smartbench_{video_id}-result.json"
        finer_path = folder / "step025" / f"smartbench_{video_id}-result.json"
        if not finer_path.exists():
            raise FileNotFoundError(f"0.25초 검증이 완료되지 않았습니다: {finer_path}")
        video_path = PROJECT / "data/videos" / f"smartbench_{video_id}.mp4"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        finer = json.loads(finer_path.read_text(encoding="utf-8"))
        if baseline["intent"] != finer["intent"]:
            raise ValueError(f"두 실행의 질문 해석이 다릅니다: {video_id}")
        verified = combine_verified(baseline, finer)
        verified = resolve_subject_tracks(video_path, verified, vision)
        verified = verify_resets(video_path, baseline["intent"], verified, vision)
        result = reduce_transitions(verified)
        result.update({
            "video": f"data/videos/smartbench_{video_id}.mp4",
            "query": baseline["query"],
            "intent": baseline["intent"],
            "candidate_sources": {
                "0.5": str(baseline_path.relative_to(PROJECT)),
                "0.25": str(finer_path.relative_to(PROJECT)),
            },
            "candidates": [item["candidate"] for item in verified],
            "verifications": verified,
            "model": args.model,
            "track_status": "model_resolved",
            "method_note": "Previously verified scene evidence reused; animal identity and resets checked across runs.",
        })
        output = folder / "union-result.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(f"{video_id}: confirmed={result['confirmed_count']} uncertain={result['uncertain_count']} "
              f"rejected={len(result['rejected'])} duplicates={len(result['duplicates'])}; "
              f"API new={vision.calls} cache={vision.cached}", flush=True)
        print(output, flush=True)


if __name__ == "__main__":
    main()
