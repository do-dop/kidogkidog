"""Count fixed-intent 0936 front-paw events from a per-frame state sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pipeline.search.state_timeline import classify_state_frames, transitions_from_states  # noqa: E402
from pipeline.search.video_aggregate import check_claimed_evidence, reduce_transitions  # noqa: E402
from scripts.validate_video_aggregate import CachedVision, QUESTIONS  # noqa: E402


EVIDENCE = PROJECT / "notebooks/smarthome-bench-evidence/new-pipeline"
FIXED = EVIDENCE / "ablation-0936-fixed"
VIDEO = PROJECT / "data/videos/smartbench_0936.mp4"
INTENT = EVIDENCE / "smartbench_0936-fixed-intent.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FIXED / "state-timeline-result.json")
    parser.add_argument("--cache-dir", type=Path, default=EVIDENCE / "api-cache")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--step", type=float, default=0.25)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)
    intent = json.loads(INTENT.read_text(encoding="utf-8"))
    vision = CachedVision(args.cache_dir, args.model, args.model)
    timeline = classify_state_frames(VIDEO, intent, vision, args.step)
    raw = transitions_from_states(timeline["labels"])
    result = {"video": str(VIDEO.relative_to(PROJECT)), "query": QUESTIONS["0936"], "intent": intent,
              "intent_source": str(INTENT.relative_to(PROJECT)), "timeline": timeline,
              "raw_state_transitions": raw,
              "scope_note": "One visible dog is assumed. State labels and independent keyframe checks remain model judgments."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    verified = []
    for event in raw["events"]:
        before, after = event["before_time"], event["after_time"]
        verified.append({
            "candidate": {"time": after, "subject": "dog at glass door"},
            "clip": [before, after],
            "observation": {
                "subject_id": "dog at glass door", "track_id": "animal_1",
                "before_time": before, "after_time": after,
                "start_time": after, "end_time": after,
                "reset_time": event["reset_time"], "status": "confirmed",
                "reason": "before→after edge in timestamped state labels",
            },
        })
    try:
        verified = check_claimed_evidence(VIDEO, intent, verified, vision)
    except Exception as exc:
        result["check_error"] = f"{type(exc).__name__}: {exc}"
        result["verifications"] = verified
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        raise
    result["verifications"] = verified
    result.update(reduce_transitions(verified))
    result["model"] = args.model
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"frames={len(timeline['labels'])} raw_edges={len(raw['events'])} "
          f"confirmed={result['confirmed_count']} uncertain={result['uncertain_count']} "
          f"rejected={len(result['rejected'])}")
    print(args.output)


if __name__ == "__main__":
    main()
