"""Run the offline full-video aggregate experiment and persist each model reply.

Example:
  .venv/bin/python scripts/validate_video_aggregate.py --ids 0518 0642
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pipeline.search.video_aggregate import (  # noqa: E402
    OpenAIJsonVision,
    analyze_video_count,
    check_claimed_evidence,
    repair_failed_claims,
    reduce_transitions,
    resolve_subject_tracks,
    verify_resets,
)


QUESTIONS = {
    "0142": "고양이가 밥을 먹다가 고개를 몇 번 들었어?",
    "0608": "강아지가 수영장 물에 몇 번 들어갔어?",
    "0936": "강아지가 유리문 앞에서 앞발을 몇 번 들었다 내렸어? 뒷발이 바닥에 남아도 포함해.",
    "0065": "강아지가 소파 좌석 위에 몇 번 올라갔어? 검은 소파와 회색 소파를 모두 세어.",
    "0518": "강아지들이 소파 좌석 위에 몇 번 올라갔어? 개별 강아지를 각각 세어.",
    "0642": "강아지들이 분수 물속에 몇 번 들어갔어? 개별 강아지를 각각 세어.",
    "0188": "강아지가 울타리에 몇 번 올라탔어?",
}


class CachedVision:
    def __init__(self, cache_dir: Path, scan_model: str, verify_model: str):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scan = OpenAIJsonVision(scan_model)
        self.verify = OpenAIJsonVision(verify_model)
        self.model = {"scan_and_intent": scan_model, "verify": verify_model}
        self.calls = 0
        self.cached = 0

    def __call__(self, prompt, frames):
        adapter = self.verify if prompt.startswith(("Verify ONE", "Independently challenge", "A previous candidate", "These images", "You are checking")) else self.scan
        digest = hashlib.sha256()
        digest.update(adapter.model.encode())
        digest.update(prompt.encode())
        for frame in frames:
            digest.update(str(frame["time"]).encode())
            digest.update(hashlib.sha256(frame["jpeg"]).digest())
        path = self.cache_dir / (digest.hexdigest() + ".json")
        if path.exists():
            self.cached += 1
            return json.loads(path.read_text(encoding="utf-8"))
        response = adapter(prompt, frames)
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.calls += 1
        print(f"  API {adapter.model}: {len(frames)} frames, reply {self.calls}; cache {self.cached}", flush=True)
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", choices=sorted(QUESTIONS), required=True)
    parser.add_argument("--scan-model", default="gpt-4o-mini")
    parser.add_argument("--verify-model", default="gpt-4o")
    parser.add_argument("--coarse-step", type=float, default=0.5)
    parser.add_argument("--fine-step", type=float, default=0.25)
    parser.add_argument("--skip-track", action="store_true",
                        help="Skip cross-window identity resolution; counts remain provisional")
    parser.add_argument("--recheck-existing", action="store_true",
                        help="Reuse stored scan/verification output; rerun only exact-frame evidence checks")
    parser.add_argument("--output", type=Path, default=PROJECT / "notebooks/smarthome-bench-evidence/new-pipeline")
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    args.output.mkdir(parents=True, exist_ok=True)
    vision = CachedVision(args.output / "api-cache", args.scan_model, args.verify_model)
    for video_id in args.ids:
        path = PROJECT / "data/videos" / f"smartbench_{video_id}.mp4"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"\n{video_id}: {QUESTIONS[video_id]}", flush=True)
        output = args.output / f"smartbench_{video_id}-result.json"
        if args.recheck_existing and output.exists():
            result = json.loads(output.read_text(encoding="utf-8"))
            result["verifications"] = check_claimed_evidence(
                path, result["intent"], result["verifications"], vision)
            result["verifications"] = repair_failed_claims(
                path, result["intent"], result["verifications"], vision)
            if not args.skip_track:
                try:
                    result["verifications"] = resolve_subject_tracks(
                        path, result["verifications"], vision)
                    result["verifications"] = verify_resets(
                        path, result["intent"], result["verifications"], vision)
                    result["track_status"] = "model_resolved"
                except Exception as exc:
                    result["track_status"] = "unresolved_error"
                    result["track_error"] = f"{type(exc).__name__}: {exc}"
            else:
                result["track_status"] = "skipped"
            result.update(reduce_transitions(result["verifications"]))
        else:
            result = analyze_video_count(path, QUESTIONS[video_id], vision,
                                         coarse_step=args.coarse_step, fine_step=args.fine_step,
                                         resolve_tracks=not args.skip_track)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{video_id}: confirmed={result['confirmed_count']}, uncertain={result['uncertain_count']}, "
              f"candidates={len(result['candidates'])}; saved {output}", flush=True)


if __name__ == "__main__":
    main()
