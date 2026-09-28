"""Read-only YOLOv8s + CLIP species-resolver shadow run on eligible frames.

This script reads the eligible evaluation manifest and exact frame images. It
does not connect to Chroma/MySQL or write to GCS.
"""

import argparse
import hashlib
import json
from pathlib import Path

from pipeline.species_resolver import DEFAULT_BBOX_PADDING, resolve_detection_result
from pipeline.yolo_detector import detect_objects_with_details, get_yolo_model, get_yolo_model_name


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "docs/evaluations"
ELIGIBLE_PATH = EVAL / "yolo-backfill-eligible-2026-09-27.json"
CLIP_EVAL_PATH = EVAL / "yolo-clip-species-evaluation-2026-09-27.json"
YOLO_DRY_RUN_PATH = EVAL / "yolo-backfill-eligible-dry-run-2026-09-27.json"
OUTPUT_PATH = EVAL / "yolo-clip-species-shadow-2026-09-27.json"
REPORT_PATH = EVAL / "yolo-clip-species-shadow-2026-09-27.md"


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-count", type=int, default=240)
    parser.add_argument(
        "--image-cache", type=Path,
        default=Path("/tmp/yolo_clip_species_eval/images"),
        help="Optional local cache populated during the read-only evaluation.",
    )
    return parser.parse_args()


def _resolve_image(frame, cache_dir, bucket, temp_dir):
    # Only use the exact previously verified frame cache or its GCS s3_key.
    # A local frame_path might be stale or belong to a different extraction.
    cached = cache_dir / (hashlib.sha1(frame["frame_id"].encode()).hexdigest() + ".jpg")
    if cached.is_file() and cached.stat().st_size:
        return cached, "evaluation_cache"
    destination = temp_dir / cached.name
    bucket.blob(frame["s3_key"]).download_to_filename(str(destination))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise FileNotFoundError(f"GCS download was empty: {frame['s3_key']}")
    return destination, "gcs_read"


def _canon(detections):
    return sorted((
        d["label"], round(float(d["confidence"]), 4),
        tuple(round(float(v), 2) for v in d["bbox"]),
    ) for d in detections)


def run(expected_count=240, image_cache=None):
    manifest = json.loads(ELIGIBLE_PATH.read_text())
    clip_eval = json.loads(CLIP_EVAL_PATH.read_text())
    prior_yolo = json.loads(YOLO_DRY_RUN_PATH.read_text())
    frames = manifest["frames"]
    if len(frames) != expected_count or expected_count != 240:
        raise SystemExit(f"Refusing run: eligible frames={len(frames)}, expected={expected_count} (must be 240)")
    if len(clip_eval["per_frame_species_resolution"]) != 240 or len(prior_yolo["frames"]) != 240:
        raise SystemExit("Evaluation input count mismatch; expected 240 rows in both comparison files")

    expected_clip = {r["frame_id"]: r for r in clip_eval["per_frame_species_resolution"]}
    expected_yolo = {r["frame_id"]: r for r in prior_yolo["frames"]}
    eligible_ids = {f["frame_id"] for f in frames}
    if eligible_ids != set(expected_clip) or eligible_ids != set(expected_yolo):
        raise SystemExit("Frame ID sets do not match; refusing to process a different set")

    cache_dir = image_cache or Path("/tmp/yolo_clip_species_eval/images")
    tmp_dir = Path("/tmp/yolo_clip_species_shadow_gcs")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    from google.cloud import storage
    bucket = storage.Client().bucket(manifest["gcs_bucket"])

    # detect_objects_with_details intentionally swallows per-frame model errors
    # for normal indexing; preflight here to avoid writing a misleading report.
    try:
        get_yolo_model()
    except Exception as exc:
        raise SystemExit(f"YOLO model unavailable; shadow run aborted before processing: {exc}") from exc

    counts = {
        "processed": 0, "failures": 0, "resolver_applied_detections": 0,
        "yolo_to_clip_species_changes": 0, "dog_to_cat": 0, "cat_to_dog": 0,
        "low_confidence_animal_recoveries": 0, "raw_yolo_mismatch_frames": 0,
        "clip_label_mismatch_frames": 0,
    }
    failures = []
    results = []
    for index, frame in enumerate(frames, 1):
        try:
            image_path, image_source = _resolve_image(frame, cache_dir, bucket, tmp_dir)
            raw = detect_objects_with_details(image_path)
            result = resolve_detection_result(image_path, raw, enabled=True)
            expected_row = expected_clip[frame["frame_id"]]
            expected_labels = set(expected_row["resolutions"][
                f"A_always_clip|pad={DEFAULT_BBOX_PADDING:.2f}|prompt=A_a_cat_dog"
            ])
            resolved_labels = {x["resolved_label"] for x in result["species_resolution"]}
            raw_mismatch = _canon(raw["object_detections"]) != _canon(expected_yolo[frame["frame_id"]]["object_detections"])
            label_mismatch = resolved_labels != expected_labels
            if raw_mismatch:
                counts["raw_yolo_mismatch_frames"] += 1
            if label_mismatch:
                counts["clip_label_mismatch_frames"] += 1

            for resolution in result["species_resolution"]:
                counts["resolver_applied_detections"] += 1
                if resolution["clip_label"] != resolution["yolo_label"]:
                    counts["yolo_to_clip_species_changes"] += 1
                    if resolution["yolo_label"] == "dog" and resolution["clip_label"] == "cat":
                        counts["dog_to_cat"] += 1
                    elif resolution["yolo_label"] == "cat" and resolution["clip_label"] == "dog":
                        counts["cat_to_dog"] += 1
                if 0.4 <= resolution["yolo_confidence"] < 0.5:
                    counts["low_confidence_animal_recoveries"] += 1

            results.append({
                "frame_id": frame["frame_id"], "video_id": frame["video_id"],
                "timestamp": frame["timestamp"], "s3_key": frame["s3_key"],
                "image_source": image_source,
                "raw_yolo_detections": result["object_detections"],
                "species_resolution": result["species_resolution"],
                "final_object_labels": result["object_labels"],
                "expected_clip_catdog_labels": sorted(expected_labels),
                "resolved_catdog_labels": sorted(resolved_labels),
                "raw_yolo_matches_previous_dry_run": not raw_mismatch,
                "clip_labels_match_evaluation": not label_mismatch,
            })
            counts["processed"] += 1
        except Exception as exc:
            failures.append({"frame_id": frame["frame_id"], "error": str(exc)})
            counts["failures"] += 1
        if index % 25 == 0 or index == len(frames):
            print(f"[{index}/{len(frames)}] processed={counts['processed']} failures={counts['failures']}", flush=True)

    result_json = {
        "read_only": True, "eligible_count": len(frames), "model": get_yolo_model_name(),
        "resolver": "existing CLIP ViT-L/14", "padding": DEFAULT_BBOX_PADDING,
        "prompts": {"cat": "a cat", "dog": "a dog"},
        "detection_threshold": 0.4, "ordinary_object_labels_threshold": 0.5,
        "feature_flag_forced_on_for_shadow": True,
        "counts": counts, "failures": failures, "frames": results,
    }
    OUTPUT_PATH.write_text(json.dumps(result_json, ensure_ascii=False, indent=2) + "\n")
    _write_report(result_json)
    return result_json


def _write_report(data):
    c = data["counts"]
    lines = [
        "# YOLO + CLIP species resolver shadow dry-run", "",
        "Eligible 240개 frame만 읽은 read-only 실행이다. Chroma, MySQL, scenes, behavior_events는 읽거나 쓰지 않았고 ineligible frame은 처리하지 않았다.", "",
        f"- 처리: {c['processed']}/240, 실패: {c['failures']}",
        f"- Resolver 적용 detection: {c['resolver_applied_detections']}",
        f"- YOLO→CLIP 종 변경: {c['yolo_to_clip_species_changes']} (dog→cat {c['dog_to_cat']}, cat→dog {c['cat_to_dog']})",
        f"- 0.4–0.5 confidence detection 회복: {c['low_confidence_animal_recoveries']}",
        f"- 이전 dry-run 대비 raw YOLO 불일치 frame: {c['raw_yolo_mismatch_frames']}",
        f"- 평가 Policy A(10% padding, prompt A) 대비 CLIP label 불일치 frame: {c['clip_label_mismatch_frames']}", "",
        "상세 JSON에는 raw detection, detection별 species_resolution, 최종 object_labels, frame별 평가 비교가 들어 있다.", "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    args = _args()
    run(expected_count=args.expected_count, image_cache=args.image_cache)
