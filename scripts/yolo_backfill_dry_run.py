#!/usr/bin/env python3
"""Read-only YOLO dry-run over every frame in an existing Chroma collection."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_COUNT = 617
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "docs/evaluations/yolo-backfill-dry-run-2026-09-27.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "docs/evaluations/yolo-backfill-dry-run-2026-09-27.md"
DEFAULT_PARTIAL = PROJECT_ROOT / "docs/evaluations/yolo-backfill-dry-run-2026-09-27.partial.json"


def _label_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(x).strip() for x in value if str(x).strip()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            return {str(x).strip() for x in parsed if str(x).strip()}
        if isinstance(parsed, str):
            return {parsed.strip()} if parsed.strip() else set()
        return {x.strip() for x in text.split(",") if x.strip()}
    return {str(value).strip()} if str(value).strip() else set()


def classify_changes(old_labels: Any, new_labels: Any) -> list[str]:
    old, new = _label_set(old_labels), _label_set(new_labels)
    if old == new:
        return ["unchanged"]
    changes = []
    if new - old:
        changes.append("label_added")
    if old - new:
        changes.append("label_removed")
    if not old and new:
        changes.append("empty_to_detected")
    if old and not new:
        changes.append("detected_to_empty")
    if "cat" in old and "cat" not in new and "dog" in new:
        changes.append("cat_to_dog")
    if "dog" in old and "dog" not in new and "cat" in new:
        changes.append("dog_to_cat")
    if "person" in old and "person" not in new:
        changes.append("person_removed")
    if "person" not in old and "person" in new:
        changes.append("person_added")
    if not old:
        changes.extend(f"empty_to_{label}" for label in ("cat", "dog", "person") if label in new)
    return changes


def _detections(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    return value if isinstance(value, list) else []


def _safe_host_name(host: str) -> str:
    """Hide userinfo if a URL-shaped host was accidentally configured."""
    candidate = host if "://" in host else f"//{host}"
    parsed = urlsplit(candidate)
    return parsed.hostname or "<invalid-host>"


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get("video_id") or "<missing>")].append(row)
        counts.update(row.get("change_types") or classify_changes(
            row.get("old_object_labels"), row.get("new_object_labels")
        ))
    changed = sum(
        (row.get("change_types") or classify_changes(row.get("old_object_labels"), row.get("new_object_labels"))) != ["unchanged"]
        for row in records
    )

    def summary(rows: list[dict[str, Any]], counter: Counter[str] | None = None) -> dict[str, Any]:
        c = counter or Counter(
            tag for r in rows for tag in (r.get("change_types") or classify_changes(r.get("old_object_labels"), r.get("new_object_labels")))
        )
        ch = sum(
            (r.get("change_types") or classify_changes(r.get("old_object_labels"), r.get("new_object_labels"))) != ["unchanged"]
            for r in rows
        )
        n = len(rows)
        return {
            "total": n, "unchanged": c["unchanged"], "changed": ch,
            "changed_ratio": ch / n if n else 0.0,
            "empty_to_detected": c["empty_to_detected"], "detected_to_empty": c["detected_to_empty"],
            "old_cat_to_new_dog": c["cat_to_dog"], "old_dog_to_new_cat": c["dog_to_cat"],
            "person_removed": c["person_removed"], "person_added": c["person_added"],
            "empty_to_cat": c["empty_to_cat"], "empty_to_dog": c["empty_to_dog"],
            "empty_to_person": c["empty_to_person"],
        }

    return {"summary": summary(records, counts), "by_video": {k: summary(v) for k, v in sorted(groups.items())}}


def build_report(data: dict[str, Any]) -> str:
    records = data.get("frames", [])
    stats = data.get("statistics") or summarize_records(records)
    overall = stats["summary"]
    lines = [
        "# YOLOv8s backfill dry-run", "",
        f"- 실행 시각(UTC): {data.get('generated_at', '미기록')}",
        f"- Chroma: `{data.get('chroma', {}).get('host', '?')}:{data.get('chroma', {}).get('port', '?')}` / `{data.get('chroma', {}).get('collection', '?')}`",
        f"- Model: `{data.get('model', '?')}`; collection threshold: {data.get('detection_threshold', '?')}; labels threshold: {data.get('labels_threshold', '?')}",
        f"- 처리 성공/실패: {data.get('success_count', 0)} / {len(data.get('failures', []))}", "",
        "## 전체 통계", "", "| 항목 | 개수 | 전체 대비 |", "|---|---:|---:|",
    ]
    metric_labels = [
        ("total", "전체 frame"), ("unchanged", "변경 없음"), ("changed", "label 변경"),
        ("empty_to_detected", "empty → detected"), ("detected_to_empty", "detected → empty"),
        ("old_cat_to_new_dog", "cat → dog"), ("old_dog_to_new_cat", "dog → cat"),
        ("person_removed", "person 제거"), ("person_added", "person 추가"),
        ("empty_to_cat", "empty → cat"), ("empty_to_dog", "empty → dog"), ("empty_to_person", "empty → person"),
    ]
    for key, label in metric_labels:
        n = overall.get(key, 0)
        lines.append(f"| {label} | {n} | {n / overall['total']:.1%} |" if overall.get("total") else f"| {label} | {n} | 0.0% |")
    lines.extend(["", "## video_id별 통계", "", "| video_id | total | unchanged | changed | empty→detected | detected→empty | cat→dog | dog→cat | person− | person+ |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for video_id, row in stats.get("by_video", {}).items():
        lines.append(f"| {video_id} | {row['total']} | {row['unchanged']} | {row['changed']} | {row['empty_to_detected']} | {row['detected_to_empty']} | {row['old_cat_to_new_dog']} | {row['old_dog_to_new_cat']} | {row['person_removed']} | {row['person_added']} |")

    candidate_groups = {
        "cat/dog 변경 후보": lambda r: "cat_to_dog" in r.get("change_types", []) or "dog_to_cat" in r.get("change_types", []),
        "person 변경 후보": lambda r: "person_added" in r.get("change_types", []) or "person_removed" in r.get("change_types", []),
        "empty/detected 변경 후보": lambda r: "empty_to_detected" in r.get("change_types", []) or "detected_to_empty" in r.get("change_types", []),
        "0.4~0.5 confidence detection 후보": lambda r: any(0.4 <= float(d.get("confidence", -1)) < 0.5 for d in r.get("object_detections", [])),
        "multi-object 후보": lambda r: len(r.get("object_detections", [])) > 1,
    }
    lines.extend(["", "## 유형별 후보 (canary 미선정)", ""])
    for title, predicate in candidate_groups.items():
        matches = [r for r in records if predicate(r)]
        lines.extend([f"### {title} ({len(matches)})", ""])
        lines.extend([
            f"- `{r.get('frame_id')}` ({r.get('video_id')}, {r.get('s3_key') or 'no s3_key'}): {r.get('old_object_labels', [])} → {r.get('new_object_labels', [])}; {', '.join(r.get('change_types', []))}"
            for r in matches[:10]
        ] or ["- 해당 없음"])
        lines.append("")
    if data.get("failures"):
        lines.extend(["## 실패", ""])
        lines.extend(f"- `{f.get('frame_id')}`: {f.get('error')}" for f in data["failures"])
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--allow-count-mismatch", action="store_true", help="명시적으로 count 불일치를 허용")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--partial-json", type=Path, default=DEFAULT_PARTIAL)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _read_checkpoint(path: Path, run_config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return {}, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_config") != run_config:
        raise ValueError("partial checkpoint settings differ from this run; use --no-resume or the matching configuration")
    return (
        {r["frame_id"]: r for r in payload.get("frames", []) if r.get("frame_id")},
        payload.get("failures", []),
    )


def _local_or_gcs_bytes(s3_key: str | None, frame_path: str | None) -> bytes:
    if frame_path:
        candidate = Path(frame_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_file():
            return candidate.read_bytes()
    if not s3_key:
        raise ValueError("no readable local frame_path or s3_key")
    from pipeline.gcs_uploader import download_bytes
    return download_bytes(s3_key)


def _run(args: argparse.Namespace) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    sys.path.insert(0, str(PROJECT_ROOT))
    host = os.getenv("CHROMA_HOST", "").strip()
    port = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name = os.getenv("CHROMA_COLLECTION", "petcam_frames")
    if not host:
        print("오류: CHROMA_HOST가 비어 있습니다. 로컬 Chroma fallback은 허용하지 않습니다.", file=sys.stderr)
        return 2

    import chromadb
    from pipeline.yolo_detector import (
        DETECTION_CONFIDENCE_THRESHOLD, OBJECT_LABELS_CONFIDENCE_THRESHOLD,
        TARGET_LABELS, build_detection_result, get_yolo_model, get_yolo_model_name,
    )
    print(f"Chroma target: {_safe_host_name(host)}:{port} / {collection_name}")
    # Do not use vector_store.get_collection(): it calls get_or_create_collection.
    client = chromadb.HttpClient(host=host, port=port, ssl=os.getenv("CHROMA_SSL", "").lower() in {"1", "true", "yes"})
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        print(f"오류: 기존 Chroma collection을 읽지 못했습니다 ({type(exc).__name__}).", file=sys.stderr)
        return 2
    count = collection.count()
    print(f"Indexed frame count: {count}")
    if count != args.expected_count and not args.allow_count_mismatch:
        print(f"경고: 기대 frame 수 {args.expected_count}와 실제 {count}가 다릅니다. 계속하려면 --allow-count-mismatch를 지정하세요.", file=sys.stderr)
        return 2
    if args.checkpoint_every < 1:
        print("오류: --checkpoint-every는 1 이상이어야 합니다.", file=sys.stderr)
        return 2

    frames = []
    offset = 0
    while offset < count:
        batch = collection.get(limit=min(100, count - offset), offset=offset, include=["metadatas"])
        ids = batch.get("ids") or []
        if not ids:
            break
        frames.extend({"frame_id": fid, "metadata": meta or {}} for fid, meta in zip(ids, batch.get("metadatas") or []))
        offset += len(ids)
    if len(frames) != count:
        print(f"오류: count={count}인데 {len(frames)}개만 읽었습니다. 추론을 시작하지 않습니다.", file=sys.stderr)
        return 2

    model_name = get_yolo_model_name()
    print(f"Model: {model_name}; detection threshold: {DETECTION_CONFIDENCE_THRESHOLD}; labels threshold: {OBJECT_LABELS_CONFIDENCE_THRESHOLD}")
    run_config = {
        "host": _safe_host_name(host), "port": port, "collection": collection_name, "count": count,
        "model": model_name, "detection_threshold": DETECTION_CONFIDENCE_THRESHOLD,
        "labels_threshold": OBJECT_LABELS_CONFIDENCE_THRESHOLD,
    }
    try:
        completed, prior_failures = ({}, []) if args.no_resume else _read_checkpoint(args.partial_json, run_config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"오류: checkpoint를 안전하게 재사용할 수 없습니다: {exc}", file=sys.stderr)
        return 2
    frame_ids = {x["frame_id"] for x in frames}
    completed = {k: v for k, v in completed.items() if k in frame_ids}
    failure_map = {x.get("frame_id"): x for x in prior_failures if x.get("frame_id") in frame_ids}
    since_checkpoint = 0
    for index, item in enumerate(frames, 1):
        fid = item["frame_id"]
        if fid in completed:
            continue
        meta = item["metadata"]
        old_labels = sorted(_label_set(meta.get("object_labels")))
        row = {
            "frame_id": fid, "video_id": meta.get("video_id"), "timestamp": meta.get("timestamp"),
            "frame_path": meta.get("frame_path"), "s3_key": meta.get("s3_key"), "old_object_labels": old_labels,
        }
        try:
            image_bytes = _local_or_gcs_bytes(meta.get("s3_key"), meta.get("frame_path"))
            with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
                image_file.write(image_bytes)
                image_file.flush()
                # Call inference without the production helper's catch-all
                # fallback-to-empty behavior, so a YOLO failure is recorded as
                # a failed frame instead of being mistaken for no detections.
                model = get_yolo_model()
                detections = []
                for prediction in model.predict(
                    source=image_file.name,
                    conf=float(DETECTION_CONFIDENCE_THRESHOLD),
                    verbose=False,
                ):
                    for box in prediction.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = prediction.names[class_id]
                        if confidence < DETECTION_CONFIDENCE_THRESHOLD or label not in TARGET_LABELS:
                            continue
                        detections.append({
                            "label": label,
                            "confidence": confidence,
                            "bbox": [float(v) for v in box.xyxy[0].tolist()],
                        })
                result = build_detection_result(
                    detections,
                    target_labels=TARGET_LABELS,
                    object_labels_threshold=OBJECT_LABELS_CONFIDENCE_THRESHOLD,
                )
            new_labels = sorted(_label_set(result.get("object_labels")))
            row.update({
                "new_object_labels": new_labels,
                "object_detections": _detections(result.get("object_detections")),
                "changed": old_labels != new_labels,
                "change_types": classify_changes(old_labels, new_labels),
            })
            completed[fid] = row
            failure_map.pop(fid, None)
        except Exception as exc:
            failure_map[fid] = {
                **row,
                "new_object_labels": None,
                "object_detections": [],
                "changed": None,
                "change_types": ["failed"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        since_checkpoint += 1
        if index == 1 or index % 50 == 0 or index == count:
            print(f"[{index}/{count}] success={len(completed)} failures={len(failure_map)}")
        if since_checkpoint >= args.checkpoint_every:
            _write_json(args.partial_json, {"generated_at": datetime.now(timezone.utc).isoformat(), "run_config": run_config, "frames": list(completed.values()), "failures": list(failure_map.values())})
            since_checkpoint = 0

    ordered = [
        completed.get(x["frame_id"]) or failure_map[x["frame_id"]]
        for x in frames if x["frame_id"] in completed or x["frame_id"] in failure_map
    ]
    failures = [failure_map[x["frame_id"]] for x in frames if x["frame_id"] in failure_map]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chroma": {"host": _safe_host_name(host), "port": port, "collection": collection_name, "count": count},
        "model": model_name, "detection_threshold": DETECTION_CONFIDENCE_THRESHOLD,
        "labels_threshold": OBJECT_LABELS_CONFIDENCE_THRESHOLD,
        "success_count": sum(row.get("changed") is not None for row in ordered),
        "failures": failures, "statistics": summarize_records([row for row in ordered if row.get("changed") is not None]), "frames": ordered,
    }
    _write_json(args.output_json, output)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_report(output), encoding="utf-8")
    _write_json(args.partial_json, {"generated_at": output["generated_at"], "run_config": run_config, "frames": [r for r in ordered if r.get("changed") is not None], "failures": failures})
    print(f"완료: success={output['success_count']}, failures={len(failures)}")
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_md}\nCheckpoint: {args.partial_json}")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
