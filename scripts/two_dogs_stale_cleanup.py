"""Exact, backed-up cleanup for the 85 stale two_dogs frames.

This script is intentionally allowlist-driven. It never deletes by video_id and
does not mutate behavior_events or cats_5min. Run with --execute only after the
SSH tunnel and database credentials are available.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from db.connection import connect
from pipeline import vector_store
from scripts.yolo_clip_canary_backfill import _chroma_rows, _get_chroma, _json_safe, _parse_json_field

EVAL = ROOT / "docs/evaluations"
STALE_PATH = EVAL / "yolo-backfill-ineligible-2026-09-27.json"
ELIGIBLE_PATH = EVAL / "yolo-backfill-eligible-2026-09-27.json"
BACKUP_PATH = EVAL / "backups/two-dogs-stale-cleanup-before-2026-09-27.json"
REPORT_PATH = EVAL / "two-dogs-stale-cleanup-result-2026-09-27.json"


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        import base64
        return {"base64": base64.b64encode(value).decode("ascii")}
    return str(value)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    tmp.replace(path)


def _load_targets():
    stale_doc = json.loads(STALE_PATH.read_text())
    eligible_doc = json.loads(ELIGIBLE_PATH.read_text())
    stale = [f for f in stale_doc["frames"] if f.get("video_id") == "two_dogs"]
    eligible = [f for f in eligible_doc["frames"] if f.get("video_id") == "two_dogs"]
    if len(stale) != 85 or len({x["frame_id"] for x in stale}) != 85:
        raise RuntimeError(f"Expected exactly 85 unique stale two_dogs frames, got {len(stale)}")
    if len(eligible) != 90 or len({x["frame_id"] for x in eligible}) != 90:
        raise RuntimeError(f"Expected exactly 90 unique eligible two_dogs frames, got {len(eligible)}")
    stale_ids, eligible_ids = {x["frame_id"] for x in stale}, {x["frame_id"] for x in eligible}
    stale_keys = {x["s3_key"] for x in stale}
    eligible_keys = {x["s3_key"] for x in eligible}
    if stale_ids & eligible_ids or stale_keys & eligible_keys:
        raise RuntimeError("Stale and eligible two_dogs manifest entries overlap")
    return stale, eligible


def _raw_float32_hex(value):
    import numpy as np
    return np.asarray(value, dtype=np.float32).tobytes().hex() if value is not None else None


def _rows_equal(a, b):
    return json.dumps(_json_safe(a), sort_keys=True, ensure_ascii=False) == json.dumps(
        _json_safe(b), sort_keys=True, ensure_ascii=False
    )


def _get_scenes(conn, frames):
    cur = conn.cursor()
    mapping, missing, duplicates = {}, [], []
    for frame in frames:
        cur.execute("SELECT * FROM scenes WHERE video_id = ? AND s3_key = ? ORDER BY id", (frame["video_id"], frame["s3_key"]))
        rows = cur.fetchall()
        mapping[frame["frame_id"]] = rows
        if not rows:
            missing.append(frame["frame_id"])
        elif len(rows) != 1:
            duplicates.append({"frame_id": frame["frame_id"], "scene_ids": [r["id"] for r in rows]})
    return mapping, missing, duplicates


def _event_refs(conn, stale):
    cur = conn.cursor()
    cur.execute("SELECT id, video_id, source_frames_json FROM behavior_events WHERE video_id = ?", ("two_dogs",))
    rows = cur.fetchall()
    stale_by_key = {(x["video_id"], x["s3_key"]): x["frame_id"] for x in stale}
    stale_ids = {x["frame_id"] for x in stale}
    refs = []
    for row in rows:
        sources = _parse_json_field(row.get("source_frames_json"))
        for i, source in enumerate(sources if isinstance(sources, list) else []):
            if not isinstance(source, dict):
                continue
            fid = source.get("frame_id")
            if fid not in stale_ids:
                fid = stale_by_key.get((source.get("video_id") or row["video_id"], source.get("s3_key")))
            if fid:
                refs.append({"event_id": row["id"], "source_index": i, "frame_id": fid})
    return rows, refs


def _preflight():
    stale, eligible = _load_targets()
    ids = [f["frame_id"] for f in stale]
    client, collection = _get_chroma()
    if collection.count() < 617:
        raise RuntimeError(f"Unexpected Chroma collection count: {collection.count()}")
    stale_chroma = _chroma_rows(collection, ids)
    eligible_chroma = _chroma_rows(collection, [f["frame_id"] for f in eligible])
    missing_stale = sorted(set(ids) - set(stale_chroma))
    missing_eligible = sorted({f["frame_id"] for f in eligible} - set(eligible_chroma))
    wrong_identity = [
        fid for fid, row in stale_chroma.items()
        if row["metadata"].get("video_id") != "two_dogs"
    ]
    conn = connect(dict_rows=True)
    scenes, missing_scenes, duplicate_scenes = _get_scenes(conn, stale)
    eligible_scenes, eligible_missing, eligible_duplicates = _get_scenes(conn, eligible)
    events, refs = _event_refs(conn, stale)
    cur = conn.cursor()
    cur.execute("SELECT * FROM scenes WHERE video_id = ? ORDER BY id", ("cats_5min",))
    cats_scenes = cur.fetchall()
    conn.close()
    errors = {
        "stale_chroma_missing": missing_stale,
        "stale_scene_missing": missing_scenes,
        "stale_scene_duplicates": duplicate_scenes,
        "stale_chroma_wrong_video": wrong_identity,
        "eligible_chroma_missing": missing_eligible,
        "eligible_scene_missing": eligible_missing,
        "eligible_scene_duplicates": eligible_duplicates,
        "stale_event_references": refs,
    }
    if any(errors.values()):
        raise RuntimeError("Preflight mismatch; no deletion attempted: " + json.dumps(errors, ensure_ascii=False))
    return stale, eligible, client, collection, stale_chroma, eligible_chroma, scenes, eligible_scenes, events, cats_scenes


def _backup(preflight):
    if BACKUP_PATH.exists():
        raise RuntimeError(f"Refusing to overwrite cleanup backup: {BACKUP_PATH}")
    stale, eligible, _, _, stale_chroma, eligible_chroma, scenes, eligible_scenes, events, cats_scenes = preflight
    chroma_records = []
    for fid, row in stale_chroma.items():
        chroma_records.append({**row, "embedding_float32_bytes_hex": _raw_float32_hex(row["embedding"])})
    # Save all fields of the exact stale rows. No event rows are modified; the
    # zero-reference scan is retained as evidence in the backup/report.
    stale_scene_rows = [rows[0] for rows in scenes.values()]
    _write_json(BACKUP_PATH, {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "exact stale two_dogs manifest entries only",
        "stale_manifest": str(STALE_PATH.relative_to(ROOT)),
        "eligible_manifest": str(ELIGIBLE_PATH.relative_to(ROOT)),
        "stale_frame_ids": [x["frame_id"] for x in stale],
        "eligible_frame_ids_snapshot": [x["frame_id"] for x in eligible],
        "chroma_stale_records": chroma_records,
        "mysql_stale_scenes": stale_scene_rows,
        "mysql_scene_id_by_frame": {fid: rows[0]["id"] for fid, rows in scenes.items()},
        "behavior_event_rows_scanned": events,
        "behavior_event_stale_references": [],
        "eligible_chroma_snapshot": [
            {**row, "embedding_float32_bytes_hex": _raw_float32_hex(row["embedding"])}
            for row in eligible_chroma.values()
        ],
        "eligible_scene_ids": {fid: rows[0]["id"] for fid, rows in eligible_scenes.items()},
    })
    return json.loads(BACKUP_PATH.read_text())


def _restore_chroma(collection, records):
    for row in records:
        # Chroma metadata accepts scalar fields; the saved embedding/document
        # are reused verbatim, and existing rows are never overwritten.
        existing = collection.get(ids=[row["frame_id"]], include=[]).get("ids", [])
        if existing:
            continue
        collection.add(
            ids=[row["frame_id"]],
            embeddings=[row["embedding"]],
            documents=[row["document"]],
            metadatas=[row["metadata"]],
        )


def _execute(preflight, backup):
    stale, eligible, client, collection, stale_chroma, eligible_chroma, scenes, eligible_scenes, _, _ = preflight
    stale_ids = [x["frame_id"] for x in stale]
    scene_ids = [rows[0]["id"] for rows in scenes.values()]
    # Exact Chroma ID deletion only.
    collection.delete(ids=stale_ids)
    after_stale = collection.get(ids=stale_ids, include=[]).get("ids", [])
    after_eligible = _chroma_rows(collection, [x["frame_id"] for x in eligible])
    eligible_backup = {x["frame_id"]: x for x in backup["eligible_chroma_snapshot"]}
    eligible_checks = {
        fid: (fid in after_eligible and _raw_float32_hex(after_eligible[fid]["embedding"]) == eligible_backup[fid]["embedding_float32_bytes_hex"]
              and after_eligible[fid]["document"] == eligible_backup[fid]["document"]
              and after_eligible[fid]["metadata"] == eligible_backup[fid]["metadata"])
        for fid in eligible_backup
    }
    if after_stale or len(eligible_checks) != 90 or not all(eligible_checks.values()):
        _restore_chroma(collection, backup["chroma_stale_records"])
        raise RuntimeError("Chroma post-delete verification failed; attempted exact Chroma restoration")

    conn = None
    try:
        conn = connect(dict_rows=True)
        cur = conn.cursor()
        for sid in scene_ids:
            cur.execute("DELETE FROM scenes WHERE id = ? AND video_id = ?", (sid, "two_dogs"))
            if cur.rowcount != 1:
                raise RuntimeError(f"Expected one row deleted for scene.id={sid}; got {cur.rowcount}")
        remaining_stale, _, duplicate_stale = _get_scenes(conn, stale)
        remaining_eligible, missing_eligible, duplicate_eligible = _get_scenes(conn, eligible)
        if any(remaining_stale.values()) or missing_eligible or duplicate_stale or duplicate_eligible:
            raise RuntimeError("MySQL in-transaction verification failed")
        conn.commit()
    except Exception:
        raw = getattr(conn, "_raw_connection", None) if conn is not None else None
        if raw is not None:
            raw.rollback()
        _restore_chroma(collection, backup["chroma_stale_records"])
        raise
    finally:
        if conn is not None:
            conn.close()

    return {
        "chroma_deleted": len(stale_ids),
        "chroma_stale_remaining": len(after_stale),
        "eligible_chroma_preserved": sum(eligible_checks.values()),
        "mysql_deleted": len(scene_ids),
        "scene_ids_deleted": scene_ids,
        "eligible_scene_preserved": 90,
        "collection_count_after": collection.count(),
    }


def _search_check():
    queries = {"cat": "고양이가 보이는 장면이 있어?", "dog": "강아지가 보이는 장면이 있어?", "person": "사람이 보이는 장면이 있어?"}
    stale_doc = json.loads(STALE_PATH.read_text())
    stale_ids = {f["frame_id"] for f in stale_doc["frames"] if f["video_id"] == "two_dogs"}
    result = {}
    for label, query in queries.items():
        hits = vector_store.search(query, top_k=1000, video_id="two_dogs")
        hit_ids = {x.get("frame_id") for x in hits}
        result[label] = {"candidate_count": len(hits), "stale_hits": sorted(hit_ids & stale_ids)}
    return result


def _resume_verify(backup):
    """Verify an interrupted/finished run without attempting any new deletion."""
    stale_records = backup["chroma_stale_records"]
    stale_ids = [r["frame_id"] for r in stale_records]
    eligible = backup["eligible_chroma_snapshot"]
    client, collection = _get_chroma()
    stale_present = collection.get(ids=stale_ids, include=[]).get("ids", [])
    eligible_now = _chroma_rows(collection, [r["frame_id"] for r in eligible])
    chroma_eligible_ok = {
        saved["frame_id"]: (
            saved["frame_id"] in eligible_now
            and saved["document"] == eligible_now[saved["frame_id"]]["document"]
            and saved["metadata"] == eligible_now[saved["frame_id"]]["metadata"]
            and saved["embedding_float32_bytes_hex"] == _raw_float32_hex(eligible_now[saved["frame_id"]]["embedding"])
        )
        for saved in eligible
    }
    conn = connect(dict_rows=True)
    cur = conn.cursor()
    old_scene_ids = [row["id"] for row in backup["mysql_stale_scenes"]]
    eligible_scene_ids = list(backup["eligible_scene_ids"].values())
    def fetch_by_ids(ids):
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        cur.execute(f"SELECT * FROM scenes WHERE id IN ({marks}) ORDER BY id", tuple(ids))
        return cur.fetchall()
    stale_scenes_now = fetch_by_ids(old_scene_ids)
    eligible_scenes_now = fetch_by_ids(eligible_scene_ids)
    cur.execute("SELECT id, video_id, source_frames_json FROM behavior_events WHERE video_id = ?", ("two_dogs",))
    events_now = cur.fetchall()
    conn.close()
    stale_key_to_id = {(r["metadata"].get("video_id"), r["metadata"].get("s3_key")): r["frame_id"] for r in stale_records}
    event_refs = []
    for event in events_now:
        sources = _parse_json_field(event.get("source_frames_json"))
        for i, source in enumerate(sources if isinstance(sources, list) else []):
            if not isinstance(source, dict):
                continue
            fid = source.get("frame_id") or stale_key_to_id.get((source.get("video_id") or event["video_id"], source.get("s3_key")))
            if fid in set(stale_ids):
                event_refs.append({"event_id": event["id"], "source_index": i, "frame_id": fid})
    return {
        "collection_count": collection.count(),
        "chroma_stale_remaining": len(stale_present),
        "chroma_eligible_preserved": sum(chroma_eligible_ok.values()),
        "chroma_eligible_total": len(eligible),
        "chroma_eligible_failures": [fid for fid, ok in chroma_eligible_ok.items() if not ok],
        "mysql_stale_scenes_remaining": len(stale_scenes_now),
        "mysql_eligible_scenes_preserved": len(eligible_scenes_now),
        "mysql_eligible_scene_ids_expected": len(eligible_scene_ids),
        "behavior_event_stale_references": event_refs,
        "searches": _search_check(),
        "suggestions": _suggestion_check(),
    }


def _suggestion_check():
    from pipeline.query_suggester import suggest_query_items
    rows = suggest_query_items(video_id="two_dogs", limit=3)
    return {
        "count": len(rows),
        "items": [{
            "question": row.get("question"),
            "question_source": row.get("question_source"),
            "source_event_id": row.get("source_event_id"),
            "original_event_id": row.get("original_event_id"),
            "event_start": row.get("event_start"),
            "event_end": row.get("event_end"),
        } for row in rows],
        "stale_cat_fallback_present": any("고양이" in str(row.get("question", "")) for row in rows),
        "limit_respected": len(rows) <= 3,
    }


def _cats_read_only_audit():
    from google.cloud import storage
    stale_doc = json.loads(STALE_PATH.read_text())
    cats_stale = [f for f in stale_doc["frames"] if f["video_id"] == "cats_5min"]
    stale_keys = {f["s3_key"] for f in cats_stale}
    eligible_doc = json.loads(ELIGIBLE_PATH.read_text())
    cats_eligible = [f for f in eligible_doc["frames"] if f["video_id"] == "cats_5min"]
    conn = connect(dict_rows=True)
    cur = conn.cursor()
    cur.execute("SELECT * FROM scenes WHERE video_id = ? ORDER BY id", ("cats_5min",))
    rows = cur.fetchall()
    cur.execute("SELECT id, video_id, action, target_object, summary, source_frames_json FROM behavior_events WHERE video_id = ? ORDER BY id", ("cats_5min",))
    events = cur.fetchall()
    conn.close()
    _, collection = _get_chroma()
    chroma_all = collection.get(where={"video_id": "cats_5min"}, include=["metadatas"])
    chroma_items = {fid: meta for fid, meta in zip(chroma_all.get("ids", []), chroma_all.get("metadatas", []))}
    bucket_name = __import__("os").getenv("GCS_BUCKET_NAME") or __import__("os").getenv("GCS_BUCKET")
    if not bucket_name:
        raise RuntimeError("GCS bucket env is required for read-only scene object audit")
    bucket = storage.Client().bucket(bucket_name)
    # List once and compare exact keys; no object downloads or writes.
    object_keys = {blob.name for blob in bucket.list_blobs(prefix="frames/cats_5min/")}
    frame_root = ROOT / "pipeline/frames/cats_5min"
    scene_groups = Counter()
    scene_rows = []
    for row in rows:
        key = row.get("s3_key")
        path = (ROOT / "pipeline" / key) if key else None
        local = bool(path and path.is_file())
        gcs = bool(key and key in object_keys)
        scene_groups["stale_manifest_key" if key in stale_keys else "other_key"] += 1
        stem = Path(key).name.split("_frame_", 1)[0] if key else None
        matched = [fid for fid, meta in chroma_items.items() if meta.get("s3_key") == key] if key else []
        same_stem = [fid for fid, meta in chroma_items.items() if key and Path(str(meta.get("s3_key", ""))).name.split("_frame_", 1)[0] == stem]
        scene_rows.append({
            "id": row.get("id"), "video_id": row.get("video_id"), "s3_key": key,
            "start_time": row.get("start_time"), "end_time": row.get("end_time"),
            "object_labels": row.get("object_labels"),
            "object_detections_json": row.get("object_detections_json"),
            "species_resolution_json": row.get("species_resolution_json"),
            "created_at": row.get("created_at"),
            "local_image_exists": local, "gcs_image_exists": gcs,
            "exact_chroma_frame_ids": matched, "same_chunk_stem_chroma_match_count": len(same_stem),
            "same_chunk_stem_chroma_frame_id_sample": same_stem[:5],
        })
    def labels_for(row):
        raw = row.get("object_labels")
        parsed = _parse_json_field(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, list):
            return [str(label) for label in parsed]
        if isinstance(parsed, str):
            return [parsed]
        return [x.strip() for x in str(raw).split(",") if x.strip()] if raw else []
    before_label_counts = Counter(label for row in rows for label in labels_for(row))
    labels_after_removing_stale_scenes = Counter()
    for row in rows:
        if row.get("s3_key") in stale_keys:
            continue
        labels_after_removing_stale_scenes.update(labels_for(row))
    event_stale_refs = []
    for event in events:
        srcs = _parse_json_field(event.get("source_frames_json"))
        for i, src in enumerate(srcs if isinstance(srcs, list) else []):
            if isinstance(src, dict) and (src.get("frame_id") in {f["frame_id"] for f in cats_stale} or src.get("s3_key") in stale_keys):
                event_stale_refs.append({"event_id": event["id"], "source_index": i})
    # Query recordings prefix without exposing signed URLs.
    chunks = [b.name for b in bucket.list_blobs(prefix="chunks/cats_5min/") if b.name.lower().endswith(".mp4")]
    other_rows = [r for r in scene_rows if r["s3_key"] not in stale_keys]
    stem_counts = Counter()
    timestamp_rows = {}
    for row in other_rows:
        match = re.search(r"([^/]+)_frame_[0-9.]+\.jpg$", str(row.get("s3_key") or ""))
        stem_counts[match.group(1) if match else "NO_STEM"] += 1
        stamp = round(float(row["start_time"]), 3)
        timestamp_rows.setdefault(stamp, []).append(row["id"])
    return {
        "stale_manifest_frame_count": len(cats_stale),
        "eligible_manifest_frame_count": len(cats_eligible),
        "scene_count": len(rows), "scene_key_groups": dict(scene_groups),
        "other_scene_rows": other_rows,
        "other_351_chunk_stem_counts": dict(stem_counts),
        "other_351_unique_start_times": len(timestamp_rows),
        "other_351_duplicate_start_time_groups": sum(len(ids) > 1 for ids in timestamp_rows.values()),
        "other_351_duplicate_rows_beyond_unique_times": sum(len(ids) - 1 for ids in timestamp_rows.values()),
        "all_scene_rows": scene_rows,
        "image_counts_all_scenes": {"gcs": sum(r["gcs_image_exists"] for r in scene_rows), "local": sum(r["local_image_exists"] for r in scene_rows)},
        "image_counts_other_351": {"gcs": sum(r["gcs_image_exists"] for r in scene_rows if r["s3_key"] not in stale_keys), "local": sum(r["local_image_exists"] for r in scene_rows if r["s3_key"] not in stale_keys)},
        "chroma_exact_scene_key_matches": sum(bool(r["exact_chroma_frame_ids"]) for r in scene_rows),
        "chroma_same_stem_matches": sum(bool(r["same_chunk_stem_chroma_match_count"]) for r in scene_rows),
        "events": events, "stale_event_refs": event_stale_refs,
        "labels_if_only_292_stale_scenes_removed": dict(labels_after_removing_stale_scenes),
        "scene_label_counts_before_292_removed": dict(before_label_counts),
        "gcs_video_chunks": chunks,
        "recordings_api_source": "GCS chunks/{video_id}/ listing; no dedicated SQL recordings table",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Back up and delete only exact two_dogs stale IDs/scene IDs")
    parser.add_argument("--resume-verify", action="store_true", help="Verify a backed-up cleanup without deleting anything")
    args = parser.parse_args()
    if args.resume_verify:
        if not BACKUP_PATH.exists():
            raise RuntimeError(f"Cleanup backup not found: {BACKUP_PATH}")
        backup = json.loads(BACKUP_PATH.read_text())
        report = {"created_at": datetime.now().astimezone().isoformat(), "scope": "two_dogs exact stale 85 only",
                  "backup": str(BACKUP_PATH.relative_to(ROOT)), "resume_verification": _resume_verify(backup)}
        try:
            report["cats_5min_read_only_audit"] = _cats_read_only_audit()
        except Exception as exc:
            report["cats_5min_read_only_audit_error"] = f"{type(exc).__name__}: {exc}"
        report["rollback_dry_run"] = {"chroma_records_restorable": len(backup["chroma_stale_records"]),
                                      "mysql_scenes_restorable": len(backup["mysql_stale_scenes"]),
                                      "writes_performed": False}
        _write_json(REPORT_PATH, report)
        cats = report.get("cats_5min_read_only_audit", {})
        print(json.dumps({
            "resume_verification": report["resume_verification"],
            "cats_5min_summary": {k: v for k, v in cats.items() if k not in {"all_scene_rows", "other_scene_rows", "events"}},
            "cats_audit_error": report.get("cats_5min_read_only_audit_error"),
            "rollback_dry_run": report["rollback_dry_run"],
            "report": str(REPORT_PATH.relative_to(ROOT)),
        }, ensure_ascii=False, default=_json_default))
        return
    preflight = _preflight()
    stale, eligible = preflight[0], preflight[1]
    print(json.dumps({"preflight": "passed", "stale_chroma": len(preflight[4]), "stale_scenes": len(preflight[6]),
                      "eligible_chroma": len(preflight[5]), "eligible_scenes": len(preflight[7]),
                      "event_rows_scanned": len(preflight[8]), "event_refs": 0,
                      "collection_count": preflight[3].count()}, ensure_ascii=False))
    backup = _backup(preflight)
    print(f"backup={BACKUP_PATH}")
    cleanup = _execute(preflight, backup) if args.execute else {"executed": False}
    report = {"created_at": datetime.now().astimezone().isoformat(), "scope": "two_dogs exact stale 85 only",
              "backup": str(BACKUP_PATH.relative_to(ROOT)), "preflight": {"stale": 85, "eligible": 90, "event_refs": 0},
              "cleanup": cleanup}
    if args.execute:
        report["post_cleanup_searches"] = _search_check()
        if any(v["stale_hits"] for v in report["post_cleanup_searches"].values()):
            report["search_verification"] = "stale IDs still present in search; inspect before further action"
        else:
            report["search_verification"] = "pass: no stale IDs in cat/dog/person results"
    report["cats_5min_read_only_audit"] = _cats_read_only_audit()
    # Rollback dry-run only: this reports exact restorable rows, no writes.
    report["rollback_dry_run"] = {"chroma_records_restorable": len(backup["chroma_stale_records"]),
                                  "mysql_scenes_restorable": len(backup["mysql_stale_scenes"]),
                                  "writes_performed": False}
    _write_json(REPORT_PATH, report)
    print(f"report={REPORT_PATH}")
    print(json.dumps({"cleanup": cleanup, "rollback_dry_run": report["rollback_dry_run"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
