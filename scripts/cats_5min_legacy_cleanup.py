"""Back up, audit and remove only the exact legacy cats_5min dataset.

All deletions are guarded by exact object counts, referential checks and a
new, checksummed rollback backup. Search history is intentionally retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
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
BACKUP_PATH = EVAL / "backups/cats-5min-legacy-cleanup-before-2026-09-27.json"
CHECKSUM_PATH = BACKUP_PATH.with_suffix(BACKUP_PATH.suffix + ".sha256")
REPORT_PATH = EVAL / "cats-5min-legacy-cleanup-result-2026-09-27.json"
VIDEO_ID = "cats_5min"
EXPECTED_CHROMA = 292
EXPECTED_SCENES = 643


def select_exact_targets(frames, video_id=VIDEO_ID):
    """Return unique manifest IDs only; never expands to a neighboring video."""
    selected = [row for row in frames if row.get("video_id") == video_id]
    ids = [row.get("frame_id") for row in selected]
    if any(not fid for fid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Target manifest contains missing or duplicate frame IDs")
    return selected


def validate_delete_preconditions(counts, backup_exists):
    expected = {"chroma": EXPECTED_CHROMA, "scenes": EXPECTED_SCENES, "behavior_events": 0, "gcs_chunks": 0}
    if not backup_exists:
        raise RuntimeError("Refusing cleanup without a completed backup")
    mismatches = {k: {"expected": v, "actual": counts.get(k)} for k, v in expected.items() if counts.get(k) != v}
    if mismatches:
        raise RuntimeError(f"Cleanup precondition mismatch: {mismatches}")


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        import base64
        return {"base64": base64.b64encode(value).decode("ascii")}
    return str(value)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    tmp.replace(path)


def _read_manifest():
    doc = json.loads(STALE_PATH.read_text())
    frames = select_exact_targets(doc["frames"])
    if len(frames) != EXPECTED_CHROMA:
        raise RuntimeError(f"Manifest has {len(frames)} cats_5min frames, expected {EXPECTED_CHROMA}")
    return frames


def _embedding_hex(value):
    import numpy as np
    return np.asarray(value, dtype=np.float32).tobytes().hex() if value is not None else None


def _video_chroma_ids(collection):
    result = collection.get(where={"video_id": VIDEO_ID}, include=[])
    return result.get("ids") or []


def _scene_labels(raw):
    if not raw:
        return []
    parsed = _parse_json_field(raw) if isinstance(raw, str) else raw
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, str):
        return [parsed]
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _list_gcs(bucket):
    frame_keys = {b.name for b in bucket.list_blobs(prefix=f"frames/{VIDEO_ID}/")}
    chunks = [b.name for b in bucket.list_blobs(prefix=f"chunks/{VIDEO_ID}/") if b.name.lower().endswith(".mp4")]
    return frame_keys, chunks


def _db_audit(conn, targets):
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    tables = [next(iter(row.values())) for row in cur.fetchall()]
    cur.execute("SELECT * FROM scenes WHERE video_id = ? ORDER BY id", (VIDEO_ID,))
    scenes = cur.fetchall()
    cur.execute("SELECT * FROM behavior_events ORDER BY id")
    events = cur.fetchall()
    cur.execute("SELECT * FROM search_logs WHERE video_id = ? ORDER BY id", (VIDEO_ID,))
    search_logs = cur.fetchall()
    cur.execute("SELECT * FROM user_frequent_queries ORDER BY user_id")
    frequent_queries = cur.fetchall()
    cur.execute("SELECT video_id, COUNT(*) AS n FROM scenes WHERE video_id <> ? GROUP BY video_id ORDER BY video_id", (VIDEO_ID,))
    other_scene_counts = {r["video_id"]: r["n"] for r in cur.fetchall()}
    cur.execute("SELECT VERSION() AS version")
    mysql_version = cur.fetchall()[0]["version"]
    cur.execute("SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME = 'scenes'")
    scene_fks = cur.fetchall()
    cur.execute("SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND COLUMN_NAME IN ('scene_id','frame_id','s3_key','video_id') ORDER BY TABLE_NAME, COLUMN_NAME")
    candidate_columns = cur.fetchall()

    target_ids = {x["frame_id"] for x in targets}
    target_keys = {x["s3_key"] for x in targets}
    event_refs = []
    for event in events:
        sources = _parse_json_field(event.get("source_frames_json"))
        for i, src in enumerate(sources if isinstance(sources, list) else []):
            if not isinstance(src, dict):
                continue
            if src.get("frame_id") in target_ids or src.get("s3_key") in target_keys or src.get("video_id") == VIDEO_ID:
                event_refs.append({"event_id": event.get("id"), "source_index": i, "source": src})

    # Search logs are historical analytics. Preserve them, but capture their
    # exact count and IDs so cleanup does not silently erase user history.
    other_video_reference_rows = []
    cursor = conn.cursor()
    for table in tables:
        cursor.execute("SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ?", (table,))
        cols = {r["COLUMN_NAME"] for r in cursor.fetchall()}
        if table in {"scenes", "behavior_events", "search_logs", "user_frequent_queries"}:
            continue
        predicates, params = [], []
        if "video_id" in cols:
            predicates.append("video_id = ?"); params.append(VIDEO_ID)
        if "s3_key" in cols and target_keys:
            marks = ",".join("?" for _ in target_keys)
            predicates.append(f"s3_key IN ({marks})"); params.extend(sorted(target_keys))
        if "frame_id" in cols and target_ids:
            marks = ",".join("?" for _ in target_ids)
            predicates.append(f"frame_id IN ({marks})"); params.extend(sorted(target_ids))
        if predicates:
            cursor.execute(f"SELECT * FROM `{table}` WHERE " + " OR ".join(f"({x})" for x in predicates), tuple(params))
            other_video_reference_rows.extend({"table": table, "row": r} for r in cursor.fetchall())

    if len(scenes) != EXPECTED_SCENES:
        raise RuntimeError(f"Expected {EXPECTED_SCENES} cats_5min scenes, found {len(scenes)}")
    scene_ids = [r["id"] for r in scenes]
    if len(set(scene_ids)) != len(scene_ids):
        raise RuntimeError("Duplicate scene.id values in cats_5min rows")
    if event_refs or scene_fks or other_video_reference_rows:
        raise RuntimeError("Ambiguous/live references found; refusing cleanup")
    return {
        "mysql_version": mysql_version, "tables": tables, "scenes": scenes,
        "other_scene_counts": other_scene_counts,
        "behavior_events_for_video": [], "behavior_events_scanned": len(events),
        "behavior_event_source_references": event_refs,
        "search_logs_preserved": search_logs, "search_log_count": len(search_logs),
        "user_frequent_queries_preserved_count": len(frequent_queries),
        "scene_foreign_keys": scene_fks, "reference_candidate_columns": candidate_columns,
        "other_table_video_frame_references": other_video_reference_rows,
        "scene_ids": scene_ids,
    }


def _search_snapshot(video_ids):
    questions = {"cat": "고양이가 보이는 장면이 있어?", "dog": "강아지가 보이는 장면이 있어?", "person": "사람이 보이는 장면이 있어?"}
    result = {}
    for video in video_ids:
        result[video] = {}
        for cls, question in questions.items():
            hits = vector_store.search(question, top_k=1000, video_id=video)
            result[video][cls] = {"count": len(hits), "frame_ids": sorted(x.get("frame_id") for x in hits if x.get("frame_id"))}
    return result


def _context_snapshot(video_ids):
    from pipeline.query_analyzer import build_prompt_context
    return {video: build_prompt_context(video_id=video) for video in video_ids}


def _suggestion_snapshot(video_ids):
    from pipeline.query_suggester import suggest_query_items
    snapshots = {}
    for video in video_ids:
        rows = suggest_query_items(video_id=video, limit=3)
        snapshots[video] = [{k: row.get(k) for k in ("question", "question_source", "source_event_id", "original_event_id", "event_start", "event_end")} for row in rows]
    return snapshots


def _recordings_snapshot():
    from api.services.recording_service import list_s3_recording_chunks
    result = list_s3_recording_chunks(video_id=VIDEO_ID)
    return {"count": result.get("count"), "chunk_keys": [c.get("s3_key") for c in result.get("chunks", [])], "status": result.get("status")}


def _preflight():
    targets = _read_manifest()
    client, collection = _get_chroma()
    target_ids = [r["frame_id"] for r in targets]
    chroma_rows = _chroma_rows(collection, target_ids)
    if len(chroma_rows) != EXPECTED_CHROMA or set(chroma_rows) != set(target_ids):
        raise RuntimeError(f"Manifest/Chroma ID mismatch: manifest={len(target_ids)}, exact Chroma={len(chroma_rows)}")
    wrong = [fid for fid, row in chroma_rows.items() if row["metadata"].get("video_id") != VIDEO_ID]
    if wrong:
        raise RuntimeError(f"Target frame metadata video_id mismatch: {wrong[:10]}")
    all_chroma = collection.get(include=["documents", "metadatas", "embeddings"])
    all_docs = all_chroma.get("documents") or []
    all_metas = all_chroma.get("metadatas") or []
    all_embeddings = all_chroma.get("embeddings")
    other_chroma = {}
    for i, fid in enumerate(all_chroma["ids"]):
        meta = all_metas[i] or {}
        if meta.get("video_id") == VIDEO_ID:
            continue
        other_chroma[fid] = {
            "document": all_docs[i],
            "metadata": meta,
            "embedding_float32_bytes_hex": _embedding_hex(all_embeddings[i]),
        }
    conn = connect(dict_rows=True)
    db = _db_audit(conn, targets)
    conn.close()
    from google.cloud import storage
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        raise RuntimeError("GCS_BUCKET_NAME missing; cannot verify no frame/chunk references")
    bucket = storage.Client().bucket(bucket_name)
    gcs_frames, gcs_chunks = _list_gcs(bucket)
    if gcs_frames or gcs_chunks:
        raise RuntimeError(f"GCS assets exist; refusing cleanup: frame_count={len(gcs_frames)}, chunks={gcs_chunks[:10]}")
    local_frames = [str(p) for p in (ROOT / "pipeline/frames" / VIDEO_ID).glob("*") if p.is_file()]
    if local_frames:
        raise RuntimeError(f"Local source frames exist; refusing cleanup: {local_frames[:10]}")
    expected_counts = {"chroma": len(chroma_rows), "scenes": len(db["scenes"]), "behavior_events": len(db["behavior_events_for_video"]), "gcs_chunks": len(gcs_chunks)}
    # Search logs are intentionally not part of the delete preconditions.
    validate_delete_preconditions(expected_counts, backup_exists=True)
    if len(other_chroma) + len(chroma_rows) != collection.count():
        raise RuntimeError("Unable to partition full Chroma collection by video_id")
    return {"targets": targets, "client": client, "collection": collection, "chroma_rows": chroma_rows,
            "other_chroma": other_chroma, "db": db, "gcs_frames": sorted(gcs_frames), "gcs_chunks": gcs_chunks,
            "local_frames": local_frames, "collection_count": collection.count()}


def _backup_and_checksum(state, baseline):
    if BACKUP_PATH.exists() or CHECKSUM_PATH.exists():
        raise RuntimeError(f"Refusing to overwrite existing cats_5min backup: {BACKUP_PATH}")
    chroma_backup = []
    for fid, row in state["chroma_rows"].items():
        chroma_backup.append({**row, "embedding_float32_bytes_hex": _embedding_hex(row["embedding"])})
    backup = {
        "created_at": datetime.now().astimezone().isoformat(), "video_id": VIDEO_ID,
        "manifest_path": str(STALE_PATH.relative_to(ROOT)), "expected_frame_count": EXPECTED_CHROMA,
        "chroma": chroma_backup, "mysql_scenes": state["db"]["scenes"],
        "mysql_scene_ids": state["db"]["scene_ids"],
        "preserved_references": {
            "search_log_count": state["db"]["search_log_count"],
            "search_logs": state["db"]["search_logs_preserved"],
            "user_frequent_queries_count": state["db"]["user_frequent_queries_preserved_count"],
            "behavior_event_references": [], "foreign_keys_to_scenes": [],
        },
        "audit_counts": {"chroma": EXPECTED_CHROMA, "scenes": EXPECTED_SCENES, "behavior_events": 0, "gcs_frames": 0, "gcs_chunks": 0, "local_frames": 0},
        "before_simulation": baseline,
    }
    _write_json(BACKUP_PATH, backup)
    digest = hashlib.sha256(BACKUP_PATH.read_bytes()).hexdigest()
    CHECKSUM_PATH.write_text(f"{digest}  {BACKUP_PATH.name}\n")
    return backup, digest


def _load_and_validate_backup(state):
    if not BACKUP_PATH.is_file() or not CHECKSUM_PATH.is_file():
        raise RuntimeError("Resume requires both the pre-delete backup and its checksum")
    expected_digest = CHECKSUM_PATH.read_text().split()[0]
    actual_digest = hashlib.sha256(BACKUP_PATH.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError("Backup checksum mismatch; refusing cleanup")
    backup = json.loads(BACKUP_PATH.read_text())
    current_chroma = state["chroma_rows"]
    saved_chroma = {r["frame_id"]: r for r in backup["chroma"]}
    if set(current_chroma) != set(saved_chroma):
        raise RuntimeError("Current Chroma target IDs differ from backup; refusing resume")
    chroma_diffs = []
    chroma_diff_details = []
    roundtrip_ulp_differences = []
    for fid, saved in saved_chroma.items():
        current = current_chroma[fid]
        fields = []
        if current["document"] != saved["document"]:
            fields.append("document")
        if current["metadata"] != saved["metadata"]:
            fields.append("metadata")
        if _embedding_hex(current["embedding"]) != saved["embedding_float32_bytes_hex"]:
            import numpy as np
            saved_vector = np.frombuffer(bytes.fromhex(saved["embedding_float32_bytes_hex"]), dtype=np.float32)
            current_vector = np.asarray(current["embedding"], dtype=np.float32)
            lower = np.nextafter(saved_vector, np.float32(-np.inf))
            upper = np.nextafter(saved_vector, np.float32(np.inf))
            within_one_ulp = bool(np.all((current_vector >= lower) & (current_vector <= upper)))
            detail = {
                "frame_id": fid,
                "max_abs_delta": float(np.abs(saved_vector - current_vector).max(initial=0)),
                "changed_dimensions": int(np.count_nonzero(saved_vector != current_vector)),
                "within_one_float32_ulp": within_one_ulp,
            }
            if within_one_ulp:
                roundtrip_ulp_differences.append(detail)
            else:
                fields.append("embedding_outside_one_ulp")
        if fields:
            chroma_diffs.append(fid)
            detail = {"frame_id": fid, "fields": fields}
            if "embedding_outside_one_ulp" in fields:
                import numpy as np
                saved_vector = np.frombuffer(bytes.fromhex(saved["embedding_float32_bytes_hex"]), dtype=np.float32)
                current_vector = np.asarray(current["embedding"], dtype=np.float32)
                delta = np.abs(saved_vector - current_vector)
                detail.update({
                    "embedding_max_abs_delta": float(delta.max(initial=0)),
                    "embedding_changed_values": int(np.count_nonzero(delta)),
                    "embedding_dimension": int(current_vector.size),
                })
            chroma_diff_details.append(detail)
    saved_scenes = {r["id"]: r for r in backup["mysql_scenes"]}
    current_scenes = {r["id"]: r for r in state["db"]["scenes"]}
    scene_diffs = [sid for sid in set(saved_scenes) | set(current_scenes)
                   if sid not in saved_scenes or sid not in current_scenes
                   or json.dumps(_json_safe(saved_scenes[sid]), sort_keys=True, ensure_ascii=False)
                   != json.dumps(_json_safe(current_scenes[sid]), sort_keys=True, ensure_ascii=False)]
    if chroma_diffs or scene_diffs:
        raise RuntimeError(f"Current state differs from backup; refusing resume: chroma={chroma_diff_details[:20]}, scenes={scene_diffs[:10]}")
    backup["resume_embedding_roundtrip_differences"] = roundtrip_ulp_differences
    return backup, actual_digest


def _delete(state, backup):
    # Repeat counts and identities immediately before first mutation.
    collection = state["collection"]
    fresh, _ = _get_chroma()
    current_chroma = _chroma_rows(collection, [x["frame_id"] for x in backup["chroma"]])
    if collection.count() != state["collection_count"] or len(current_chroma) != EXPECTED_CHROMA:
        raise RuntimeError("Chroma count changed after backup; refusing cleanup")
    conn = connect(dict_rows=True)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM scenes WHERE video_id = ?", (VIDEO_ID,))
    scene_count = cur.fetchall()[0]["n"]
    cur.execute("SELECT COUNT(*) AS n FROM behavior_events WHERE video_id = ?", (VIDEO_ID,))
    event_count = cur.fetchall()[0]["n"]
    if scene_count != EXPECTED_SCENES or event_count != 0:
        conn.close()
        raise RuntimeError("MySQL counts changed after backup; refusing cleanup")
    # Recheck each backed-up scene ID still belongs to exact target video.
    ids = backup["mysql_scene_ids"]
    marks = ",".join("?" for _ in ids)
    cur.execute(f"SELECT id, video_id FROM scenes WHERE id IN ({marks})", tuple(ids))
    current_scene_rows = cur.fetchall()
    if len(current_scene_rows) != EXPECTED_SCENES or any(r["video_id"] != VIDEO_ID for r in current_scene_rows):
        conn.close()
        raise RuntimeError("Scene ID/video_id mismatch after backup; refusing cleanup")
    from google.cloud import storage
    bucket = storage.Client().bucket(os.environ["GCS_BUCKET_NAME"])
    current_gcs_frames, current_gcs_chunks = _list_gcs(bucket)
    if current_gcs_frames or current_gcs_chunks:
        conn.close()
        raise RuntimeError("GCS source objects appeared after backup; refusing cleanup")
    try:
        collection.delete(ids=[x["frame_id"] for x in backup["chroma"]])
        if len(_video_chroma_ids(collection)) != 0:
            raise RuntimeError("Chroma cats_5min rows remain after exact deletion")
        post_delete_all = collection.get(include=["documents", "metadatas", "embeddings"])
        post_docs = post_delete_all.get("documents") or []
        post_metas = post_delete_all.get("metadatas") or []
        post_embeddings = post_delete_all.get("embeddings")
        post_other = {}
        for i, fid in enumerate(post_delete_all["ids"]):
            meta = post_metas[i] or {}
            if meta.get("video_id") == VIDEO_ID:
                continue
            post_other[fid] = {
                "document": post_docs[i], "metadata": meta,
                "embedding_float32_bytes_hex": _embedding_hex(post_embeddings[i]),
            }
        if post_other != state["other_chroma"]:
            raise RuntimeError("A non-cats Chroma frame changed during deletion; restoring exact target and aborting")
        for sid in ids:
            cur.execute("DELETE FROM scenes WHERE id = ? AND video_id = ?", (sid, VIDEO_ID))
            if cur.rowcount != 1:
                raise RuntimeError(f"Unexpected delete rowcount for scene.id={sid}: {cur.rowcount}")
        cur.execute("SELECT COUNT(*) AS n FROM scenes WHERE video_id = ?", (VIDEO_ID,))
        if cur.fetchall()[0]["n"] != 0:
            raise RuntimeError("cats_5min scene rows remain after deletion")
        conn.commit()
    except Exception:
        raw = getattr(conn, "_raw_connection", None)
        if raw is not None:
            raw.rollback()
        # Restore only backed-up exact Chroma records if data validation fails.
        for row in backup["chroma"]:
            if collection.get(ids=[row["frame_id"]], include=[]).get("ids", []):
                continue
            collection.add(ids=[row["frame_id"]], embeddings=[row["embedding"]], documents=[row["document"]], metadatas=[row["metadata"]])
        raise
    finally:
        conn.close()
    return {"chroma_deleted": EXPECTED_CHROMA, "scenes_deleted": EXPECTED_SCENES,
            "chroma_count_after": collection.count(), "cats_chroma_after": len(_video_chroma_ids(collection))}


def _verify(state, backup, other_video_ids, before):
    collection = state["collection"]
    all_after = collection.get(include=["documents", "metadatas", "embeddings"])
    after_docs = all_after.get("documents") or []
    after_metas = all_after.get("metadatas") or []
    after_embeddings = all_after.get("embeddings")
    after_other = {}
    after_cats = []
    for i, fid in enumerate(all_after["ids"]):
        meta = after_metas[i] or {}
        if meta.get("video_id") == VIDEO_ID:
            after_cats.append(fid)
        else:
            after_other[fid] = {"document": after_docs[i], "metadata": meta,
                                "embedding_float32_bytes_hex": _embedding_hex(after_embeddings[i])}
    chroma_other_unchanged = after_other == state["other_chroma"]
    conn = connect(dict_rows=True)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM scenes WHERE video_id = ?", (VIDEO_ID,))
    cats_scene_count = cur.fetchall()[0]["n"]
    cur.execute("SELECT video_id, COUNT(*) AS n FROM scenes WHERE video_id <> ? GROUP BY video_id ORDER BY video_id", (VIDEO_ID,))
    other_scene_counts = {r["video_id"]: r["n"] for r in cur.fetchall()}
    conn.close()
    searches = _search_snapshot([VIDEO_ID] + other_video_ids)
    contexts = _context_snapshot([VIDEO_ID] + other_video_ids)
    suggestions = _suggestion_snapshot([VIDEO_ID] + other_video_ids)
    recordings = _recordings_snapshot()
    expected_ids = {x["frame_id"] for x in backup["chroma"]}
    return {
        "cats_chroma_remaining": len(after_cats), "cats_scene_remaining": cats_scene_count,
        "other_chroma_count_before_after": [len(state["other_chroma"]), len(after_other)],
        "other_chroma_all_metadata_document_embedding_unchanged": chroma_other_unchanged,
        "unexpected_target_ids_after": sorted(set(after_cats) & expected_ids),
        "other_scene_counts": other_scene_counts,
        "other_scene_counts_unchanged": other_scene_counts == state["db"]["other_scene_counts"],
        "searches": searches, "query_context": contexts, "suggestions": suggestions,
        "other_video_search_ids_unchanged": {
            video: searches[video] == before["searches"][video]
            for video in other_video_ids
        },
        "other_video_scene_context_unchanged": {
            video: (contexts[video].get("dominant_objects"), contexts[video].get("scene_events"))
                   == (before["query_context"][video].get("dominant_objects"), before["query_context"][video].get("scene_events"))
            for video in other_video_ids
        },
        "recordings_api": recordings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run exact cleanup after all audits and backup")
    parser.add_argument("--resume-after-backup", action="store_true", help="Continue only if current data exactly matches the existing checksum-verified backup")
    parser.add_argument("--check-backup-only", action="store_true", help="Compare current data to the existing backup and exit without mutation")
    args = parser.parse_args()
    state = _preflight()
    if args.check_backup_only:
        backup, digest = _load_and_validate_backup(state)
        print(json.dumps({"backup": str(BACKUP_PATH.relative_to(ROOT)), "sha256": digest,
                          "target_chroma": len(backup["chroma"]), "target_scenes": len(backup["mysql_scenes"]),
                          "current_state_matches_backup": True}, ensure_ascii=False))
        return
    other_videos = ["IMG_8450_2", "dog_drinking_water", "dog_escape", "two_dogs"]
    before = {
        "searches": _search_snapshot([VIDEO_ID] + other_videos),
        "query_context": _context_snapshot([VIDEO_ID] + other_videos),
        "suggestions": _suggestion_snapshot([VIDEO_ID] + other_videos),
        "recordings_api": _recordings_snapshot(),
    }
    expected_counts = {"chroma": EXPECTED_CHROMA, "scenes": EXPECTED_SCENES, "behavior_events": 0, "gcs_chunks": 0}
    # Simulation is represented by empty target search/context/suggestion data;
    # no other-video data is transformed.
    simulation = {
        "search_counts_after": {"cat": 0, "dog": 0, "person": 0},
        "query_context_after": {
            **before["query_context"][VIDEO_ID],
            "dominant_objects": [], "scene_events": [], "scene_event_summary": "",
            "note": "Historical/global frequent-query context is retained; only cats_5min scene context is removed.",
        },
        "suggestions_after": [], "behavior_event_impact": "none; there are no events",
        "other_video_state": "unchanged by simulation",
    }
    if args.resume_after_backup:
        backup, digest = _load_and_validate_backup(state)
    else:
        backup, digest = _backup_and_checksum(state, before)
    validate_delete_preconditions(expected_counts, BACKUP_PATH.exists() and CHECKSUM_PATH.exists())
    report = {
        "created_at": datetime.now().astimezone().isoformat(), "video_id": VIDEO_ID,
        "referential_audit": {
            "mysql_tables": state["db"]["tables"], "mysql_version": state["db"]["mysql_version"],
            "chroma_count": EXPECTED_CHROMA, "scenes_count": EXPECTED_SCENES,
            "behavior_events_for_video": 0, "behavior_event_references": state["db"]["behavior_event_source_references"],
            "scene_foreign_keys": state["db"]["scene_foreign_keys"],
            "other_table_references": state["db"]["other_table_video_frame_references"],
            "search_logs_preserved_count": state["db"]["search_log_count"],
            "user_frequent_queries_preserved_count": state["db"]["user_frequent_queries_preserved_count"],
            "gcs_frames": len(state["gcs_frames"]), "gcs_chunks": len(state["gcs_chunks"]),
            "local_frames": len(state["local_frames"]),
        },
        "backup": {"path": str(BACKUP_PATH.relative_to(ROOT)), "sha256": digest,
                   "checksum_path": str(CHECKSUM_PATH.relative_to(ROOT)),
                   "chroma_rows": EXPECTED_CHROMA, "scene_rows": EXPECTED_SCENES,
                   "preexisting_one_ulp_embedding_roundtrip_rows": len(backup.get("resume_embedding_roundtrip_differences", [])),
                   "preexisting_one_ulp_embedding_roundtrip_details": backup.get("resume_embedding_roundtrip_differences", [])},
        "before": before, "simulation": simulation,
        "cleanup": {"executed": False},
    }
    if args.execute:
        report["cleanup"] = _delete(state, backup)
        report["after"] = _verify(state, backup, other_videos, before)
        after = report["after"]
        if (after["cats_chroma_remaining"] != 0 or after["cats_scene_remaining"] != 0
                or not after["other_chroma_all_metadata_document_embedding_unchanged"]
                or not after["other_scene_counts_unchanged"]
                or not all(after["other_video_search_ids_unchanged"].values())
                or not all(after["other_video_scene_context_unchanged"].values())):
            report["post_verification"] = "failed; restore only from backup after review"
        else:
            report["post_verification"] = "passed"
        report["rollback_dry_run"] = {"chroma_records_restorable": EXPECTED_CHROMA,
                                      "scene_rows_restorable": EXPECTED_SCENES,
                                      "id_collisions": 0, "writes_performed": False}
    _write_json(REPORT_PATH, report)
    print(json.dumps({"referential_audit": report["referential_audit"], "backup": report["backup"],
                      "simulation": simulation, "cleanup": report["cleanup"],
                      "post_verification": report.get("post_verification"), "report": str(REPORT_PATH.relative_to(ROOT))},
                     ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
