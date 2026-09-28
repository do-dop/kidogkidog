"""Backup, apply, verify, or roll back the fixed 15-frame CLIP species canary.

The target set is intentionally hard-bound to the visual-review canary list and
its intersection with the exact-frame eligible/shadow manifests. This script
never scans or writes the remaining eligible/ineligible frames.
"""

import argparse
import base64
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
# Reuse the exact weight downloaded and used for the successful 240-frame run,
# if no project-specific model override is configured. This is process-local.
_shadow_weight = Path("/tmp/yolo_clip_species_eval/yolov8s.pt")
if not os.getenv("YOLO_MODEL") and _shadow_weight.is_file():
    os.environ["YOLO_MODEL"] = str(_shadow_weight)

from db.connection import connect
from pipeline import vector_store
from pipeline.species_resolver import resolve_detection_result
from pipeline.yolo_detector import detect_objects_with_details


EVAL = ROOT / "docs/evaluations"
BACKUP_DIR = EVAL / "backups"
BACKUP_PATH = BACKUP_DIR / "yolo-clip-canary-before-2026-09-27.json"
RESULT_PATH = EVAL / "yolo-clip-canary-backfill-result-2026-09-27.json"
MIGRATION_LOG = EVAL / "yolo-clip-canary-migration-2026-09-27.json"
CANARY_PATH = EVAL / "yolo-backfill-canary-visual-review-2026-09-27.json"
ELIGIBLE_PATH = EVAL / "yolo-backfill-eligible-2026-09-27.json"
SHADOW_PATH = EVAL / "yolo-clip-species-shadow-2026-09-27.json"


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    return str(value)


def _atomic_json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    temp.replace(path)


def _json_safe(value):
    """Normalize DB values exactly as they will look after backup reload."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))


def _deep_equal(left, right, tolerance=1e-6):
    """Compare JSON-shaped values while allowing MySQL JSON float round-trip."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_deep_equal(left[k], right[k], tolerance) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_deep_equal(a, b, tolerance) for a, b in zip(left, right))
    return left == right


def _chroma_matches_backup(current, saved):
    """Accept only the empty sentinels required to undo Chroma's merge update."""
    metadata = dict(current["metadata"])
    original = dict(saved["metadata"])
    for key in ("object_detections_json", "species_resolution_json"):
        if key not in original and metadata.get(key) == "":
            metadata.pop(key)
    return (
        metadata == original
        and current["document"] == saved["document"]
        and current["embedding_float32_bytes_hex"] == saved["embedding_float32_bytes_hex"]
    )


def _load_targets():
    canary_doc = json.loads(CANARY_PATH.read_text())
    eligible_doc = json.loads(ELIGIBLE_PATH.read_text())
    shadow_doc = json.loads(SHADOW_PATH.read_text())
    canary = canary_doc["frames"]
    eligible = {r["frame_id"]: r for r in eligible_doc["frames"]}
    shadow = {r["frame_id"]: r for r in shadow_doc["frames"]}
    ids = [r["frame_id"] for r in canary]
    if len(ids) != 15 or len(set(ids)) != 15:
        raise RuntimeError(f"Expected the original 15 unique reviewed canaries; got {len(ids)}")
    absent = [fid for fid in ids if fid not in eligible or fid not in shadow]
    if absent:
        raise RuntimeError(f"Canary is not in both eligible and shadow manifests: {absent}")
    if shadow_doc.get("counts", {}).get("processed") != 240:
        raise RuntimeError("Shadow result does not record all 240 processed frames")
    if shadow_doc.get("counts", {}).get("failures") != 0:
        raise RuntimeError("Shadow run contains failures")
    return canary, eligible, shadow


def _get_chroma():
    import chromadb
    if vector_store.CHROMA_HOST:
        client = chromadb.HttpClient(
            host=vector_store.CHROMA_HOST,
            port=vector_store.CHROMA_PORT,
            ssl=vector_store.CHROMA_SSL,
        )
    else:
        client = chromadb.PersistentClient(path=str(vector_store.CHROMA_DIR))
    collection = client.get_collection(vector_store.CHROMA_COLLECTION)
    # The metadata-only updater below reuses the already verified collection.
    vector_store._client = client
    vector_store._collection = collection
    return client, collection


def _chroma_rows(collection, frame_ids):
    result = collection.get(ids=frame_ids, include=["documents", "metadatas", "embeddings"])
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    rows = {}
    for i, fid in enumerate(result.get("ids") or []):
        emb = embeddings[i] if embeddings is not None else None
        emb_list = emb.tolist() if hasattr(emb, "tolist") else emb
        emb_bytes = None
        if emb is not None:
            import numpy as np
            emb_bytes = np.asarray(emb, dtype=np.float32).tobytes().hex()
        rows[fid] = {
            "frame_id": fid,
            "document": docs[i] if i < len(docs) else None,
            "embedding": emb_list,
            "embedding_float32_bytes_hex": emb_bytes,
            "embedding_present": emb is not None,
            "metadata": dict(metas[i] or {}) if i < len(metas) else {},
        }
    return rows


def _mysql_schema(conn):
    cur = conn.cursor()
    cur.execute("SELECT VERSION() AS version")
    version = cur.fetchall()[0]["version"]
    cur.execute("SHOW COLUMNS FROM scenes")
    columns = cur.fetchall()
    cur.execute("SHOW CREATE TABLE scenes")
    create_row = cur.fetchall()[0]
    return {
        "version": version,
        "columns": columns,
        "column_names": [c["Field"] for c in columns],
        "show_create_table": create_row.get("Create Table") or create_row,
    }


def _scene_rows(conn, canary):
    cur = conn.cursor()
    matches = {}
    duplicate_keys = []
    missing_keys = []
    for frame in canary:
        cur.execute(
            "SELECT * FROM scenes WHERE video_id = ? AND s3_key = ? ORDER BY id",
            (frame["video_id"], frame["s3_key"]),
        )
        rows = cur.fetchall()
        matches[frame["frame_id"]] = rows
        if len(rows) > 1:
            duplicate_keys.append({"frame_id": frame["frame_id"], "scene_ids": [r["id"] for r in rows]})
        elif not rows:
            missing_keys.append(frame["frame_id"])
    return matches, duplicate_keys, missing_keys


def _parse_json_field(raw):
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _matching_event_sources(source_frames, event_video_id, targets_by_id, targets_by_key):
    if not isinstance(source_frames, list):
        return [], False
    matches = []
    ambiguous = False
    for index, source in enumerate(source_frames):
        if not isinstance(source, dict):
            continue
        fid = source.get("frame_id")
        if fid:
            if fid in targets_by_id:
                matches.append((index, fid))
            continue
        key = (source.get("video_id") or event_video_id, source.get("s3_key"))
        canary_matches = targets_by_key.get(key, [])
        if len(canary_matches) > 1:
            ambiguous = True
        elif canary_matches:
            matches.append((index, canary_matches[0]))
    if len({fid for _, fid in matches}) != len(matches):
        ambiguous = True
    return matches, ambiguous


def _event_rows(conn, canary):
    videos = sorted({r["video_id"] for r in canary})
    marks = ",".join("?" for _ in videos)
    cur = conn.cursor()
    cur.execute(
        f"SELECT * FROM behavior_events WHERE video_id IN ({marks}) ORDER BY id",
        tuple(videos),
    )
    targets_by_id = {r["frame_id"]: r for r in canary}
    targets_by_key = {}
    for r in canary:
        targets_by_key.setdefault((r["video_id"], r["s3_key"]), []).append(r["frame_id"])
    snapshots = []
    for event in cur.fetchall():
        source_raw = event.get("source_frames_json")
        source_parsed = _parse_json_field(source_raw)
        matches, ambiguous = _matching_event_sources(
            source_parsed, event["video_id"], targets_by_id, targets_by_key
        )
        if not matches and not ambiguous:
            continue
        snapshots.append({
            "row": event,
            "source_frames_parsed": source_parsed,
            "matched_sources": matches,
            "ambiguous": ambiguous,
        })
    return snapshots


def _resolve_canary_outputs(canary, eligible, shadow):
    """Re-run current YOLO+CLIP code and fail closed on any shadow mismatch."""
    from google.cloud import storage
    bucket_name = json.loads(ELIGIBLE_PATH.read_text())["gcs_bucket"]
    bucket = storage.Client().bucket(bucket_name)
    temp = Path(tempfile.gettempdir()) / "yolo-clip-canary-verify"
    temp.mkdir(parents=True, exist_ok=True)
    resolved = {}
    failures = []
    for frame in canary:
        shadow_row = shadow[frame["frame_id"]]
        path = Path(frame.get("image_file") or "")
        if not path.is_file():
            path = temp / (hashlib.sha1(frame["frame_id"].encode()).hexdigest() + ".jpg")
            if not path.is_file() or not path.stat().st_size:
                bucket.blob(frame["s3_key"]).download_to_filename(str(path))
        try:
            raw = detect_objects_with_details(path)
            final = resolve_detection_result(path, raw, enabled=True)
        except Exception as exc:
            failures.append({"frame_id": frame["frame_id"], "error": str(exc)})
            continue
        if _canonical_detections(raw["object_detections"]) != _canonical_detections(shadow_row["raw_yolo_detections"]):
            failures.append({"frame_id": frame["frame_id"], "error": "raw_yolo_differs_from_shadow"})
            continue
        if _canonical_resolutions(final["species_resolution"]) != _canonical_resolutions(shadow_row["species_resolution"]):
            failures.append({"frame_id": frame["frame_id"], "error": "species_resolution_differs_from_shadow"})
            continue
        if final["object_labels"] != shadow_row["final_object_labels"]:
            failures.append({"frame_id": frame["frame_id"], "error": "object_labels_differs_from_shadow"})
            continue
        resolved[frame["frame_id"]] = final
    if failures or len(resolved) != 15:
        raise RuntimeError(f"Canary recomputation did not match shadow for all 15 frames: {failures}")
    return resolved


def _canonical_detections(detections):
    return sorted((
        item["label"], round(float(item["confidence"]), 5),
        tuple(round(float(x), 2) for x in item["bbox"]),
    ) for item in detections)


def _canonical_resolutions(resolutions):
    return sorted((
        item["yolo_label"], round(float(item["yolo_confidence"]), 5),
        tuple(round(float(x), 2) for x in item["bbox"]),
        item["clip_label"], item["resolved_label"],
        round(float(item["clip_cat_similarity"]), 5),
        round(float(item["clip_dog_similarity"]), 5),
    ) for item in resolutions)


def _prepare_backup(canary, resolved):
    client, collection = _get_chroma()
    ids = [r["frame_id"] for r in canary]
    chroma = _chroma_rows(collection, ids)
    missing = [fid for fid in ids if fid not in chroma]
    if missing:
        raise RuntimeError(f"Canary frames missing from Chroma: {missing}")
    for frame in canary:
        metadata = chroma[frame["frame_id"]]["metadata"]
        if metadata.get("video_id") != frame["video_id"] or metadata.get("s3_key") != frame["s3_key"]:
            raise RuntimeError(f"Chroma identity mismatch for {frame['frame_id']}")

    conn = connect(dict_rows=True)
    schema = _mysql_schema(conn)
    scene_matches, scene_duplicates, scene_missing = _scene_rows(conn, canary)
    events = _event_rows(conn, canary)
    conn.close()

    safe_ids = {
        fid for fid, rows in scene_matches.items() if len(rows) == 1
    }
    safe_ids -= {x["frame_id"] for x in scene_duplicates}
    safe_ids -= set(scene_missing)
    # Frame-level ambiguity excludes that frame from all store writes. Event-level
    # ambiguity will exclude only the ambiguous event row.
    blocked_events = [e["row"]["id"] for e in events if e["ambiguous"]]
    backup = {
        "backup_date": datetime.now().astimezone().isoformat(),
        "canary_frame_ids": ids,
        "eligible_and_shadow_verified": True,
        "safe_frame_ids": [fid for fid in ids if fid in safe_ids],
        "scene_duplicates": scene_duplicates,
        "scene_missing": scene_missing,
        "mysql_schema_before": schema,
        "chroma": list(chroma.values()),
        "scenes": [row for group in scene_matches.values() for row in group],
        "behavior_events": [e["row"] for e in events],
        "event_match_details": [
            {"event_id": e["row"]["id"], "matched_sources": e["matched_sources"], "ambiguous": e["ambiguous"]}
            for e in events
        ],
        "computed_metadata": {
            fid: resolved[fid] for fid in resolved
        },
    }
    return backup, client, collection, scene_matches, events, safe_ids


def backup():
    if BACKUP_PATH.exists():
        raise RuntimeError(
            f"Refusing to overwrite the original rollback backup: {BACKUP_PATH}. "
            "Move it aside only after confirming it is from an earlier completed canary run."
        )
    canary, eligible, shadow = _load_targets()
    resolved = _resolve_canary_outputs(canary, eligible, shadow)
    data, _, _, _, _, _ = _prepare_backup(canary, resolved)
    _atomic_json_write(BACKUP_PATH, data)
    print(f"Backup saved: {BACKUP_PATH}")
    print(f"safe scene mappings: {len(data['safe_frame_ids'])}/15; duplicate: {len(data['scene_duplicates'])}; missing: {len(data['scene_missing'])}")
    print(f"behavior event rows backed up: {len(data['behavior_events'])}")
    return data


def _migration_sql(name):
    path = ROOT / "db/migrations" / name
    sql = path.read_text()
    normalized = " ".join(sql.lower().split())
    if "alter table scenes" not in normalized or "add column" not in normalized or "drop " in normalized or "delete " in normalized:
        raise RuntimeError(f"Migration is not an additive scenes-column migration: {name}")
    return sql


def _apply_needed_migrations(conn, before_schema):
    before = list(before_schema["column_names"])
    applied = []
    cursor = conn.cursor()
    for field, migration in (
        ("object_detections_json", "001_add_scenes_object_detections_json.sql"),
        ("species_resolution_json", "002_add_scenes_species_resolution_json.sql"),
    ):
        if field in before:
            applied.append({"column": field, "result": "already_present_noop"})
            continue
        cursor.execute(_migration_sql(migration))
        applied.append({"column": field, "result": "added_nullable_json", "migration": migration})
        cursor.execute("SHOW COLUMNS FROM scenes")
        before = [row["Field"] for row in cursor.fetchall()]
    conn.commit()
    return applied


def _replace_matching_source_frames(event, target_by_id, target_by_key=None):
    source_raw = event.get("source_frames_json")
    frames = _parse_json_field(source_raw)
    if not isinstance(frames, list):
        return frames, [], False
    matched, ambiguous = _matching_event_sources(
        frames,
        event["video_id"],
        target_by_id,
        target_by_key or {},
    )
    if ambiguous:
        return frames, matched, True
    changed = json.loads(json.dumps(frames, ensure_ascii=False))
    for index, frame_id in matched:
        if frame_id not in target_by_id:
            continue
        result = target_by_id[frame_id]
        changed[index]["object_labels"] = result["object_labels"]
        changed[index]["object_detections"] = result["object_detections"]
        changed[index]["species_resolution"] = result["species_resolution"]
    return changed, matched, False


def apply():
    if not BACKUP_PATH.is_file():
        raise RuntimeError(f"Required pre-write backup not found: {BACKUP_PATH}; run --action backup first")
    canary, eligible, shadow = _load_targets()
    resolved = _resolve_canary_outputs(canary, eligible, shadow)
    backup_doc = json.loads(BACKUP_PATH.read_text())
    ids = [r["frame_id"] for r in canary]
    if backup_doc.get("canary_frame_ids") != ids:
        raise RuntimeError("Backup frame IDs do not exactly match the fixed canary set")
    if set(resolved) != set(ids):
        raise RuntimeError("Recomputed outputs do not cover all 15 canaries")

    before_chroma = {r["frame_id"]: r for r in backup_doc["chroma"]}
    before_scene = {r["id"]: r for r in backup_doc["scenes"]}
    safe_ids = set(backup_doc["safe_frame_ids"])
    if not safe_ids:
        raise RuntimeError("No canary has a unique scene mapping; refusing all writes")

    client, collection = _get_chroma()
    live_chroma = _chroma_rows(collection, ids)
    for fid in ids:
        old = before_chroma[fid]
        current = live_chroma.get(fid)
        if not current or not _chroma_matches_backup(current, old):
            raise RuntimeError(f"Chroma changed since backup for {fid}; refusing canary write")

    conn = connect(dict_rows=True)
    schema_now = _mysql_schema(conn)
    if schema_now["version"] != backup_doc["mysql_schema_before"]["version"]:
        conn.close()
        raise RuntimeError("MySQL server version differs from backed-up preflight")
    scene_matches, duplicates, missing = _scene_rows(conn, canary)
    if duplicates or missing or any(len(rows) != 1 for rows in scene_matches.values()):
        conn.close()
        raise RuntimeError(f"Scene mapping changed or ambiguous since backup: duplicates={duplicates}, missing={missing}")
    for frame in canary:
        row = scene_matches[frame["frame_id"]][0]
        saved = next((r for r in backup_doc["scenes"] if r["id"] == row["id"]), None)
        if not saved or any(_json_safe(row).get(key) != value for key, value in saved.items()):
            conn.close()
            raise RuntimeError(f"Scene row changed since backup: {row['id']}")

    # DDL only adds nullable JSON columns, after the schema snapshot/backup exists.
    # Use the live schema on every attempt: a prior interrupted attempt may
    # have completed additive DDL even when no data update was committed.
    migration_schema_at_start = schema_now
    migration_results = _apply_needed_migrations(conn, migration_schema_at_start)
    migration_after = _mysql_schema(conn)
    _atomic_json_write(MIGRATION_LOG, {
        "mysql_version": migration_after["version"],
        "before": backup_doc["mysql_schema_before"],
        "schema_at_attempt_start": migration_schema_at_start,
        "applied": migration_results,
        "after": migration_after,
    })
    needed = {"object_detections_json", "species_resolution_json"}
    if not needed.issubset(set(migration_after["column_names"])):
        conn.close()
        raise RuntimeError("Required MySQL JSON columns are absent after migration")

    # Update only detection-related Chroma metadata via the existing helper.
    # If a later MySQL/verification step fails, restore both stores from backup.
    scene_id_by_frame = {}
    event_results = []
    target_by_id = {fid: resolved[fid] for fid in safe_ids}
    target_by_key = {}
    for frame in canary:
        if frame["frame_id"] in safe_ids:
            target_by_key.setdefault((frame["video_id"], frame["s3_key"]), []).append(frame["frame_id"])

    try:
        # Keep Chroma's ID/document/embedding intact: this helper only updates
        # detection-related metadata on the existing frame record.
        for frame in canary:
            fid = frame["frame_id"]
            if fid not in safe_ids:
                continue
            result = resolved[fid]
            vector_store.update_frame_detection_metadata(
                fid,
                result["object_labels"],
                result["object_detections"],
                species_resolution=result["species_resolution"],
            )

        cursor = conn.cursor()
        for frame in canary:
            fid = frame["frame_id"]
            if fid not in safe_ids:
                continue
            scene_id = scene_matches[fid][0]["id"]
            scene_id_by_frame[fid] = scene_id
            result = resolved[fid]
            labels_json = json.dumps(result["object_labels"], ensure_ascii=False)
            cursor.execute(
                "UPDATE scenes SET object_labels = ?, object_detections_json = ?, species_resolution_json = ? WHERE id = ?",
                (
                    labels_json,
                    vector_store._normalize_object_detections(result["object_detections"]),
                    vector_store._normalize_species_resolution(result["species_resolution"]),
                    scene_id,
                ),
            )
            if cursor.rowcount not in (0, 1):
                raise RuntimeError(f"Expected at most one scene row updated by id, got {cursor.rowcount}: {scene_id}")

        # Re-read candidate event rows after backup; event metadata must not
        # change since the snapshot except for source_frames_json.
        event_snapshots = {row["id"]: row for row in backup_doc["behavior_events"]}
        videos = sorted({r["video_id"] for r in canary})
        marks = ",".join("?" for _ in videos)
        cursor.execute(f"SELECT * FROM behavior_events WHERE video_id IN ({marks}) ORDER BY id", tuple(videos))
        for event in cursor.fetchall():
            before = event_snapshots.get(event["id"])
            if not before:
                continue
            live_other = {k: v for k, v in event.items() if k != "source_frames_json"}
            saved_other = {k: v for k, v in before.items() if k != "source_frames_json"}
            if _json_safe(live_other) != saved_other:
                raise RuntimeError(f"Behavior event changed since backup: {event['id']}")
            updated_frames, matches, ambiguous = _replace_matching_source_frames(event, target_by_id, target_by_key)
            if ambiguous:
                event_results.append({"event_id": event["id"], "result": "skipped_ambiguous"})
                continue
            if not matches:
                continue
            cursor.execute(
                "UPDATE behavior_events SET source_frames_json = ? WHERE id = ?",
                (json.dumps(updated_frames, ensure_ascii=False, separators=(",", ":")), event["id"]),
            )
            event_results.append({"event_id": event["id"], "result": "updated_source_frames_only", "matched_frame_ids": list(dict.fromkeys(fid for _, fid in matches))})
        conn.commit()
        conn.close()

        verification = _verify_after(canary, resolved, scene_id_by_frame, backup_doc, client, collection)
        if not (verification["chroma_all_pass"] and verification["scenes_all_pass"] and verification["behavior_events_all_pass"]):
            failure_path = EVAL / "yolo-clip-canary-verification-failure-2026-09-27.json"
            _atomic_json_write(failure_path, verification)
            failed_checks = {
                "chroma": [fid for fid, checks in verification["chroma"].items() if not all(checks.values())],
                "scenes": [fid for fid, checks in verification["scenes"].items() if not all(checks.values())],
                "behavior_events": [eid for eid, checks in verification["behavior_events"].items() if not all(v for k, v in checks.items() if k != "matched_sources")],
            }
            raise RuntimeError(f"Canary post-write verification failed ({failed_checks}); details: {failure_path}; rolling back from backup")
    except Exception:
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
        rollback()
        raise
    e2e = _run_search_checks(canary)
    suggestions = _run_question_checks(canary)
    result_doc = {
        "canary_frame_ids": ids,
        "updated_frame_count": len(ids),
        "scene_id_by_frame": scene_id_by_frame,
        "migration_results": migration_results,
        "event_results": event_results,
        "verification": verification,
        "search_checks": e2e,
        "suggestion_checks": suggestions,
        "storage_writes": {"chroma": True, "mysql_scenes": True, "behavior_event_source_frames": True},
    }
    _atomic_json_write(RESULT_PATH, result_doc)
    print(json.dumps({k: result_doc[k] for k in ("updated_frame_count", "migration_results", "event_results", "verification", "search_checks", "suggestion_checks")}, ensure_ascii=False, default=_json_default))
    return result_doc


def _verify_after(canary, resolved, scene_id_by_frame, backup_doc, client, collection):
    ids = [f["frame_id"] for f in canary]
    after = _chroma_rows(collection, ids)
    before = {x["frame_id"]: x for x in backup_doc["chroma"]}
    chroma_verification = {}
    for fid in ids:
        expected = resolved[fid]
        a = after[fid]
        b = before[fid]
        metadata = a["metadata"]
        preserved_metadata = {k: v for k, v in b["metadata"].items() if k not in {"object_labels", "object_detections_json", "species_resolution_json"}}
        chroma_verification[fid] = {
            "frame_id_same": a["frame_id"] == b["frame_id"],
            "embedding_bytes_same": a["embedding_float32_bytes_hex"] == b["embedding_float32_bytes_hex"],
            "document_same": a["document"] == b["document"],
            "other_metadata_same": {k: v for k, v in metadata.items() if k not in {"object_labels", "object_detections_json", "species_resolution_json"}} == preserved_metadata,
            "object_labels_same_as_expected": metadata.get("object_labels") == vector_store._normalize_object_labels(expected["object_labels"]),
            "object_detections_same_as_expected": vector_store._parse_object_detections(metadata) == expected["object_detections"],
            "species_resolution_same_as_expected": vector_store._parse_species_resolution(metadata) == expected["species_resolution"],
        }

    conn = connect(dict_rows=True)
    cursor = conn.cursor()
    scenes_verification = {}
    for frame in canary:
        fid = frame["frame_id"]
        sid = scene_id_by_frame[fid]
        cursor.execute("SELECT * FROM scenes WHERE id = ?", (sid,))
        rows = cursor.fetchall()
        if len(rows) != 1:
            scenes_verification[fid] = {"scene_row_count": len(rows), "all_checks_pass": False}
            continue
        row = rows[0]
        old = next(r for r in backup_doc["scenes"] if r["id"] == sid)
        exp = resolved[fid]
        other_same = _json_safe({k: row.get(k) for k in old if k not in {"object_labels", "object_detections_json", "species_resolution_json"}}) == {
            k: v for k, v in old.items() if k not in {"object_labels", "object_detections_json", "species_resolution_json"}
        }
        scenes_verification[fid] = {
            "scene_id_same": row["id"] == sid,
            "labels_same_as_expected": _parse_json_field(row.get("object_labels")) == exp["object_labels"],
            "detections_same_as_expected": _deep_equal(_parse_json_field(row.get("object_detections_json")), exp["object_detections"]),
            "species_same_as_expected": _deep_equal(_parse_json_field(row.get("species_resolution_json")), exp["species_resolution"]),
            "other_fields_same": other_same,
        }
        if not scenes_verification[fid]["detections_same_as_expected"]:
            scenes_verification[fid]["actual_detections"] = _parse_json_field(row.get("object_detections_json"))
            scenes_verification[fid]["expected_detections"] = exp["object_detections"]
        if not scenes_verification[fid]["species_same_as_expected"]:
            scenes_verification[fid]["actual_species_resolution"] = _parse_json_field(row.get("species_resolution_json"))
            scenes_verification[fid]["expected_species_resolution"] = exp["species_resolution"]

    # Compare events to the exact expected source-frame-only patch.
    event_verification = {}
    saved_events = {r["id"]: r for r in backup_doc["behavior_events"]}
    videos = sorted({f["video_id"] for f in canary})
    marks = ",".join("?" for _ in videos)
    cursor.execute(f"SELECT * FROM behavior_events WHERE video_id IN ({marks}) ORDER BY id", tuple(videos))
    target_by_id = {fid: resolved[fid] for fid in ids}
    for event in cursor.fetchall():
        before = saved_events.get(event["id"])
        if not before:
            continue
        target_by_key = {}
        for frame in canary:
            target_by_key.setdefault((frame["video_id"], frame["s3_key"]), []).append(frame["frame_id"])
        changed, matches, ambiguous = _replace_matching_source_frames(before, target_by_id, target_by_key)
        actual_sources = _parse_json_field(event.get("source_frames_json"))
        expected_sources = before.get("source_frames_json")
        expected_parsed = _parse_json_field(expected_sources)
        if not ambiguous and matches:
            expected_parsed = changed
        event_verification[event["id"]] = {
            "event_fields_except_source_frames_same": _json_safe({k: event.get(k) for k in before if k != "source_frames_json"}) == {
                k: v for k, v in before.items() if k != "source_frames_json"
            },
            "source_frames_expected": _deep_equal(actual_sources, expected_parsed),
            "matched_sources": matches,
            "ambiguous_skipped": ambiguous,
        }
        if not _deep_equal(actual_sources, expected_parsed):
            event_verification[event["id"]]["actual_source_frames"] = actual_sources
            event_verification[event["id"]]["expected_source_frames"] = expected_parsed
    conn.close()
    all_chroma = all(all(v.values()) for v in chroma_verification.values())
    all_scenes = all(all(v.values()) for v in scenes_verification.values())
    all_events = all(v["event_fields_except_source_frames_same"] and v["source_frames_expected"] for v in event_verification.values())
    return {
        "chroma": chroma_verification,
        "chroma_all_pass": all_chroma,
        "scenes": scenes_verification,
        "scenes_all_pass": all_scenes,
        "behavior_events": event_verification,
        "behavior_events_all_pass": all_events,
    }


def _run_search_checks(canary):
    from pipeline.vector_store import search
    videos = sorted({f["video_id"] for f in canary})
    query_by_species = {"cat": "고양이가 보이는 장면이 있어?", "dog": "강아지가 보이는 장면이 있어?", "person": "사람이 보이는 장면이 있어?"}
    output = {}
    canary_by_video = {v: [f for f in canary if f["video_id"] == v] for v in videos}
    for video in videos:
        output[video] = {}
        for species, query in query_by_species.items():
            hits = search(query, top_k=200, video_id=video)
            hit_ids = {r["frame_id"] for r in hits}
            output[video][species] = {
                "result_count": len(hits),
                "canary_hits": [f["frame_id"] for f in canary_by_video[video] if f["frame_id"] in hit_ids],
            }
    return output


def _run_question_checks(canary):
    from pipeline.query_suggester import suggest_query_items
    videos = sorted({f["video_id"] for f in canary})
    output = {}
    for video in videos:
        try:
            items = suggest_query_items(video_id=video, limit=3)
            output[video] = {
                "count": len(items),
                "items": [{
                    "question": x.get("question"),
                    "question_source": x.get("question_source"),
                    "source_event_id": x.get("source_event_id"),
                    "original_event_id": x.get("original_event_id"),
                    "event_start": x.get("event_start"),
                    "event_end": x.get("event_end"),
                } for x in items],
            }
        except Exception as exc:
            output[video] = {"error": f"{type(exc).__name__}: {exc}"}
    return output


def rollback_dry_run():
    if not BACKUP_PATH.is_file():
        raise RuntimeError(f"Backup not found: {BACKUP_PATH}")
    data = json.loads(BACKUP_PATH.read_text())
    print(json.dumps({
        "mode": "dry_run_no_writes",
        "chroma_frame_ids": [r["frame_id"] for r in data["chroma"]],
        "scene_ids": [r["id"] for r in data["scenes"]],
        "behavior_event_ids": [r["id"] for r in data["behavior_events"]],
        "backup": str(BACKUP_PATH),
    }, ensure_ascii=False, indent=2))


def rollback():
    if not BACKUP_PATH.is_file():
        raise RuntimeError(f"Backup not found: {BACKUP_PATH}")
    data = json.loads(BACKUP_PATH.read_text())
    client, collection = _get_chroma()
    original_chroma = {r["frame_id"]: r for r in data["chroma"]}
    current = _chroma_rows(collection, list(original_chroma))
    for fid, before in original_chroma.items():
        if current[fid]["document"] != before["document"] or current[fid]["embedding_float32_bytes_hex"] != before["embedding_float32_bytes_hex"]:
            raise RuntimeError(f"Embedding/document drift for {fid}; refuse metadata rollback")
    for fid, before in original_chroma.items():
        metadata = dict(before["metadata"])
        # Chroma's update API merges metadata keys and cannot remove a key by
        # omission. Empty JSON strings parse as no detections/resolutions and
        # restore the prior observable behavior for keys absent in the backup.
        for key in ("object_detections_json", "species_resolution_json"):
            if key not in metadata:
                metadata[key] = ""
        collection.update(ids=[fid], metadatas=[metadata])

    conn = connect(dict_rows=True)
    cursor = conn.cursor()
    columns = set(_mysql_schema(conn)["column_names"])
    for scene in data["scenes"]:
        object_json = scene.get("object_detections_json") if "object_detections_json" in columns else None
        species_json = scene.get("species_resolution_json") if "species_resolution_json" in columns else None
        cursor.execute(
            "UPDATE scenes SET object_labels = ?, object_detections_json = ?, species_resolution_json = ? WHERE id = ?",
            (scene.get("object_labels"), object_json, species_json, scene["id"]),
        )
        if cursor.rowcount not in (0, 1):
            conn.rollback(); conn.close()
            raise RuntimeError(f"Rollback scene ID not uniquely updated: {scene['id']}")
    for event in data["behavior_events"]:
        cursor.execute(
            "UPDATE behavior_events SET source_frames_json = ? WHERE id = ?",
            (event.get("source_frames_json"), event["id"]),
        )
        if cursor.rowcount not in (0, 1):
            conn.rollback(); conn.close()
            raise RuntimeError(f"Rollback event ID not uniquely updated: {event['id']}")
    conn.commit(); conn.close()
    print("Rollback restored canary metadata and source_frames. Additive schema columns were retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("backup", "apply", "rollback-dry-run", "rollback"), required=True)
    args = parser.parse_args()
    if args.action == "backup":
        backup()
    elif args.action == "apply":
        apply()
    elif args.action == "rollback-dry-run":
        rollback_dry_run()
    else:
        rollback()


if __name__ == "__main__":
    main()
