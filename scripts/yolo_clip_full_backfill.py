"""Safely apply the verified CLIP species metadata to the 240 eligible frames.

The eligible manifest is the only write allowlist. This script never scans or
mutates the ineligible set, and it preserves existing Chroma documents and
embeddings. Run `backup` first, then `apply`.
"""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
_shadow_weight = Path("/tmp/yolo_clip_species_eval/yolov8s.pt")
if not os.getenv("YOLO_MODEL") and _shadow_weight.is_file():
    os.environ["YOLO_MODEL"] = str(_shadow_weight)

from db.connection import connect
from pipeline import vector_store
from pipeline.species_resolver import resolve_detection_result
from pipeline.yolo_detector import detect_objects_with_details
from scripts.yolo_clip_canary_backfill import (
    _chroma_rows,
    _deep_equal,
    _get_chroma,
    _json_safe,
    _matching_event_sources,
    _mysql_schema,
    _parse_json_field,
)

EVAL = ROOT / "docs/evaluations"
BACKUP = EVAL / "backups/yolo-clip-full-backfill-before-2026-09-27.json"
REPORT = EVAL / "yolo-clip-full-backfill-result-2026-09-27.json"
ELIGIBLE = EVAL / "yolo-backfill-eligible-2026-09-27.json"
SHADOW = EVAL / "yolo-clip-species-shadow-2026-09-27.json"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    temp.replace(path)


def _rollback_db(conn):
    raw = getattr(conn, "_raw_connection", None)
    if raw is not None:
        raw.rollback()
    else:
        rollback_method = getattr(conn, "rollback", None)
        if rollback_method:
            rollback_method()


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return {"base64": __import__("base64").b64encode(value).decode("ascii")}
    return str(value)


def _load_manifests():
    eligible_doc = json.loads(ELIGIBLE.read_text())
    shadow_doc = json.loads(SHADOW.read_text())
    frames = eligible_doc["frames"]
    shadow = {r["frame_id"]: r for r in shadow_doc["frames"]}
    ids = [r["frame_id"] for r in frames]
    if len(frames) != 240 or len(set(ids)) != 240 or len(shadow) != 240 or set(ids) != set(shadow):
        raise RuntimeError("Eligible and shadow manifests must contain the same exact 240 unique frame IDs")
    counts = Counter(r["video_id"] for r in frames)
    expected = {"IMG_8450_2": 33, "dog_drinking_water": 89, "dog_escape": 28, "two_dogs": 90}
    if dict(counts) != expected:
        raise RuntimeError(f"Eligible video counts differ from the approved manifest: {dict(counts)}")
    if shadow_doc.get("counts", {}).get("processed") != 240 or shadow_doc.get("counts", {}).get("failures") != 0:
        raise RuntimeError("Shadow manifest is incomplete or contains failures")
    return frames, shadow, eligible_doc["gcs_bucket"], shadow_doc


def _scene_lookup(conn, frames):
    cursor = conn.cursor()
    by_frame, duplicates, missing = {}, [], []
    for frame in frames:
        cursor.execute("SELECT * FROM scenes WHERE video_id = ? AND s3_key = ? ORDER BY id", (frame["video_id"], frame["s3_key"]))
        rows = cursor.fetchall()
        by_frame[frame["frame_id"]] = rows
        if not rows:
            missing.append(frame["frame_id"])
        elif len(rows) > 1:
            duplicates.append({"frame_id": frame["frame_id"], "scene_ids": [r["id"] for r in rows]})
    return by_frame, duplicates, missing


def _event_snapshots(conn, frames):
    videos = sorted({r["video_id"] for r in frames})
    marks = ",".join("?" for _ in videos)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM behavior_events WHERE video_id IN ({marks}) ORDER BY id", tuple(videos))
    target_by_id = {r["frame_id"]: r for r in frames}
    target_by_key = defaultdict(list)
    for frame in frames:
        target_by_key[(frame["video_id"], frame["s3_key"])].append(frame["frame_id"])
    events, ambiguous = [], []
    for row in cursor.fetchall():
        sources = _parse_json_field(row.get("source_frames_json"))
        matches, is_ambiguous = _matching_event_sources(sources, row["video_id"], target_by_id, target_by_key)
        if is_ambiguous:
            ambiguous.append(row["id"])
        if matches or is_ambiguous:
            events.append({"row": row, "matched_sources": matches, "ambiguous": is_ambiguous})
    return events, ambiguous


def backup():
    if BACKUP.exists():
        raise RuntimeError(f"Refusing to overwrite full-backfill backup: {BACKUP}")
    frames, _, _, _ = _load_manifests()
    ids = [r["frame_id"] for r in frames]
    client, collection = _get_chroma()
    chroma_map = _chroma_rows(collection, ids)
    missing_chroma = [fid for fid in ids if fid not in chroma_map]
    if missing_chroma:
        raise RuntimeError(f"Chroma is missing eligible frame IDs: {missing_chroma[:20]}")
    identity_mismatch = []
    for frame in frames:
        metadata = chroma_map[frame["frame_id"]]["metadata"]
        if metadata.get("video_id") != frame["video_id"] or metadata.get("s3_key") != frame["s3_key"]:
            identity_mismatch.append(frame["frame_id"])
    if identity_mismatch:
        raise RuntimeError(f"Chroma identity differs from the allowlist: {identity_mismatch[:20]}")

    conn = connect(dict_rows=True)
    schema = _mysql_schema(conn)
    needed = {"object_detections_json", "species_resolution_json"}
    absent = sorted(needed - set(schema["column_names"]))
    if absent:
        conn.close()
        raise RuntimeError(f"Required MySQL columns are missing; migrations are forbidden in this run: {absent}")
    scenes, duplicates, missing_scenes = _scene_lookup(conn, frames)
    events, ambiguous_events = _event_snapshots(conn, frames)
    conn.close()

    safe_frame_ids = [
        frame["frame_id"] for frame in frames
        if len(scenes[frame["frame_id"]]) == 1 and frame["frame_id"] in chroma_map
    ]
    duplicate_ids = {item["frame_id"] for item in duplicates}
    safe_frame_ids = [fid for fid in safe_frame_ids if fid not in duplicate_ids and fid not in missing_scenes]
    data = {
        "created_at": datetime.now().astimezone().isoformat(),
        "eligible_manifest": str(ELIGIBLE.relative_to(ROOT)),
        "shadow_manifest": str(SHADOW.relative_to(ROOT)),
        "expected_count": 240,
        "current_state_includes_canary": True,
        "mysql_schema": schema,
        "allowlisted_frames": frames,
        "safe_frame_ids": safe_frame_ids,
        "missing_chroma": missing_chroma,
        "scene_duplicates": duplicates,
        "scene_missing": missing_scenes,
        "event_ambiguous_ids": ambiguous_events,
        "chroma": list(chroma_map.values()),
        "scenes": [row for rows in scenes.values() for row in rows],
        "scene_mapping": {fid: [row["id"] for row in rows] for fid, rows in scenes.items()},
        "behavior_events": [item["row"] for item in events],
        "event_match_details": [
            {"event_id": item["row"]["id"], "matched_sources": item["matched_sources"], "ambiguous": item["ambiguous"]}
            for item in events
        ],
    }
    _write_json(BACKUP, data)
    print(json.dumps({
        "backup": str(BACKUP), "chroma_frames": len(chroma_map),
        "scene_rows": len(data["scenes"]), "unique_scene_mappings": len(safe_frame_ids),
        "event_rows": len(events), "scene_duplicates": len(duplicates),
        "scene_missing": len(missing_scenes), "ambiguous_events": len(ambiguous_events),
        "mysql_version": schema["version"],
    }, ensure_ascii=False))
    return data


def _compute_all(frames, shadow, bucket_name):
    from google.cloud import storage
    bucket = storage.Client().bucket(bucket_name)
    root = Path(tempfile.gettempdir()) / "yolo-clip-full-backfill"
    root.mkdir(parents=True, exist_ok=True)
    computed, failures = {}, []
    for index, frame in enumerate(frames, 1):
        fid = frame["frame_id"]
        path = root / (hashlib.sha1(fid.encode()).hexdigest() + ".jpg")
        try:
            if not path.is_file() or path.stat().st_size == 0:
                bucket.blob(frame["s3_key"]).download_to_filename(str(path))
            raw = detect_objects_with_details(path)
            final = resolve_detection_result(path, raw, enabled=True)
            expected = shadow[fid]
            checks = {
                "raw_object_detections": _deep_equal(raw["object_detections"], expected["raw_yolo_detections"]),
                "species_resolution": _deep_equal(final["species_resolution"], expected["species_resolution"]),
                "object_labels": final["object_labels"] == expected["final_object_labels"],
            }
            if not all(checks.values()):
                failures.append({"frame_id": fid, "reason": "shadow_mismatch", "checks": checks})
            else:
                computed[fid] = final
        except Exception as exc:
            failures.append({"frame_id": fid, "reason": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0 or index == len(frames):
            print(f"YOLO+CLIP shadow check [{index}/{len(frames)}]", flush=True)
    if failures or len(computed) != 240:
        _write_json(REPORT, {"phase": "prewrite_shadow_validation", "computed": len(computed), "failures": failures})
        raise RuntimeError(f"Pre-write shadow comparison failed: {len(computed)}/240 matched; report={REPORT}")
    return computed


def _labels(value):
    if isinstance(value, list):
        return sorted(set(str(x) for x in value))
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return sorted(set(str(x) for x in parsed))
        except (ValueError, TypeError):
            pass
        return sorted(set(x.strip() for x in value.split(",") if x.strip()))
    return []


def _chroma_matches(metadata, expected):
    return (
        _labels(metadata.get("object_labels")) == sorted(expected["object_labels"])
        and _deep_equal(vector_store._parse_object_detections(metadata), expected["object_detections"])
        and _deep_equal(vector_store._parse_species_resolution(metadata), expected["species_resolution"])
    )


def _scene_matches(row, expected):
    return (
        _labels(row.get("object_labels")) == sorted(expected["object_labels"])
        and _deep_equal(_parse_json_field(row.get("object_detections_json")), expected["object_detections"])
        and _deep_equal(_parse_json_field(row.get("species_resolution_json")), expected["species_resolution"])
    )


def _event_updates(events, computed, safe_ids):
    target_by_id = {fid: computed[fid] for fid in safe_ids}
    target_by_key = defaultdict(list)
    for frame in json.loads(ELIGIBLE.read_text())["frames"]:
        if frame["frame_id"] in safe_ids:
            target_by_key[(frame["video_id"], frame["s3_key"])].append(frame["frame_id"])
    patches, skipped = {}, []
    for item in events:
        row = item["row"]
        sources = _parse_json_field(row.get("source_frames_json"))
        if item["ambiguous"]:
            skipped.append({"event_id": row["id"], "reason": "ambiguous_source_match"})
            patches[row["id"]] = {"before": row, "source_frames": sources, "frame_ids": [], "needs_update": False, "ambiguous": True}
            continue
        if not isinstance(sources, list):
            continue
        matches, ambiguous = _matching_event_sources(sources, row["video_id"], target_by_id, target_by_key)
        if ambiguous:
            skipped.append({"event_id": row["id"], "reason": "ambiguous_source_match"})
            patches[row["id"]] = {"before": row, "source_frames": sources, "frame_ids": [], "needs_update": False, "ambiguous": True}
            continue
        updated = json.loads(json.dumps(sources, ensure_ascii=False))
        changed_ids = []
        for index, fid in matches:
            value = computed[fid]
            replacement = {
                "object_labels": value["object_labels"],
                "object_detections": value["object_detections"],
                "species_resolution": value["species_resolution"],
            }
            if not _deep_equal({k: updated[index].get(k) for k in replacement}, replacement):
                updated[index].update(replacement)
                changed_ids.append(fid)
        if matches:
            patches[row["id"]] = {
                "before": row, "source_frames": updated,
                "frame_ids": sorted(set(changed_ids)), "needs_update": bool(changed_ids), "ambiguous": False,
            }
    return patches, skipped


def _change_statistics(frames, before_chroma, computed, shadow):
    rows = []
    for frame in frames:
        fid = frame["frame_id"]
        old = _labels(before_chroma[fid]["metadata"].get("object_labels"))
        new = sorted(computed[fid]["object_labels"])
        old_set, new_set = set(old), set(new)
        raw = computed[fid]["object_detections"]
        low_resolved = any(
            0.4 <= float(d["confidence"]) < 0.5
            and d["label"] in {"cat", "dog"}
            and any(r["bbox"] == d["bbox"] and r["resolved_label"] in new_set for r in computed[fid]["species_resolution"])
            for d in raw
        )
        changes = {
            "changed": old != new,
            "empty_to_detected": not old_set and bool(new_set),
            "detected_to_empty": bool(old_set) and not new_set,
            "dog_to_cat": "dog" in old_set and "cat" in new_set and "dog" not in new_set,
            "cat_to_dog": "cat" in old_set and "dog" in new_set and "cat" not in new_set,
            "low_confidence_animal_recovery": low_resolved,
            "person_added": "person" not in old_set and "person" in new_set,
            "person_removed": "person" in old_set and "person" not in new_set,
            "labels_added": sorted(new_set - old_set),
            "labels_removed": sorted(old_set - new_set),
        }
        rows.append({"frame_id": fid, "video_id": frame["video_id"], "old_labels": old, "new_labels": new, **changes})
    def summarize(items):
        summary = {key: sum(bool(r[key]) for r in items) for key in (
            "changed", "empty_to_detected", "detected_to_empty", "dog_to_cat", "cat_to_dog",
            "low_confidence_animal_recovery", "person_added", "person_removed",
        )}
        summary["unchanged"] = len(items) - summary["changed"]
        summary["labels_added"] = sum(len(r["labels_added"]) for r in items)
        summary["labels_removed"] = sum(len(r["labels_removed"]) for r in items)
        summary["total"] = len(items)
        return summary
    videos = {}
    for video in sorted({r["video_id"] for r in rows}):
        videos[video] = summarize([r for r in rows if r["video_id"] == video])
    return {"overall": summarize(rows), "by_video": videos, "frames": rows}


def apply():
    if not BACKUP.is_file():
        raise RuntimeError(f"Create the current-state backup first: {BACKUP}")
    frames, shadow, bucket_name, shadow_doc = _load_manifests()
    backup_doc = json.loads(BACKUP.read_text())
    frame_ids = [r["frame_id"] for r in frames]
    if backup_doc.get("expected_count") != 240 or backup_doc.get("allowlisted_frames") != frames:
        raise RuntimeError("Backup does not match the exact approved eligible manifest")
    computed = _compute_all(frames, shadow, bucket_name)
    client, collection = _get_chroma()
    current_chroma = _chroma_rows(collection, frame_ids)
    before_chroma = {r["frame_id"]: r for r in backup_doc["chroma"]}
    for fid in frame_ids:
        row = current_chroma.get(fid)
        old = before_chroma.get(fid)
        if not row or not old or not _deep_equal(row["metadata"], old["metadata"]) or row["document"] != old["document"] or row["embedding_float32_bytes_hex"] != old["embedding_float32_bytes_hex"]:
            raise RuntimeError(f"Chroma changed since backup for {fid}; refusing write")

    conn = connect(dict_rows=True)
    schema = _mysql_schema(conn)
    if schema["version"] != backup_doc["mysql_schema"]["version"]:
        conn.close()
        raise RuntimeError("MySQL server version changed since backup")
    required = {"object_detections_json", "species_resolution_json"}
    if not required.issubset(schema["column_names"]):
        conn.close()
        raise RuntimeError("Required JSON columns disappeared; schema changes are forbidden in this step")
    scene_matches, duplicates, missing = _scene_lookup(conn, frames)
    if duplicates != backup_doc["scene_duplicates"] or missing != backup_doc["scene_missing"]:
        conn.close()
        raise RuntimeError("Scene mapping changed since full backup")
    old_scenes = {row["id"]: row for row in backup_doc["scenes"]}
    for fid, rows in scene_matches.items():
        for row in rows:
            saved = old_scenes.get(row["id"])
            if not saved or any(_json_safe(row).get(k) != v for k, v in saved.items()):
                conn.close()
                raise RuntimeError(f"Scene row changed since backup: {row['id']}")
    event_rows, ambiguous_events = _event_snapshots(conn, frames)
    old_events = {row["id"]: row for row in backup_doc["behavior_events"]}
    for item in event_rows:
        saved = old_events.get(item["row"]["id"])
        if not saved or not _deep_equal(_json_safe(item["row"]), saved):
            conn.close()
            raise RuntimeError(f"Behavior event changed since backup: {item['row']['id']}")

    blocked = set(backup_doc["missing_chroma"]) | set(backup_doc["scene_missing"])
    blocked |= {x["frame_id"] for x in backup_doc["scene_duplicates"]}
    safe_ids = [fid for fid in frame_ids if fid not in blocked]
    scene_id_by_frame = {fid: scene_matches[fid][0]["id"] for fid in safe_ids}
    current_scenes = {fid: scene_matches[fid][0] for fid in safe_ids}
    chroma_already = {fid for fid in safe_ids if _chroma_matches(current_chroma[fid]["metadata"], computed[fid])}
    scenes_already = {fid for fid in safe_ids if _scene_matches(current_scenes[fid], computed[fid])}
    already = [fid for fid in safe_ids if fid in chroma_already and fid in scenes_already]
    to_update = [fid for fid in safe_ids if fid not in already]
    stats = _change_statistics(frames, before_chroma, computed, shadow)
    event_patches, skipped_events = _event_updates(event_rows, computed, safe_ids)
    cursor = conn.cursor()
    updated_ids = []
    updated_event_ids = []
    try:
        for fid in to_update:
            result = computed[fid]
            if fid not in chroma_already:
                vector_store.update_frame_detection_metadata(
                    fid, result["object_labels"], result["object_detections"],
                    species_resolution=result["species_resolution"],
                )
            if fid not in scenes_already:
                labels_json = json.dumps(result["object_labels"], ensure_ascii=False)
                cursor.execute(
                    "UPDATE scenes SET object_labels = ?, object_detections_json = ?, species_resolution_json = ? WHERE id = ?",
                    (labels_json, vector_store._normalize_object_detections(result["object_detections"]),
                     vector_store._normalize_species_resolution(result["species_resolution"]), scene_id_by_frame[fid]),
                )
                if cursor.rowcount not in (0, 1):
                    raise RuntimeError(f"Unsafe scene update rowcount for {fid}: {cursor.rowcount}")
            updated_ids.append(fid)
        for event_id, patch in event_patches.items():
            if not patch["needs_update"]:
                continue
            cursor.execute(
                "UPDATE behavior_events SET source_frames_json = ? WHERE id = ?",
                (json.dumps(patch["source_frames"], ensure_ascii=False, separators=(",", ":")), event_id),
            )
            if cursor.rowcount not in (0, 1):
                raise RuntimeError(f"Unsafe event update rowcount: {event_id}")
            updated_event_ids.append(event_id)
        conn.commit()
    except Exception:
        _rollback_db(conn)
        conn.close()
        _write_json(REPORT, {
            "phase": "write_error", "updated_ids_before_error": updated_ids,
            "updated_event_ids_before_error": updated_event_ids,
            "error": "write interrupted; inspect current state against full backup before rollback",
        })
        raise
    conn.close()

    verification = _verify_full(frames, computed, backup_doc, scene_id_by_frame, event_patches, client, collection)
    _write_json(REPORT, {
        "phase": "post_write_verification", "eligible_count": 240,
        "shadow_match_count": 240, "shadow_mismatch_count": 0,
        "already_applied": len(already), "newly_updated": len(updated_ids),
        "skipped": len(blocked), "failed": 240 - len(already) - len(updated_ids) - len(blocked),
        "skipped_frame_ids": sorted(blocked), "scene_id_by_frame": scene_id_by_frame,
        "updated_event_ids": updated_event_ids, "skipped_events": skipped_events,
        "statistics": stats, "verification": verification,
        "shadow_model": shadow_doc.get("model"), "resolver": shadow_doc.get("resolver"),
    })
    if not (verification["chroma_all_pass"] and verification["scenes_all_pass"] and verification["events_all_pass"]):
        raise RuntimeError(f"Post-write verification failed; report saved to {REPORT}. Assess failed scope before rollback.")

    result = json.loads(REPORT.read_text())
    result["end_to_end"] = _end_to_end_checks(frames)
    _write_json(REPORT, result)

    return json.loads(REPORT.read_text())


def _end_to_end_checks(frames):
    from pipeline.query_suggester import suggest_query_items
    query_map = {
        "cat": "고양이가 보이는 장면이 있어?",
        "dog": "강아지가 보이는 장면이 있어?",
        "person": "사람이 보이는 장면이 있어?",
    }
    active = sorted({r["video_id"] for r in frames})
    stale_doc = json.loads((EVAL / "yolo-backfill-ineligible-2026-09-27.json").read_text())
    stale_frames = stale_doc.get("frames", [])
    stale_ids = {r["frame_id"] for r in stale_frames}
    stale_keys = {(r["video_id"], r["s3_key"]): r["frame_id"] for r in stale_frames}
    query_videos = active + ["cats_5min"]
    object_search = {}
    for video in query_videos:
        object_search[video] = {}
        for species, query in query_map.items():
            try:
                hits = vector_store.search(query, top_k=500, video_id=video)
                hit_ids = {item.get("frame_id") for item in hits}
                object_search[video][species] = {
                    "result_count": len(hits),
                    "eligible_canary_ids_in_results": sorted(hit_ids & {f["frame_id"] for f in frames if f["video_id"] == video}),
                    "ineligible_ids_in_results": sorted(hit_ids & stale_ids),
                }
            except Exception as exc:
                object_search[video][species] = {"error": f"{type(exc).__name__}: {exc}"}

    videos_for_suggestions = active + ["cats_5min"]
    suggestions = {}
    label_rows = defaultdict(list)
    conn = connect(dict_rows=True)
    cursor = conn.cursor()
    marks = ",".join("?" for _ in videos_for_suggestions)
    cursor.execute(f"SELECT id, video_id, s3_key, object_labels FROM scenes WHERE video_id IN ({marks}) ORDER BY id", tuple(videos_for_suggestions))
    for row in cursor.fetchall():
        label_rows[row["video_id"]].append(row)
    # Read all behavior sources for stale-video impact only; do not use them to
    # alter any event row or any ineligible frame metadata.
    stale_event_sources = defaultdict(list)
    cursor.execute("SELECT id, video_id, action, target_object, summary, source_frames_json FROM behavior_events WHERE video_id IN (?, ?)", ("cats_5min", "two_dogs"))
    for event in cursor.fetchall():
        sources = _parse_json_field(event.get("source_frames_json"))
        for source in sources if isinstance(sources, list) else []:
            if not isinstance(source, dict):
                continue
            fid = source.get("frame_id") or stale_keys.get((source.get("video_id") or event["video_id"], source.get("s3_key")))
            if fid in stale_ids:
                stale_event_sources[event["video_id"]].append({
                    "event_id": event["id"], "frame_id": fid,
                    "object_labels": _labels(source.get("object_labels")),
                    "action": event.get("action"), "target_object": event.get("target_object"),
                })
    conn.close()

    stale_label_counts = {}
    aggregate_label_counts = {}
    for video in ("two_dogs", "cats_5min"):
        stale_counts, all_counts = Counter(), Counter()
        for row in label_rows.get(video, []):
            labels = _labels(row.get("object_labels"))
            all_counts.update(labels)
            if (video, row.get("s3_key")) in stale_keys:
                stale_counts.update(labels)
        stale_label_counts[video] = dict(stale_counts)
        aggregate_label_counts[video] = dict(all_counts)

    for video in videos_for_suggestions:
        try:
            items = suggest_query_items(video_id=video, limit=3)
            all_counts = Counter(label for row in label_rows.get(video, []) for label in _labels(row.get("object_labels")))
            stale_counts = Counter()
            if video in {"two_dogs", "cats_5min"}:
                for row in label_rows.get(video, []):
                    if (video, row.get("s3_key")) in stale_keys:
                        stale_counts.update(_labels(row.get("object_labels")))
            suggestions[video] = {
                "count": len(items),
                "items": [],
                "aggregate_label_counts": dict(all_counts),
            }
            for item in items:
                question = item.get("question", "")
                source = item.get("question_source") or "fallback"
                mentions = [species for species, aliases in {
                    "cat": ("고양이", "cat"), "dog": ("강아지", "개", "dog"), "person": ("사람", "person", "human")
                }.items() if any(token in question.lower() for token in aliases)]
                suggestions[video]["items"].append({
                    "question": question,
                    "question_source": source,
                    "source_event_id": item.get("source_event_id"),
                    "original_event_id": item.get("original_event_id"),
                    "event_start": item.get("event_start"),
                    "event_end": item.get("event_end"),
                    "fallback_label_evidence": {
                        label: {"aggregate_scene_count": all_counts[label], "stale_frame_count": stale_counts[label]}
                        for label in mentions
                    } if "fallback" in source.lower() else {},
                })
        except Exception as exc:
            suggestions[video] = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "read_only_searches": object_search,
        "suggestions": suggestions,
        "stale_video_label_counts": {"all_scenes": aggregate_label_counts, "ineligible_scenes": stale_label_counts},
        "stale_behavior_event_sources": {video: items for video, items in stale_event_sources.items()},
        "stale_frame_counts": {video: sum(1 for r in stale_frames if r["video_id"] == video) for video in ("cats_5min", "two_dogs")},
    }


def _verify_full(frames, computed, backup_doc, scene_id_by_frame, event_patches, client, collection):
    ids = [f["frame_id"] for f in frames]
    after_chroma = _chroma_rows(collection, ids)
    old_chroma = {r["frame_id"]: r for r in backup_doc["chroma"]}
    chroma_checks = {}
    for frame in frames:
        fid = frame["frame_id"]
        if fid not in scene_id_by_frame:
            continue
        before, after = old_chroma[fid], after_chroma[fid]
        expected = computed[fid]
        old_other = {k: v for k, v in before["metadata"].items() if k not in {"object_labels", "object_detections_json", "species_resolution_json"}}
        after_other = {k: v for k, v in after["metadata"].items() if k not in {"object_labels", "object_detections_json", "species_resolution_json"}}
        chroma_checks[fid] = {
            "frame_id_same": after["frame_id"] == fid,
            "document_same": after["document"] == before["document"],
            "embedding_bytes_same": after["embedding_float32_bytes_hex"] == before["embedding_float32_bytes_hex"],
            "other_metadata_same": after_other == old_other,
            "object_labels_expected": _labels(after["metadata"].get("object_labels")) == sorted(expected["object_labels"]),
            "object_detections_expected": _deep_equal(vector_store._parse_object_detections(after["metadata"]), expected["object_detections"]),
            "species_resolution_expected": _deep_equal(vector_store._parse_species_resolution(after["metadata"]), expected["species_resolution"]),
        }
    conn = connect(dict_rows=True)
    cursor = conn.cursor()
    before_scenes = {r["id"]: r for r in backup_doc["scenes"]}
    scene_checks = {}
    for fid, sid in scene_id_by_frame.items():
        cursor.execute("SELECT * FROM scenes WHERE id = ?", (sid,))
        rows = cursor.fetchall()
        frame = next(x for x in frames if x["frame_id"] == fid)
        expected = computed[fid]
        if len(rows) != 1:
            scene_checks[fid] = {"scene_id": sid, "row_count": len(rows), "pass": False}
            continue
        row = rows[0]
        before = before_scenes[sid]
        other_keys = set(before) - {"object_labels", "object_detections_json", "species_resolution_json"}
        scene_checks[fid] = {
            "scene_id_same": row["id"] == sid,
            "identity_same": row["video_id"] == frame["video_id"] and row["s3_key"] == frame["s3_key"],
            "other_fields_same": _deep_equal(_json_safe({k: row.get(k) for k in other_keys}), {k: before[k] for k in other_keys}),
            "object_labels_expected": _labels(row.get("object_labels")) == sorted(expected["object_labels"]),
            "object_detections_expected": _deep_equal(_parse_json_field(row.get("object_detections_json")), expected["object_detections"]),
            "species_resolution_expected": _deep_equal(_parse_json_field(row.get("species_resolution_json")), expected["species_resolution"]),
        }
    saved_events = {r["id"]: r for r in backup_doc["behavior_events"]}
    event_checks = {}
    for eid, patch in event_patches.items():
        cursor.execute("SELECT * FROM behavior_events WHERE id = ?", (eid,))
        rows = cursor.fetchall()
        before = saved_events[eid]
        if len(rows) != 1:
            event_checks[eid] = {"row_count": len(rows), "pass": False}
            continue
        row = rows[0]
        actual_sources = _parse_json_field(row.get("source_frames_json"))
        event_checks[eid] = {
            "event_body_same": _deep_equal(_json_safe({k: v for k, v in row.items() if k != "source_frames_json"}),
                                            {k: v for k, v in before.items() if k != "source_frames_json"}),
            "source_frames_expected": _deep_equal(actual_sources, patch["source_frames"]),
        }
    conn.close()
    return {
        "chroma": chroma_checks,
        "chroma_all_pass": all(all(v.values()) for v in chroma_checks.values()),
        "scenes": scene_checks,
        "scenes_all_pass": all(all(v.values()) for v in scene_checks.values()) and len(scene_checks) == len(scene_id_by_frame),
        "events": event_checks,
        "events_all_pass": all(all(v.values()) for v in event_checks.values()),
    }


def rollback():
    if not BACKUP.is_file():
        raise RuntimeError(f"Full backfill backup not found: {BACKUP}")
    backup_doc = json.loads(BACKUP.read_text())
    ids = [row["frame_id"] for row in backup_doc["chroma"]]
    client, collection = _get_chroma()
    current = _chroma_rows(collection, ids)
    for saved in backup_doc["chroma"]:
        fid = saved["frame_id"]
        live = current.get(fid)
        if not live or live["document"] != saved["document"] or live["embedding_float32_bytes_hex"] != saved["embedding_float32_bytes_hex"]:
            raise RuntimeError(f"Refusing rollback because frame/document/embedding changed: {fid}")

    conn = connect(dict_rows=True)
    cursor = conn.cursor()
    live_schema = _mysql_schema(conn)
    columns = set(live_schema["column_names"])
    if not {"object_detections_json", "species_resolution_json"}.issubset(columns):
        conn.close()
        raise RuntimeError("Rollback requires the currently installed detection JSON columns")
    for scene in backup_doc["scenes"]:
        cursor.execute(
            "UPDATE scenes SET object_labels = ?, object_detections_json = ?, species_resolution_json = ? WHERE id = ?",
            (scene.get("object_labels"), scene.get("object_detections_json"), scene.get("species_resolution_json"), scene["id"]),
        )
        if cursor.rowcount not in (0, 1):
            _rollback_db(conn); conn.close()
            raise RuntimeError(f"Unsafe rollback scene rowcount for id {scene['id']}")
    for event in backup_doc["behavior_events"]:
        cursor.execute("UPDATE behavior_events SET source_frames_json = ? WHERE id = ?", (event.get("source_frames_json"), event["id"]))
        if cursor.rowcount not in (0, 1):
            _rollback_db(conn); conn.close()
            raise RuntimeError(f"Unsafe rollback event rowcount for id {event['id']}")
    conn.commit()
    conn.close()

    for saved in backup_doc["chroma"]:
        metadata = dict(saved["metadata"])
        for key in ("object_detections_json", "species_resolution_json"):
            if key not in metadata:
                metadata[key] = ""
        collection.update(ids=[saved["frame_id"]], metadatas=[metadata])
    print("Full backfill rollback restored metadata; schema columns and embeddings were retained.")


def rollback_dry_run():
    if not BACKUP.is_file():
        raise RuntimeError(f"Backup not found: {BACKUP}")
    d = json.loads(BACKUP.read_text())
    print(json.dumps({"mode": "dry_run_no_writes", "chroma_frames": len(d["chroma"]),
                      "scene_rows": len(d["scenes"]), "behavior_event_rows": len(d["behavior_events"]),
                      "backup": str(BACKUP)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("backup", "apply", "rollback-dry-run", "rollback"), required=True)
    args = parser.parse_args()
    if args.action == "backup":
        backup()
    elif args.action == "apply":
        result = apply()
        print(json.dumps({k: result.get(k) for k in (
            "already_applied", "newly_updated", "skipped", "failed", "statistics", "verification"
        )}, ensure_ascii=False, default=_json_default))
    elif args.action == "rollback-dry-run":
        rollback_dry_run()
    else:
        rollback()


if __name__ == "__main__":
    main()
