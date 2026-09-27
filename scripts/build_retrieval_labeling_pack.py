#!/usr/bin/env python3
"""Build an image-only human labeling pack from the exact eligible frame manifest.

Reads missing frame images from GCS and writes contact sheets/forms locally.
It does not call retrieval code or write to GCS, Chroma, or MySQL.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/evaluations/yolo-backfill-eligible-2026-09-27.json"
GT_PATH = ROOT / "docs/evaluations/retrieval-ground-truth-2026-09-27.json"
OUT_DIR = ROOT / "docs/evaluations/retrieval-labeling"
VIDEO_CACHE = Path(tempfile.gettempdir()) / "retrieval-labeling-videos-2026-09-27"
VIDEO_COUNTS = {"IMG_8450_2": 33, "dog_drinking_water": 89, "dog_escape": 28, "two_dogs": 90}
COLUMNS = 4
ROWS = 4
PER_PAGE = COLUMNS * ROWS
THUMB_SIZE = (320, 180)
CARD_HEIGHT = 226
MARGIN = 16
GAP = 12


def load_dotenv() -> None:
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def validate_manifest(frames: list[dict]) -> None:
    ids = [str(frame.get("frame_id") or "") for frame in frames]
    if len(frames) != 240 or len(set(ids)) != 240:
        raise ValueError(f"Expected 240 unique eligible frames, got total={len(frames)} unique={len(set(ids))}")
    counts = Counter(str(frame.get("video_id") or "") for frame in frames)
    if dict(counts) != VIDEO_COUNTS:
        raise ValueError(f"Unexpected eligible video counts: {dict(counts)}")
    if any(frame.get("video_id") not in VIDEO_COUNTS for frame in frames):
        raise ValueError("Manifest contains a video outside the four allowed videos")


def _chunk_stem(frame: dict) -> str:
    stem = Path(frame["s3_key"]).stem
    if "_frame_" not in stem:
        raise ValueError(f"Cannot derive chunk stem from {frame['s3_key']}")
    return stem.split("_frame_", 1)[0]


def acquire_images(frames: list[dict]) -> tuple[dict[str, Path], dict[str, str]]:
    load_dotenv()
    cache_dir = Path("/tmp/yolo_eval_images")
    work_dir = Path(tempfile.gettempdir()) / "retrieval-labeling-images-2026-09-27"
    work_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, Path] = {}
    source: dict[str, str] = {}
    missing = []
    for frame in frames:
        frame_id = str(frame["frame_id"])
        cache_path = cache_dir / (hashlib.sha1(frame_id.encode()).hexdigest() + ".jpg")
        if cache_path.is_file():
            image_paths[frame_id] = cache_path
            source[frame_id] = "verified_local_cache"
        elif (work_dir / (hashlib.sha1(frame_id.encode()).hexdigest() + ".jpg")).is_file():
            image_paths[frame_id] = work_dir / (hashlib.sha1(frame_id.encode()).hexdigest() + ".jpg")
            source[frame_id] = "gcs_exact_object"
        else:
            missing.append(frame)

    if missing:
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise RuntimeError("GCS_BUCKET_NAME is not configured; cannot fetch missing exact frame images")
        from google.cloud import storage
        bucket = storage.Client().bucket(bucket_name)
        failures = []
        for index, frame in enumerate(missing, start=1):
            frame_id = str(frame["frame_id"])
            path = work_dir / (hashlib.sha1(frame_id.encode()).hexdigest() + ".jpg")
            try:
                bucket.blob(frame["s3_key"]).download_to_filename(str(path))
                image_paths[frame_id] = path
                source[frame_id] = "gcs_exact_object"
            except Exception as exc:
                failures.append({"frame_id": frame_id, "s3_key": frame.get("s3_key"), "error": str(exc)})
            if index % 25 == 0 or index == len(missing):
                print(f"GCS reads: {index}/{len(missing)}")
        if failures:
            raise RuntimeError(f"Could not obtain {len(failures)} exact GCS frames; first failures: {failures[:3]}")
    if set(image_paths) != {str(frame["frame_id"]) for frame in frames}:
        raise RuntimeError("Resolved image set does not exactly match eligible frame IDs")
    return image_paths, source


def acquire_source_chunks(frames: list[dict]) -> list[dict]:
    """Match source videos only by exact (video_id, frame-derived chunk_stem)."""
    load_dotenv()
    pairs: dict[tuple[str, str], None] = {}
    for frame in frames:
        pairs[(str(frame["video_id"]), _chunk_stem(frame))] = None

    extensions = {".mp4", ".mov", ".m4v"}
    local_matches: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for search_root in (ROOT / "simulator/chunks", ROOT / "chunks", ROOT / "data/videos", ROOT / "pipeline"):
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                for video_id, stem in pairs:
                    if path.stem == stem and video_id in path.parts:
                        local_matches[(video_id, stem)].append(path)

    gcs_matches: dict[tuple[str, str], list[dict]] = defaultdict(list)
    storage_client = None
    if any(not local_matches.get(pair) for pair in pairs):
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise RuntimeError("GCS_BUCKET_NAME is required to audit exact source chunks")
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        for video_id in sorted({pair[0] for pair in pairs}):
            wanted = {stem for vid, stem in pairs if vid == video_id}
            for blob in bucket.list_blobs(prefix=f"chunks/{video_id}/"):
                path = Path(blob.name)
                # Exact directory scope and exact basename; no fuzzy/name-similarity match.
                if path.stem in wanted and path.suffix.lower() in extensions:
                    gcs_matches[(video_id, path.stem)].append({"gcs_path": blob.name, "size_bytes": int(blob.size or 0)})

    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)
    inventory = []
    for video_id, stem in sorted(pairs):
        local = local_matches.get((video_id, stem), [])
        gcs = gcs_matches.get((video_id, stem), [])
        entry = {
            "video_id": video_id,
            "chunk_stem": stem,
            "local_exact_matches": [str(path.resolve()) for path in local],
            "gcs_exact_matches": gcs,
            "status": "ambiguous" if len(local) > 1 or len(gcs) > 1 or (local and gcs) else "unavailable",
            "playable_local_path": None,
        }
        if len(local) == 1 and not gcs:
            entry["status"] = "local_exact"
            entry["playable_local_path"] = str(local[0].resolve())
        elif len(gcs) == 1 and not local:
            entry["status"] = "gcs_exact"
            object_path = gcs[0]["gcs_path"]
            destination = VIDEO_CACHE / video_id / f"{stem}{Path(object_path).suffix.lower()}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.stat().st_size != gcs[0]["size_bytes"]:
                # Read-only fetch into OS temp for direct local playback; never upload.
                bucket_name = os.getenv("GCS_BUCKET_NAME")
                storage_client.bucket(bucket_name).blob(object_path).download_to_filename(str(destination))
            if destination.stat().st_size != gcs[0]["size_bytes"]:
                raise RuntimeError(f"Downloaded chunk size mismatch for exact source {object_path}")
            entry["playable_local_path"] = str(destination.resolve())
        inventory.append(entry)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "source-chunk-inventory.json").write_text(json.dumps({
        "read_only": True,
        "matched_by": "exact video_id + exact chunk_stem derived from eligible frame s3_key",
        "unique_frame_chunk_pairs": len(pairs),
        "local_exact_count": sum(row["status"] == "local_exact" for row in inventory),
        "gcs_exact_count": sum(row["status"] == "gcs_exact" for row in inventory),
        "unavailable_count": sum(row["status"] == "unavailable" for row in inventory),
        "ambiguous_count": sum(row["status"] == "ambiguous" for row in inventory),
        "chunks": inventory,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return inventory


def _font(size: int):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if Path(name).is_file():
            try:
                return ImageFont.truetype(name, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def build_sheets(frames: list[dict], image_paths: dict[str, Path]) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_video_chunk: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for frame in frames:
        by_video_chunk[(str(frame["video_id"]), _chunk_stem(frame))].append(frame)
    ordered = []
    for (video_id, chunk_stem), chunk_frames in sorted(by_video_chunk.items()):
        chunk_frames.sort(key=lambda item: (float(item["timestamp"]), str(item["frame_id"])))
        ordered.extend(chunk_frames)

    video_number: Counter = Counter()
    sheets = []
    for video_id in VIDEO_COUNTS:
        video_frames = [frame for frame in ordered if frame["video_id"] == video_id]
        for page_start in range(0, len(video_frames), PER_PAGE):
            page_frames = video_frames[page_start:page_start + PER_PAGE]
            page_number = page_start // PER_PAGE + 1
            width = MARGIN * 2 + COLUMNS * THUMB_SIZE[0] + (COLUMNS - 1) * GAP
            height = 72 + MARGIN * 2 + ROWS * CARD_HEIGHT + (ROWS - 1) * GAP
            canvas = Image.new("RGB", (width, height), "#f3f4f6")
            draw = ImageDraw.Draw(canvas)
            draw.text((MARGIN, 20), f"{video_id}  |  page {page_number}", fill="#111827", font=_font(20))
            video_number[video_id] += len(page_frames)
            for local_index, frame in enumerate(page_frames):
                global_number = page_start + local_index + 1
                col, row = local_index % COLUMNS, local_index // COLUMNS
                x = MARGIN + col * (THUMB_SIZE[0] + GAP)
                y = 64 + row * (CARD_HEIGHT + GAP)
                draw.rounded_rectangle((x, y, x + THUMB_SIZE[0], y + CARD_HEIGHT), radius=6, fill="white", outline="#d1d5db")
                with Image.open(image_paths[str(frame["frame_id"])]) as opened:
                    thumb = ImageOps.fit(opened.convert("RGB"), THUMB_SIZE, method=Image.Resampling.LANCZOS)
                canvas.paste(thumb, (x, y))
                short_id = str(frame["frame_id"])[-22:]
                timestamp = float(frame["timestamp"])
                # Text is limited to neutral identity/time information only.
                draw.text((x + 5, y + 184), f"#{global_number:03d}  {timestamp:.2f}s  {_chunk_stem(frame)}", fill="#111827", font=_font(12))
                draw.text((x + 5, y + 204), f"id …{short_id}", fill="#4b5563", font=_font(11))
            filename = f"{video_id}-contact-sheet-{page_number:02d}.jpg"
            canvas.save(OUT_DIR / filename, format="JPEG", quality=88, optimize=True)
            sheets.append({
                "video_id": video_id,
                "page": page_number,
                "frame_count": len(page_frames),
                "filename": filename,
                "frame_ids": [str(frame["frame_id"]) for frame in page_frames],
            })
    if dict(video_number) != VIDEO_COUNTS:
        raise AssertionError(f"Contact sheet counts do not match manifest: {dict(video_number)}")
    return sheets


def build_frame_thumbnails(frames: list[dict], image_paths: dict[str, Path]) -> dict[str, str]:
    thumb_dir = OUT_DIR / "frame-thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for frame in frames:
        frame_id = str(frame["frame_id"])
        filename = hashlib.sha1(frame_id.encode()).hexdigest() + ".jpg"
        target = thumb_dir / filename
        if not target.exists():
            with Image.open(image_paths[frame_id]) as opened:
                thumb = ImageOps.fit(opened.convert("RGB"), THUMB_SIZE, method=Image.Resampling.LANCZOS)
                thumb.save(target, format="JPEG", quality=82, optimize=True)
        paths[frame_id] = f"frame-thumbnails/{filename}"
    return paths


def _labeling_mode(query: dict) -> str:
    temporal_ids = {
        "behavior_drinking_water", "behavior_approach_food_bowl", "behavior_dog_running",
        "behavior_moving", "location_couch_movement", "location_under_couch",
        "event_food_bowl_grounded_img8450", "event_water_grounded_dog_water",
    }
    return "video_temporal" if query["query_id"] in temporal_ids else "frame_visual"


def make_labeling_form(
    frames: list[dict], sheets: list[dict], image_source: dict[str, str],
    thumbnail_paths: dict[str, str], source_chunks: list[dict],
) -> None:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    queries = []
    for item in gt["queries"]:
        scope = item.get("scope") or ({"kind": "video", "video_ids": [item["video_id"]]} if item.get("video_id") else {"kind": "global", "video_ids": list(VIDEO_COUNTS)})
        queries.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "type": item.get("type", item.get("query_type")),
            "query_type": item.get("query_type", item.get("type")),
            "labeling_mode": _labeling_mode(item),
            "scope": scope,
            "video_id": item.get("video_id"),
            "source_event_id": item.get("source_event_id"),
            "event_start": item.get("event_start"),
            "event_end": item.get("event_end"),
            "source_artifacts": item.get("source_artifacts", []),
            "scope_basis": item.get("scope_basis"),
            "historical_scope_note": item.get("historical_scope_note"),
            "expected_video_ids": item.get("expected_video_ids"),
            "relevant_time_ranges": [],
            "relevant_frame_ids": [],
            "frame_judgments": {},
            "notes": "",
            "ground_truth_status": "pending_human_review",
            "human_review_complete": False,
            "reviewed_video_ids": [],
            "no_relevant_results": None,
        })
    catalog = []
    order_by_video: Counter = Counter()
    sheet_by_frame = {frame_id: sheet["filename"] for sheet in sheets for frame_id in sheet["frame_ids"]}
    chunk_by_pair = {(item["video_id"], item["chunk_stem"]): item for item in source_chunks}
    for frame in sorted(frames, key=lambda item: (str(item["video_id"]), _chunk_stem(item), float(item["timestamp"]), str(item["frame_id"]))):
        video_id = str(frame["video_id"])
        order_by_video[video_id] += 1
        catalog.append({
            "frame_number": order_by_video[video_id],
            "frame_id": frame["frame_id"],
            "video_id": video_id,
            "timestamp": float(frame["timestamp"]),
            "chunk_stem": _chunk_stem(frame),
            "contact_sheet": sheet_by_frame[str(frame["frame_id"])],
            "thumbnail": thumbnail_paths[str(frame["frame_id"])],
            "image_source": image_source[str(frame["frame_id"])],
            "source_chunk_status": chunk_by_pair[(video_id, _chunk_stem(frame))]["status"],
            "source_video_local_path": chunk_by_pair[(video_id, _chunk_stem(frame))]["playable_local_path"],
        })
    form = {
        "schema_version": gt.get("schema_version", "1.0"),
        "status": "pending_human_review",
        "description": gt.get("description"),
        "scope": {**gt["scope"], "total_unique_frames": len(catalog), "stale_frames_included": 0},
        "judgment_instructions": {**gt.get("judgment_instructions", {}), "time_tolerance": "No time tolerance is applied."},
        "queries": queries,
        "contact_sheets": sheets,
        "frame_catalog": catalog,
        "source_chunks": source_chunks,
        "range_format": {"video_id": "required", "chunk_stem": "required", "start": "seconds, inclusive", "end": "seconds, inclusive", "relevance": "relevant | not_relevant | uncertain"},
    }
    (OUT_DIR / "labeling-form.json").write_text(json.dumps(form, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_labeling_guide(queries, sheets, source_chunks)


def make_labeling_guide(queries: list[dict], sheets: list[dict], source_chunks: list[dict]) -> None:
    lines = [
        "# Retrieval Ground Truth labeling guide",
        "",
        "이 자료는 retrieval 결과와 독립적으로 실제 프레임을 보고 정답 구간을 표시하기 위한 pack입니다. contact sheet에는 원본 이미지와 중립적인 frame 번호·timestamp·chunk stem·짧은 frame ID만 표시합니다.",
        "",
        "## 판단 기준",
        "",
        "- 객체 질문(강아지/고양이/사람): 해당 객체가 실제 이미지에 보이는 구간만 relevant로 표시합니다.",
        "- 물 마시기: 혀/입을 물에 대고 마시는 모습 또는 연속 프레임에서 음수 행동이 확인되는 구간만 relevant입니다. 물그릇 옆에 있다는 사실만으로는 relevant가 아닙니다.",
        "- 사료 그릇 접근: 이전 위치보다 그릇 방향으로 이동하는 시간 흐름이 확인되어야 합니다. 이미 그릇 옆에 선 정지 frame만으로 접근이라고 판정하지 않습니다.",
        "- 움직임/뛰기: 앞뒤 frame/video에서 위치 변화가 확인되어야 합니다. 서 있거나 걷는 모습은 ‘뛰는 장면’의 정답이 아닙니다.",
        "- 소파 아래 진입: 실제로 소파 아래 영역으로 들어가는 과정 또는 소파 아래 위치가 명확해야 relevant입니다.",
        "- 위치·대상(소파 근처, 급식기 주변): 반려동물이 해당 대상/위치에 실제로 있는 구간을 표시합니다.",
        "- event-grounded query: event metadata를 정답으로 옮기지 말고 해당 영상 contact sheet의 이미지를 직접 확인합니다.",
        "- 정지 frame만으로 행동을 확정하기 어렵거나 가려짐/해상도 문제로 확신할 수 없으면 `uncertain`으로 남깁니다.",
        "",
        "## 시간 구간 입력",
        "",
        "가능하면 프레임 하나보다 행동이 확인되는 구간을 `relevant_time_ranges`에 기록합니다. start/end는 chunk-local seconds이며 양 끝을 포함합니다. 각 range에는 반드시 `video_id`와 `chunk_stem`을 함께 입력합니다. 서로 다른 chunk의 같은 timestamp는 별도 구간입니다. tolerance는 적용하지 않습니다.",
        "",
        "```json",
        '{"chunk_stem":"petcam_xxx_000","video_id":"dog_drinking_water","start":10.0,"end":18.0,"relevance":"relevant"}',
        "```",
        "",
        "불확실한 범위는 `relevance: \"uncertain\"`으로 적고 notes에 이유를 기록합니다. 정확한 장면 한두 프레임만 확인되면 `relevant_frame_ids` 또는 `frame_judgments`를 사용할 수 있습니다.",
        "",
        "## Scope 확인",
        "",
        "현재 15개 query의 scope 값은 기존 `retrieval-ground-truth-2026-09-27.json`을 그대로 따릅니다. `video_id`가 없는 query는 전체 eligible corpus를 대상으로 하는 global query이고, event-grounded 3개는 지정된 단일 video scope입니다. 다만 과거 explicit-object-search artifact에는 고양이/사람 query를 각 video에 따로 고정 실행한 기록이 있습니다. 그 과거 per-video 결과를 이번 15개 global query로 바꾸어 해석하지 않도록 구분해 두었습니다. query를 추가하거나 scope를 임의 변경하지 않았습니다.",
        "",
        "| query_id | labeling_mode | scope | query |",
        "|---|---|---|---|",
    ]
    for query in queries:
        scope = query.get("scope", {})
        scope_desc = "global (전체 eligible corpus)" if scope.get("kind") == "global" else "video: " + ", ".join(scope.get("video_ids", []))
        lines.append(f"| {query['query_id']} | {_labeling_mode(query)} | {scope_desc} | {query['query']} |")
    lines += [
        "",
        "`global` query는 네 영상 전체를 검수해야 완료할 수 있습니다. UI에서 각 video를 선택하고 ‘이 video 검수 완료’를 눌러 네 개 모두 기록하세요. event-grounded query는 지정된 video 하나만 확인하면 됩니다.",
        "",
        "## Exact source chunk playback",
        "",
    ]
    for chunk in source_chunks:
        lines.append(f"- {chunk['video_id']} / `{chunk['chunk_stem']}`: `{chunk['status']}` — `{chunk.get('playable_local_path') or 'unavailable'}`")
    lines += [
        "",
        "정확한 (video_id, chunk_stem) 쌍이 확인된 chunk만 `/tmp/retrieval-labeling-videos-2026-09-27/`에 로컬 재생용으로 내려받습니다. browser가 재생하지 못하면 위 경로를 직접 video player로 열고 chunk-local timestamp를 참고하세요.",
        "",
        "## Contact sheets",
        "",
    ]
    for sheet in sheets:
        lines.append(f"- [{sheet['filename']}]({sheet['filename']}) — {sheet['video_id']}, page {sheet['page']}, {sheet['frame_count']} frames")
    lines += [
        "",
        "모든 query의 초기 상태는 `pending_human_review`입니다. 전체 scope를 확인하고, relevant range/frame을 입력하거나 ‘검수했지만 relevant 결과 없음’을 명시해야 `labeled`로 export됩니다. 후자는 `no_relevant_results: true`로 표현합니다. 빈 배열만으로는 검수 완료가 아닙니다.",
        "",
    ]
    (OUT_DIR / "labeling-guide.md").write_text("\n".join(lines), encoding="utf-8")


def make_index(form: dict) -> None:
    embedded = json.dumps(form, ensure_ascii=False).replace("</", "<\\/")
    page = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Retrieval GT visual labeling</title>
<style>
body{font:15px system-ui,sans-serif;margin:22px;color:#111827;background:#f8fafc}header{position:sticky;top:0;background:#f8fafc;padding:12px 0;z-index:2;border-bottom:1px solid #cbd5e1}select,input,button,textarea{font:inherit;padding:6px;margin:3px}button{cursor:pointer}.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}.panel{background:white;border:1px solid #cbd5e1;border-radius:8px;padding:14px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:10px}.card{background:white;border:1px solid #d1d5db;border-radius:6px;padding:8px}.card img{width:100%;height:auto}.sheet{max-width:100%;height:auto;margin:8px 0;border:1px solid #cbd5e1}.hidden{display:none}.small{font-size:12px;color:#475569}.warning{color:#9a3412}.good{color:#166534}video{max-width:100%;max-height:55vh;background:#111}.range{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}.query-text{font-size:18px;font-weight:650}
</style></head><body>
<header><h1>Visual Ground Truth Labeling</h1><p>Human review only. No retrieval results, model labels, similarity scores, or ranking are shown.</p>
<p><a href="labeling-guide.md">Labeling guide</a> · <a href="labeling-form.json">JSON form</a></p>
<div class="row"><label>Query <select id="query"></select></label><label>Video <select id="video"></select></label><label>Chunk <select id="chunk"></select></label></div>
<div id="queryInfo" class="panel"></div><div id="scopeProgress" class="small"></div></header>
<main>
<section class="panel"><h2>Source video chunk</h2><p id="sourceInfo" class="small"></p><video id="player" controls preload="metadata"></video><div class="row"><button id="setStart">현재 재생 시각을 시작으로</button><button id="setEnd">현재 재생 시각을 끝으로</button><a id="openVideo" target="_blank" rel="noopener">외부 video player로 열기</a></div></section>
<section class="panel"><h2>Contact sheet</h2><div id="sheets"></div></section>
<section class="panel"><h2>Frame 단위 판단</h2><p>정지 이미지로 판단 가능한 질문에 사용하세요. 행동을 판단하기 어려우면 uncertain을 선택하고 video를 확인하세요.</p><div id="frames" class="grid"></div></section>
<section class="panel"><h2>Relevant time range</h2><div class="range"><label>Start (s) <input id="rangeStart" type="number" step="0.01"></label><label>End (s) <input id="rangeEnd" type="number" step="0.01"></label><label>판정 <select id="rangeRelevance"><option value="relevant">relevant</option><option value="not_relevant">not relevant</option><option value="uncertain">uncertain</option></select></label><input id="rangeNote" placeholder="uncertain이면 근거/이유 입력"><button id="addRange">구간 추가</button></div><div id="ranges"></div></section>
<section class="panel"><h2>Scope 검수 완료</h2><label><input id="noRelevant" type="checkbox"> 전체 scope를 확인했으며 relevant result가 없음</label><p><button id="markVideo">현재 video 검수 완료로 표시</button> <button id="completeQuery">현재 query 검수 완료</button></p><p id="validation" class="warning"></p></section>
<section class="panel"><h2>Export</h2><p>부분 입력도 pending 상태로 다운로드할 수 있습니다. labeled 상태는 scope 전체 검수와 명시적인 positive 또는 “relevant 없음”이 확인된 query만 받습니다.</p><button id="export">Ground Truth JSON 다운로드</button></section>
</main>
<script>
const data = __DATA__;
const allVideos = Object.keys(data.scope.video_counts);
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const querySel=$('query'), videoSel=$('video'), chunkSel=$('chunk');
for (const q of data.queries) { const o=document.createElement('option');o.value=q.query_id;o.textContent=`${q.query_id} · ${q.labeling_mode} · ${q.query}`;querySel.append(o); }
function qNow(){return data.queries.find(q=>q.query_id===querySel.value)}
function scopedVideos(q){return q.scope?.video_ids?.length?q.scope.video_ids:allVideos}
function selectedVideo(){return videoSel.value}
function chunksFor(video){return data.source_chunks.filter(c=>c.video_id===video&&c.playable_local_path)}
function renderVideos(){const q=qNow(), allowed=scopedVideos(q), previous=videoSel.value;videoSel.replaceChildren();for(const v of allowed){const o=document.createElement('option');o.value=v;o.textContent=v;videoSel.append(o)}if(allowed.includes(previous))videoSel.value=previous;renderAll()}
function renderChunkOptions(){const chunks=chunksFor(selectedVideo()), prior=chunkSel.value;chunkSel.replaceChildren();for(const c of chunks){const o=document.createElement('option');o.value=c.chunk_stem;o.textContent=`${c.chunk_stem} (${c.status})`;chunkSel.append(o)}if(chunks.some(c=>c.chunk_stem===prior))chunkSel.value=prior;renderPlayer()}
function renderPlayer(){const video=selectedVideo(), chunk=chunksFor(video).find(c=>c.chunk_stem===chunkSel.value), player=$('player');player.pause();player.removeAttribute('src');player.load();if(!chunk){$('sourceInfo').textContent='이 video의 정확한 source chunk를 찾지 못했습니다. contact sheet와 frame timestamp만으로 검수하세요.';$('openVideo').removeAttribute('href');return}const url=chunk.playable_local_url||'';player.src=url;player.load();$('sourceInfo').textContent=`Exact source: ${chunk.gcs_exact_matches?.[0]?.gcs_path||chunk.local_exact_matches?.[0]||chunk.chunk_stem} · local file: ${chunk.playable_local_path}`;$('openVideo').href=url||'#'}
function renderInfo(){const q=qNow(), mode=q.labeling_mode==='video_temporal'?'시간 흐름 확인 필요':'contact sheet 중심';$('queryInfo').innerHTML=`<div class="query-text">${esc(q.query)}</div><div>type: ${esc(q.type)} · labeling_mode: <b>${esc(q.labeling_mode)}</b> (${mode})</div><div>scope: ${esc(q.scope?.kind)} · ${esc(scopedVideos(q).join(', '))}</div><div>status: <b>${esc(q.ground_truth_status)}</b> · no_relevant_results: ${esc(q.no_relevant_results)}</div>${q.historical_scope_note?`<p class="small">${esc(q.historical_scope_note)}</p>`:''}`;const done=q.reviewed_video_ids||[];$('scopeProgress').textContent=`검수 완료 video: ${done.length}/${scopedVideos(q).length} (${done.join(', ')||'없음'})`;$('noRelevant').checked=q.no_relevant_results===true}
function renderSheets(){const v=selectedVideo();$('sheets').innerHTML=data.contact_sheets.filter(s=>s.video_id===v).map(s=>`<div><p>${esc(s.filename)} · ${s.frame_count} frames</p><a href="${esc(s.filename)}"><img class="sheet" loading="lazy" src="${esc(s.filename)}" alt="contact sheet"></a></div>`).join('')}
function renderFrames(){const q=qNow(),v=selectedVideo(),chunk=chunkSel.value;const rows=data.frame_catalog.filter(f=>f.video_id===v&&(!chunk||f.chunk_stem===chunk));$('frames').innerHTML=rows.map(f=>{const j=q.frame_judgments[f.frame_id]||{},name='frame_'+btoa(unescape(encodeURIComponent(f.frame_id))).replace(/=/g,'');return `<div class="card"><img loading="lazy" src="${esc(f.thumbnail)}" alt="frame ${f.frame_number}"><div>#${f.frame_number} · ${f.timestamp.toFixed(2)}s · ${esc(f.chunk_stem)}</div><div class="small">${esc(f.frame_id)}</div><div>${['relevant','not_relevant','uncertain'].map(value=>`<label><input type="radio" name="${name}" value="${value}" ${j.relevance===value?'checked':''}>${value}</label>`).join('')}</div><input class="frame-note" data-frame="${esc(f.frame_id)}" value="${esc(j.note||'')}" placeholder="uncertain note"></div>`}).join('');for(const radio of $('frames').querySelectorAll('input[type=radio]'))radio.addEventListener('change',e=>{const card=e.target.closest('.card'),fid=card.querySelector('.frame-note').dataset.frame,prev=qNow().frame_judgments[fid]||{};qNow().frame_judgments[fid]={...prev,relevance:e.target.value};if(e.target.value==='relevant')qNow().no_relevant_results=false;syncFrameIds();renderInfo()});for(const note of $('frames').querySelectorAll('.frame-note'))note.addEventListener('input',e=>{const q=qNow(),fid=e.target.dataset.frame,j=q.frame_judgments[fid];if(j)j.note=e.target.value})}
function syncFrameIds(){const q=qNow();q.relevant_frame_ids=Object.entries(q.frame_judgments).filter(([,j])=>j.relevance==='relevant').map(([fid])=>fid)}
function renderRanges(){const q=qNow();$('ranges').innerHTML=q.relevant_time_ranges.map((r,i)=>`<div class="range">${esc(r.video_id)} / ${esc(r.chunk_stem)} · ${r.start}–${r.end}s · ${esc(r.relevance)} ${r.note?`· ${esc(r.note)}`:''}<button data-range="${i}">삭제</button></div>`).join('');for(const b of $('ranges').querySelectorAll('button'))b.onclick=()=>{q.relevant_time_ranges.splice(Number(b.dataset.range),1);renderRanges()}}
function renderAll(){if(!qNow())return;renderInfo();renderChunkOptions();renderSheets();renderFrames();renderRanges();$('validation').textContent=''}
querySel.onchange=()=>renderVideos();videoSel.onchange=()=>renderAll();chunkSel.onchange=()=>{renderPlayer();renderFrames()};
$('setStart').onclick=()=>{$('rangeStart').value=Number($('player').currentTime||0).toFixed(2)};$('setEnd').onclick=()=>{$('rangeEnd').value=Number($('player').currentTime||0).toFixed(2)};
    $('addRange').onclick=()=>{const q=qNow(),start=Number($('rangeStart').value),end=Number($('rangeEnd').value),video=selectedVideo(),chunk=chunkSel.value,relevance=$('rangeRelevance').value,note=$('rangeNote').value.trim();if(!chunk||!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<start){$('validation').textContent='정확한 chunk를 선택하고 0 이상 start ≤ end를 입력하세요.';return}if(relevance==='uncertain'&&!note){$('validation').textContent='uncertain 구간에는 note를 입력하세요.';return}q.relevant_time_ranges.push({video_id:video,chunk_stem:chunk,start,end,relevance,...(note?{note}:{})});if(relevance==='relevant')q.no_relevant_results=false;$('rangeNote').value='';renderRanges();renderInfo()};
$('markVideo').onclick=()=>{const q=qNow(),v=selectedVideo();q.reviewed_video_ids=[...new Set([...(q.reviewed_video_ids||[]),v])];renderInfo()};
$('noRelevant').onchange=e=>{qNow().no_relevant_results=e.target.checked?true:null};
function validateComplete(q){const scope=scopedVideos(q),reviewed=q.reviewed_video_ids||[];if(!scope.every(v=>reviewed.includes(v)))return 'scope의 모든 video를 검수 완료로 표시해야 합니다.';const positives=(q.relevant_time_ranges||[]).some(r=>r.relevance==='relevant')||(q.relevant_frame_ids||[]).length>0;if(q.no_relevant_results===true&&positives)return 'relevant 입력과 no_relevant_results=true를 동시에 사용할 수 없습니다.';if(q.no_relevant_results!==true&&!positives)return 'relevant 구간/frame을 입력하거나 relevant result 없음에 체크하세요.';for(const r of q.relevant_time_ranges||[])if(r.relevance==='uncertain'&&!r.note)return 'uncertain time range에 note가 필요합니다.';for(const j of Object.values(q.frame_judgments||{}))if(j.relevance==='uncertain'&&!j.note)return 'uncertain frame에 note가 필요합니다.';return ''}
$('completeQuery').onclick=()=>{const q=qNow();syncFrameIds();const positive=(q.relevant_time_ranges||[]).some(r=>r.relevance==='relevant')||(q.relevant_frame_ids||[]).length>0;if(!$('noRelevant').checked&&positive)q.no_relevant_results=false;const err=validateComplete(q);if(err){$('validation').textContent=err;return}q.no_relevant_results=$('noRelevant').checked;q.human_review_complete=true;q.ground_truth_status='labeled';$('validation').className='good';$('validation').textContent='현재 query가 labeled로 표시되었습니다. export하여 저장하세요.';renderInfo()};
$('export').onclick=()=>{syncFrameIds();const cloned=JSON.parse(JSON.stringify(data));delete cloned.frame_catalog;delete cloned.contact_sheets;delete cloned.source_chunks;const allDone=cloned.queries.every(q=>q.ground_truth_status==='labeled');cloned.status=allDone?'labeled':'pending_human_review';const blob=new Blob([JSON.stringify(cloned,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='retrieval-ground-truth-2026-09-27.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
renderVideos();
</script></body></html>
'''.replace("__DATA__", embedded)
    (OUT_DIR / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frames = manifest["frames"]
    validate_manifest(frames)
    image_paths, image_source = acquire_images(frames)
    sheets = build_sheets(frames, image_paths)
    thumbnails = build_frame_thumbnails(frames, image_paths)
    source_chunks = acquire_source_chunks(frames)
    make_labeling_form(frames, sheets, image_source, thumbnails, source_chunks)
    form = json.loads((OUT_DIR / "labeling-form.json").read_text(encoding="utf-8"))
    for chunk in form["source_chunks"]:
        path = chunk.get("playable_local_path")
        chunk["playable_local_url"] = Path(path).as_uri() if path else None
    # Store file URLs for browser playback, retaining the exact paths for manual opening.
    (OUT_DIR / "labeling-form.json").write_text(json.dumps(form, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_index(form)
    print(f"Frames: {len(frames)} unique; contact sheets: {len(sheets)}")
    for video_id in VIDEO_COUNTS:
        print(f"{video_id}: {VIDEO_COUNTS[video_id]} frames")
    print(f"Pack: {OUT_DIR}")


if __name__ == "__main__":
    main()
