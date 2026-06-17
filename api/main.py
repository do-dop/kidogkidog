from datetime import datetime, timedelta, timezone
import hashlib
import mimetypes
from pathlib import Path
import re
import tempfile
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from celery import Celery
from pydantic import BaseModel
import os

from db.metadata import (
    delete_user_frequent_query,
    get_scene_records,
    get_user_recent_queries,
    get_user_top_queries,
    init_db,
    insert_search_log,
    upsert_user_frequent_query,
)
from pipeline.query_suggester import get_suggestion_behavior_events, suggest_query_items
from pipeline.rag_chain import run_rag_query
from pipeline.vector_store import get_indexed_frames, index_frames
from pipeline.s3_uploader import (
    create_presigned_url,
    download_bytes,
    download_range,
    download_video,
    get_object_metadata,
    list_objects,
)

app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = PROJECT_ROOT / "pipeline" / "frames"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

celery_app = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672/")
)


class ChunkRequest(BaseModel):
    chunk_path: str | None = None
    video_path: str | None = None


class QueryRequest(BaseModel):
    query: str
    video_id: str | None = None
    top_k: int = 5
    user_id: str | None = None
    source_event_id: int | None = None
    event_start: float | None = None
    event_end: float | None = None


class SuggestionRequest(BaseModel):
    user_id: str | None = None
    video_id: str | None = None
    limit: int = 5


class FrequentQueryDeleteRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


def get_chunk_index(s3_key):
    match = re.search(r"_(\d{3})\.mp4$", Path(s3_key).name)
    if not match:
        return 0

    return int(match.group(1))


def get_chunk_recorded_at(s3_key, last_modified=None):
    filename = Path(s3_key).name
    match = re.search(r"petcam_(\d{8})_(\d{6})_(\d{3})\.mp4$", filename)

    if match:
        base = datetime.strptime(
            f"{match.group(1)}{match.group(2)}",
            "%Y%m%d%H%M%S",
        )
        return base + timedelta(seconds=int(match.group(3)) * 60)

    if last_modified:
        if last_modified.tzinfo:
            return last_modified.astimezone(timezone.utc).replace(tzinfo=None)
        return last_modified

    return None


def get_chunk_period_from_hour(hour):

    if hour < 12:
        return "오전"
    if hour < 17:
        return "오후"
    if hour < 21:
        return "저녁"
    return "야간"


def format_timestamp(dt):
    if not dt:
        return None

    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc)

    return dt.isoformat()


def media_path_for_key(s3_key):
    return f"/media/s3?key={quote(s3_key)}"


def thumbnail_path_for_key(s3_key):
    return f"/media/thumbnail?key={quote(s3_key)}"


def find_thumbnail_key(video_id, chunk_key):
    chunk_stem = Path(chunk_key).stem

    try:
        scenes = get_scene_records(video_id=video_id)
    except Exception:
        return None

    for scene in scenes:
        s3_key = scene.get("s3_key")

        if s3_key and f"{chunk_stem}_frame_" in s3_key:
            return s3_key

    return None


@app.get("/media/thumbnail")
def get_s3_video_thumbnail(key: str):
    if not key:
        raise HTTPException(status_code=400, detail="key가 필요합니다.")

    cache_dir = Path(tempfile.gettempdir()) / "kidogkidog_thumbnails"
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    thumbnail_path = cache_dir / f"{digest}.jpg"

    if not thumbnail_path.exists():
        video_path = cache_dir / f"{digest}.mp4"

        try:
            download_video(key, str(video_path))

            import cv2

            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            cap.release()

            if not ok:
                raise ValueError("영상 첫 프레임을 읽을 수 없습니다.")

            cv2.imwrite(str(thumbnail_path), frame)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"썸네일을 생성하지 못했습니다: {exc}",
            ) from exc
        finally:
            try:
                video_path.unlink(missing_ok=True)
            except Exception:
                pass

    return Response(
        content=thumbnail_path.read_bytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/media/s3")
def get_s3_media(key: str, request: Request):
    if not key:
        raise HTTPException(status_code=400, detail="key가 필요합니다.")

    media_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    try:
        if range_header:
            metadata = get_object_metadata(key)
            total_size = int(metadata["ContentLength"])
            range_value = range_header.replace("bytes=", "", 1)
            start_text, _, end_text = range_value.partition("-")
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else min(start + 1024 * 1024 - 1, total_size - 1)
            end = min(end, total_size - 1)

            if start > end or start >= total_size:
                return Response(
                    status_code=416,
                    headers={
                        "Content-Range": f"bytes */{total_size}",
                        "Accept-Ranges": "bytes",
                    },
                )

            data = download_range(key, start, end)

            return Response(
                content=data,
                status_code=206,
                media_type=media_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{total_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(data)),
                    "Cache-Control": "private, max-age=300",
                },
            )

        data = download_bytes(key)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"S3 객체를 불러오지 못했습니다: {exc}",
        ) from exc

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{Path(key).name}"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, max-age=300",
        },
    )


@app.get("/recordings/chunks")
def list_recording_chunks(video_id: str | None = None):
    prefix = f"chunks/{video_id}/" if video_id else "chunks/"

    try:
        objects = list_objects(prefix)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"S3 청크 목록을 불러오지 못했습니다: {exc}",
        ) from exc

    chunks = []

    for obj in objects:
        key = obj["Key"]

        if not key.lower().endswith(".mp4"):
            continue

        parts = Path(key).parts
        chunk_video_id = parts[1] if len(parts) >= 2 and parts[0] == "chunks" else "unknown"
        index = get_chunk_index(key)
        start_seconds = index * 60
        recorded_at = get_chunk_recorded_at(key, obj.get("LastModified"))
        thumbnail_key = find_thumbnail_key(chunk_video_id, key)

        chunks.append({
            "id": key,
            "s3_key": key,
            "video_id": chunk_video_id,
            "filename": Path(key).name,
            "chunk_index": index,
            "start_seconds": start_seconds,
            "duration_seconds": 60,
            "recorded_at": recorded_at.isoformat() if recorded_at else None,
            "recording_date": recorded_at.date().isoformat() if recorded_at else None,
            "time_label": recorded_at.strftime("%H:%M") if recorded_at else None,
            "period": get_chunk_period_from_hour(recorded_at.hour) if recorded_at else "오전",
            "size": obj.get("Size", 0),
            "last_modified": format_timestamp(obj.get("LastModified")),
            "url": create_presigned_url(key, expires_in=3600),
            "media_path": media_path_for_key(key),
            "thumbnail_s3_key": thumbnail_key,
            "thumbnail_url": create_presigned_url(thumbnail_key, expires_in=3600) if thumbnail_key else None,
            "thumbnail_media_path": media_path_for_key(thumbnail_key) if thumbnail_key else thumbnail_path_for_key(key),
        })

    chunks.sort(key=lambda item: (item["video_id"], item["chunk_index"], item["filename"]))

    return {
        "status": "ok",
        "prefix": prefix,
        "count": len(chunks),
        "chunks": chunks,
    }


@app.get("/recordings/events")
def list_recording_events(video_id: str | None = None, limit: int = 50):
    safe_limit = max(1, min(limit, 5))

    try:
        events = get_suggestion_behavior_events(video_id=video_id, limit=safe_limit)
    except Exception as exc:
        if "no such table: behavior_events" in str(exc):
            return {
                "status": "ok",
                "count": 0,
                "events": [],
            }

        raise HTTPException(
            status_code=500,
            detail=f"행동 이벤트 메타데이터를 불러오지 못했습니다: {exc}",
        ) from exc

    return {
        "status": "ok",
        "count": len(events),
        "events": events,
    }


@app.post("/frames/reindex")
def reindex_local_frames(video_id: str | None = None):
    if not FRAMES_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=f"프레임 폴더가 없습니다: {FRAMES_DIR}",
        )

    try:
        if video_id:
            target_dir = FRAMES_DIR / video_id
            if not target_dir.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"영상 프레임 폴더가 없습니다: {target_dir}",
                )
            index_frames(str(target_dir), video_id=video_id)
        else:
            index_frames(str(FRAMES_DIR))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"로컬 프레임 인덱스 갱신 실패: {exc}",
        ) from exc

    return {
        "status": "ok",
        "video_id": video_id,
        "frame_root": str(FRAMES_DIR if not video_id else FRAMES_DIR / video_id),
    }


@app.get("/suggestions")
def get_suggestions(
    user_id: str | None = None,
    video_id: str | None = None,
    limit: int = 5,
):
    safe_limit = max(1, min(limit, 10))

    try:
        question_items = suggest_query_items(
            user_id=user_id,
            video_id=video_id,
            limit=safe_limit,
        )
        questions = [item["question"] for item in question_items]
        behavior_events = get_suggestion_behavior_events(
            video_id=video_id,
            limit=5,
        )
        indexed_frame_count = len(get_indexed_frames(video_id=video_id))
        top_queries = get_user_top_queries(user_id=user_id, limit=5) if user_id else []
        recent_queries = get_user_recent_queries(user_id=user_id, limit=5) if user_id else []
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"추천 질문을 생성하지 못했습니다: {exc}",
        ) from exc

    return {
        "status": "ok",
        "video_id": video_id,
        "questions": questions,
        "question_sources": question_items,
        "behavior_events": behavior_events,
        "indexed_frame_count": indexed_frame_count,
        "top_queries": top_queries,
        "recent_queries": recent_queries,
    }


@app.post("/suggestions")
def post_suggestions(request: SuggestionRequest):
    return get_suggestions(
        user_id=request.user_id,
        video_id=request.video_id,
        limit=request.limit,
    )


@app.delete("/users/{user_id}/frequent-queries")
def delete_frequent_query(user_id: str, request: FrequentQueryDeleteRequest):
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id가 필요합니다.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query가 필요합니다.")

    try:
        deleted_count = delete_user_frequent_query(
            user_id=user_id,
            query_raw=request.query,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"자주 찾는 검색어를 삭제하지 못했습니다: {exc}",
        ) from exc

    return {
        "status": "ok",
        "deleted_count": deleted_count,
        "query": request.query,
    }


@app.post("/upload")
def upload(request: ChunkRequest):
    s3_path = request.video_path or request.chunk_path

    if not s3_path:
        raise HTTPException(
            status_code=400,
            detail="chunk_path 또는 video_path가 필요합니다."
        )

    celery_app.send_task("pipeline.tasks.process_chunk", args=[s3_path])

    return {
        "status": "received",
        "s3_path": s3_path,
    }


@app.post("/query")
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="query가 필요합니다."
        )

    top_k = max(1, min(request.top_k, 10))

    import time

    started_at = time.perf_counter()

    try:
        rag_result = run_rag_query(
            query=request.query,
            video_id=request.video_id,
            top_k=top_k,
            user_id=request.user_id,
            source_event_id=request.source_event_id,
            event_start=request.event_start,
            event_end=request.event_end,
        )
    finally:
        latency_ms = int((time.perf_counter() - started_at) * 1000)

    if request.user_id:
        result_count = len(rag_result.get("results", []))
        try:
            insert_search_log(
                user_id=request.user_id,
                query_raw=request.query,
                video_id=request.video_id,
                top_k=top_k,
                result_count=result_count,
                latency_ms=latency_ms,
            )
            upsert_user_frequent_query(
                user_id=request.user_id,
                query_raw=request.query,
            )
        except Exception as exc:
            print(f"검색 로그 저장 실패: {exc}", flush=True)

    return {
        "status": "ok",
        "query": request.query,
        "video_id": request.video_id,
        "top_k": top_k,
        "answer": rag_result["answer"],
        "results": rag_result["results"],
        "evidence_items": rag_result.get("evidence_items", rag_result["results"]),
        "behavior_events": rag_result.get("behavior_events", []),
        "used_llm": rag_result["used_llm"],
    }
