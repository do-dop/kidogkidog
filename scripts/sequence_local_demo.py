"""Run the reviewed sequence demo from local MP4, without MySQL or GCS.

Usage: python -m uvicorn scripts.sequence_local_demo:app --host 127.0.0.1 --port 8000
The source videos are ~/Downloads/funny_moments.mp4 and funny_moments2.mp4. Only
the reviewed clips are presented; this is a repeatable local demonstration, not a model eval.
"""

import csv
import importlib
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from db import behavior_events as behavior_db

os.environ["QUESTION_ROUTER_MODE"] = "off"
search_module = importlib.import_module("api.routers.search")
search_module.insert_search_log = lambda **_kwargs: None
search_module.upsert_user_frequent_query = lambda **_kwargs: None

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/private/tmp/kidog-sequence-local-demo")

# 영상별 로컬 원본, 청크 파일명 기준(촬영 시작 시각), 검수된 근거가 있는 60초 청크 번호.
VIDEOS = {
    "funny_moments": {
        "source": Path.home() / "Downloads/funny_moments.mp4",
        "base": "petcam_20261014_161557",
        "chunks": (0, 4, 5),
    },
    "funny_moments2": {
        "source": Path.home() / "Downloads/funny_moments2.mp4",
        "base": "petcam_20261014_173000",
        "chunks": (2, 3, 4, 5, 7, 10),
    },
}


def _events():
    with (ROOT / "docs/sequence-accuracy-labels-2026-11-05.csv").open(encoding="utf-8", newline="") as stream:
        labels = [row for row in csv.DictReader(stream) if row["video_id"] in VIDEOS]
    rows = []
    for row in labels:
        video_id = row["video_id"]
        base = VIDEOS[video_id]["base"]
        chunk_index = int(float(row["representative_second"]) // 60)
        second = float(row["representative_second"]) - chunk_index * 60
        rows.append({
            "id": int(row["event_id"]),
            "video_id": video_id,
            "start_time": second,
            "end_time": second + 8,
            "summary": row["reason"],
            "action": row["reason"],
            "confidence": 0.8,
            "source_frames": [
                {
                    "s3_key": f"frames/{video_id}/{base}_{chunk_index:03d}_frame_{second + offset:.2f}.jpg",
                    "timestamp": second + offset,
                    "frame_id": f"demo-{row['event_id']}-{offset:g}",
                }
                for offset in (0, 2, 4)
            ],
        })
    return rows


EVENTS = _events()


def _local_events(video_id=None):
    return [row.copy() for row in EVENTS if video_id is None or row["video_id"] == video_id]


behavior_db.get_behavior_events_for_aggregation = _local_events

# 이 데모에는 일반 검색용 벡터 인덱스와 완전한 행동 타임라인이 없다.
# 다른 질문에 그럴듯한 오답을 내는 대신, 데모 범위를 화면에 명시한다.
from pipeline.search import rag_chain
from pipeline.search.sequence_query import is_sequence_query

_run_rag_query = rag_chain.run_rag_query


def _demo_rag_query(*, query: str, video_id: str | None = None, **kwargs):
    if is_sequence_query(query):
        return _run_rag_query(query=query, video_id=video_id, **kwargs)
    return {
        "query": query,
        "video_id": video_id,
        "answer": (
            "이 로컬 데모는 검수된 시퀀스 질문만 재현합니다. "
            "일반 장면 검색과 횟수 질문은 전체 영상 인덱스·행동 기록이 연결된 서비스에서 확인해야 해요."
        ),
        "results": [],
        "evidence_items": [],
        "behavior_events": [],
        "used_llm": False,
        "query_type": "demo_unsupported",
    }


rag_chain.run_rag_query = _demo_rag_query


def _clip_path(video_id: str, index: int) -> Path:
    return OUTPUT / f"{VIDEOS[video_id]['base']}_{index:03d}.mp4"


def _find_clip(key: str) -> tuple[str, int] | None:
    """청크·프레임 키에서 (영상, 청크 번호)를 찾는다."""
    for video_id, info in VIDEOS.items():
        for index in info["chunks"]:
            if f"{info['base']}_{index:03d}" in key:
                return video_id, index
    return None


def _prepare_clips():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for video_id, info in VIDEOS.items():
        if not info["source"].is_file():
            print(f"로컬 원본 영상이 없어 {video_id}는 건너뜁니다: {info['source']}", flush=True)
            info["chunks"] = ()
            continue
        for index in info["chunks"]:
            clip = _clip_path(video_id, index)
            if not clip.is_file():
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(index * 60), "-i", str(info["source"]), "-t", "60", "-an",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
                    "-movflags", "+faststart", str(clip),
                ], check=True)
            thumb = clip.with_suffix(".jpg")
            if not thumb.is_file():
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", "2", "-i", str(clip), "-frames:v", "1", "-q:v", "3", str(thumb),
                ], check=True)


_prepare_clips()
app = FastAPI(title="Kidogkidog local sequence demo")
app.include_router(search_module.router)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "local-reviewed-sequence-demo"}


@app.get("/recordings/chunks")
def recording_chunks():
    chunks = []
    for video_id, info in VIDEOS.items():
        start = datetime.strptime(info["base"].split("_", 1)[1], "%Y%m%d_%H%M%S")
        for index in info["chunks"]:
            clip = _clip_path(video_id, index)
            key = f"chunks/{video_id}/{clip.name}"
            recorded = start + timedelta(minutes=index)
            chunks.append({
                "id": key, "s3_key": key, "video_id": video_id, "filename": clip.name,
                "chunk_index": index, "start_seconds": index * 60, "duration_seconds": 60,
                "recorded_at": recorded.isoformat(), "recording_date": recorded.date().isoformat(),
                "time_label": recorded.strftime("%H:%M"), "period": "오후",
                "size": clip.stat().st_size, "last_modified": None, "url": "",
                "media_path": f"/media/s3?key={key}", "thumbnail_url": None,
                "thumbnail_media_path": f"/media/thumbnail?key={key}",
            })
    return {"status": "ok", "prefix": "chunks/", "count": len(chunks), "chunks": chunks}


_FRAME_KEY = re.compile(r"_(\d{3})_frame_([0-9.]+)\.jpg$")


def _frame_path(key: str) -> Path | None:
    """프레임 키(영상·청크 번호·초)로 로컬 청크에서 JPG를 한 번만 뽑아 캐시한다."""
    match = _FRAME_KEY.search(key)
    found = _find_clip(key)
    if not match or found is None:
        return None
    video_id, index = found
    second = float(match.group(2))
    frame = OUTPUT / f"{VIDEOS[video_id]['base']}_{index:03d}_frame_{second:.2f}.jpg"
    if not frame.is_file():
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(second), "-i", str(_clip_path(video_id, index)), "-frames:v", "1", "-q:v", "3", str(frame),
        ], check=True)
    return frame


@app.get("/media/s3")
def media(key: str):
    if key.endswith(".jpg") and "/frames/" in f"/{key}":
        frame = _frame_path(key)
        if frame is None:
            raise HTTPException(status_code=404)
        return FileResponse(frame, media_type="image/jpeg")
    found = _find_clip(key)
    if found is None:
        raise HTTPException(status_code=404)
    return FileResponse(_clip_path(*found), media_type="video/mp4")


@app.get("/media/thumbnail")
def thumbnail(key: str):
    found = _find_clip(key)
    if found is None:
        raise HTTPException(status_code=404)
    return FileResponse(_clip_path(*found).with_suffix(".jpg"), media_type="image/jpeg")


@app.get("/recordings/events")
def recording_events():
    return {"status": "ok", "events": []}


@app.get("/suggestions")
def suggestions():
    questions = [
        "행동을 시간순으로 알려줘",
        "수영장 장면 다음에는 뭐가 나와?",
        "문을 나간 다음 뭐했어?",
    ]
    return {
        "status": "ok", "questions": questions, "question_sources": [],
        "behavior_events": [], "indexed_frame_count": len(EVENTS),
        "top_queries": [], "recent_queries": [],
    }
