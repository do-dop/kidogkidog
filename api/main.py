import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.routers import frames, media, recordings, search, suggestions, upload
from db.schema import init_db


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _preload_clip():
    """무거운 임포트(torch, clip)와 모델 로드를 첫 검색 요청 전에 시작한다.

    첫 요청이 와서야 불러오면 콜드 스타트에서 그만큼이 사용자 대기 시간에 더해진다.
    실패해도 서비스는 떠야 하고, 첫 검색 때 다시 시도한다.
    """
    try:
        from pipeline.models.clip_embedder import load_clip_model

        load_clip_model()
    except Exception as exc:
        print(f"CLIP 사전 로드 실패 (첫 검색 때 다시 시도): {exc}", flush=True)


@app.on_event("startup")
def startup():
    init_db()
    if os.getenv("PRELOAD_CLIP", "true").lower() in {"1", "true", "yes"}:
        threading.Thread(target=_preload_clip, name="clip-preload", daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok"}


if os.getenv("METRICS_ENABLED", "false").lower() in {"1", "true", "yes"}:
    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(media.router)
app.include_router(recordings.router)
app.include_router(frames.router)
app.include_router(suggestions.router)
app.include_router(upload.router)
app.include_router(search.router)
