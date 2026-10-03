import os

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


@app.on_event("startup")
def startup():
    init_db()


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
