from fastapi import FastAPI, HTTPException
from celery import Celery
from pydantic import BaseModel
import os

from pipeline.rag_chain import run_rag_query

app = FastAPI()

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


@app.get("/health")
def health():
    return {"status": "ok"}


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

    rag_result = run_rag_query(
        query=request.query,
        video_id=request.video_id,
        top_k=top_k,
        user_id=request.user_id,
    )

    return {
        "status": "ok",
        "query": request.query,
        "video_id": request.video_id,
        "top_k": top_k,
        "answer": rag_result["answer"],
        "results": rag_result["results"],
        "used_llm": rag_result["used_llm"],
    }