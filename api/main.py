from fastapi import FastAPI, HTTPException
from celery import Celery
from pydantic import BaseModel
import os

app = FastAPI()

celery_app = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@rabbitmq:5672/')
)

class ChunkRequest(BaseModel):
    chunk_path: str = None
    video_path: str = None

class QueryRequest(BaseModel):
    query: str
    video_id: str = None  # 특정 영상만 검색할 때 (없으면 전체 검색)
    top_k: int = 5        # 상위 몇 개 결과 반환

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
def upload(request: ChunkRequest):
    s3_path = request.video_path or request.chunk_path
    if not s3_path:
        raise HTTPException(status_code=400, detail="chunk_path 또는 video_path가 필요합니다.")

    # Celery task 큐에 등록
    celery_app.send_task('pipeline.tasks.process_chunk', args=[s3_path])
    return {"status": "received", "s3_path": s3_path}

@app.post("/query")
def query(request: QueryRequest):
    # 자연어 쿼리 수신 (나중에 구현)
    return {
        "status": "ok",
        "query": request.query,
        "results": []
    }
