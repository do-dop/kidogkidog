from fastapi import FastAPI
from celery import Celery
from pydantic import BaseModel

app = FastAPI()

celery_app = Celery(
    'tasks',
    broker='amqp://guest:guest@rabbitmq:5672/'
)

class ChunkRequest(BaseModel):
    chunk_path: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
def upload(request: ChunkRequest):
    # Celery task 큐에 등록
    celery_app.send_task('pipeline.tasks.process_chunk', args=[request.chunk_path])
    return {"status": "received", "chunk_path": request.chunk_path}

@app.post("/query")
def query():
    # 자연어 쿼리 수신 (나중에 구현)
    return {"status": "query received"}