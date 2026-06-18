from fastapi import APIRouter, HTTPException

from api.celery_app import celery_app
from api.schemas import ChunkRequest


router = APIRouter(tags=["upload"])


@router.post("/upload")
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
