import hmac
import os
from pathlib import PurePosixPath

from fastapi import APIRouter, Header, HTTPException
from kombu.exceptions import OperationalError

from api.celery_app import celery_app
from api.schemas import ChunkRequest
from pipeline.storage.gcs_uploader import object_exists


router = APIRouter(tags=["upload"])


@router.post("/upload")
def upload(request: ChunkRequest, x_upload_token: str | None = Header(default=None)):
    expected_token = os.getenv("UPLOAD_API_TOKEN")
    if expected_token and not hmac.compare_digest(x_upload_token or "", expected_token):
        raise HTTPException(status_code=401, detail="업로드 인증이 필요합니다.")

    object_key = request.video_path or request.chunk_path

    if not object_key:
        raise HTTPException(
            status_code=400,
            detail="chunk_path 또는 video_path가 필요합니다."
        )

    normalized = str(PurePosixPath(object_key))
    if (
        object_key != normalized
        or "\\" in object_key
        or ".." in PurePosixPath(object_key).parts
        or not object_key.startswith(("chunks/", "videos/"))
        or len(PurePosixPath(object_key).parts) < 2
    ):
        raise HTTPException(status_code=400, detail="올바른 GCS 영상 경로가 필요합니다.")

    if not object_exists(object_key):
        raise HTTPException(status_code=404, detail="GCS 영상을 찾을 수 없습니다.")

    try:
        task = celery_app.send_task("pipeline.tasks.process_chunk", args=[object_key])
    except OperationalError as exc:
        raise HTTPException(status_code=503, detail="작업 큐에 연결할 수 없습니다.") from exc

    return {
        "status": "received",
        "task_id": task.id,
        "s3_path": object_key,
    }
