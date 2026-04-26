from celery import Celery
from pipeline.motion_detector import detect_motion
from pipeline.frame_extractor import extract_frames
from pipeline.s3_uploader import download_video
import os

celery_app = Celery(
    'tasks',
    broker='amqp://guest:guest@localhost:5672/'
)

@celery_app.task(name='pipeline.tasks.process_chunk')
def process_chunk(chunk_path):
    """
    청크 수신 → S3 다운로드 → motion 감지 → 프레임 추출
    """
    print(f"[Task 시작] {chunk_path}")

    # 1. S3에서 청크 다운로드
    local_path = f"/tmp/{os.path.basename(chunk_path)}"
    download_video(chunk_path, local_path)
    print(f"다운로드 완료: {local_path}")

    # 2. 프레임 추출 (motion 감지 포함)
    frames = extract_frames(local_path)
    print(f"프레임 {len(frames)}개 추출 완료")

    # 3. 임시 파일 삭제
    os.remove(local_path)
    print(f"임시 파일 삭제: {local_path}")

    return {
        "chunk_path": chunk_path,
        "frame_count": len(frames),
        "frames": frames
    }