from celery import Celery
from pipeline.frame_extractor import extract_frames
from pipeline.s3_uploader import download_video, upload_frame
from pipeline.vector_store import index_frame
from db.metadata import init_db, insert_scene
import os
from pathlib import Path

celery_app = Celery(
    'tasks',
    broker='amqp://guest:guest@localhost:5672/'
)


def upload_frame_to_s3(local_frame_path, video_id):
    """프레임을 S3에 업로드"""
    return upload_frame(local_frame_path, video_id)

@celery_app.task(name='pipeline.tasks.process_chunk')
def process_chunk(chunk_path):
    """
    청크 수신 → S3 다운로드 → motion 감지 → 프레임 추출
    """
    print(f"[Task 시작] {chunk_path}")
    chunk_key = Path(chunk_path)
    video_id = chunk_key.parent.name if chunk_key.parent.name != "chunks" else chunk_key.stem
    frame_output_dir = Path("pipeline/frames") / video_id

    # 1. S3에서 청크 다운로드
    local_path = f"/tmp/{os.path.basename(chunk_path)}"
    download_video(chunk_path, local_path)
    print(f"다운로드 완료: {local_path}")

    # 2. 프레임 추출 (motion 감지 포함)
    frames = extract_frames(
        local_path,
        output_dir=str(frame_output_dir),
        frame_prefix=Path(local_path).stem,
    )
    print(f"프레임 {len(frames)}개 추출 완료")

    # 3. 프레임을 S3/ChromaDB에 저장하고 로컬 파일 삭제
    uploaded_frames = []
    for frame in frames:
        frame_s3_key = upload_frame_to_s3(
            frame["frame_path"],
            video_id,
        )
        frame_id = index_frame(
            frame["frame_path"],
            frame_root=frame_output_dir,
            video_id=video_id,
            s3_key=frame_s3_key,
        )
        Path(frame["frame_path"]).unlink()
        print(f"로컬 프레임 삭제: {frame['frame_path']}")
        uploaded_frames.append({
            **frame,
            "frame_id": frame_id,
            "s3_key": frame_s3_key,
        })
    print(f"프레임 {len(uploaded_frames)}개 S3/ChromaDB 저장 및 로컬 삭제 완료")

    # 4. SQLite에 메타데이터 저장
    init_db()
    for frame in uploaded_frames:
        insert_scene(
            video_id=video_id,
            start_time=frame["timestamp"],
            end_time=frame["timestamp"] + 1.0,
            s3_key=frame["s3_key"]
        )
    print(f"메타데이터 {len(uploaded_frames)}개 저장 완료!")

    # 5. 임시 파일 삭제
    os.remove(local_path)
    print(f"임시 파일 삭제: {local_path}")

    return {
        "chunk_path": chunk_path,
        "frame_count": len(uploaded_frames),
        "frames": uploaded_frames
    }
