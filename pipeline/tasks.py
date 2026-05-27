from celery import Celery
from pipeline.frame_extractor import extract_frames
from pipeline.s3_uploader import download_video, upload_frame
from pipeline.vector_store import index_frame
from pipeline.yolo_detector import detect_objects
from db.metadata import init_db, insert_scene
import os
import json
from pathlib import Path

FRAME_ROOT = Path("pipeline/frames")

celery_app = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@localhost:5672/')
)


def upload_frame_to_s3(local_frame_path, video_id):
    """프레임을 S3에 업로드"""
    return upload_frame(local_frame_path, video_id)


def index_frame_to_chromadb(local_frame_path, video_id, s3_key=None, object_labels=None):
    """프레임을 CLIP 임베딩 후 ChromaDB에 저장"""
    return index_frame(
        local_frame_path,
        frame_root=FRAME_ROOT,
        video_id=video_id,
        s3_key=s3_key,
        object_labels=object_labels,
    )


def get_video_id_from_s3_key(s3_key):
    key = Path(s3_key)
    parts = key.parts
    if len(parts) >= 2 and parts[0] == "chunks":
        return parts[1]
    if len(parts) >= 2 and parts[0] == "videos":
        return key.stem
    return key.stem


@celery_app.task(name='pipeline.tasks.process_chunk')
def process_chunk(chunk_path):
    """
    S3 영상/청크 수신
    → 다운로드
    → motion 감지
    → 프레임 추출
    → S3 업로드
    → YOLO 객체 감지
    → CLIP 임베딩
    → ChromaDB 저장
    → SQLite 메타데이터 저장
    """
    print(f"[Task 시작] {chunk_path}")
    video_id = get_video_id_from_s3_key(chunk_path)
    frame_output_dir = FRAME_ROOT / video_id

    # 1. S3에서 영상/청크 다운로드
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

    # 3. 프레임을 먼저 S3에 모두 업로드
    uploaded_frames = []
    for frame in frames:
        frame_s3_key = upload_frame_to_s3(
            frame["frame_path"],
            video_id,
        )
        uploaded_frames.append({
            **frame,
            "s3_key": frame_s3_key,
        })
    print(f"프레임 {len(uploaded_frames)}개 S3 업로드 완료")

    # 4. 업로드된 프레임에 YOLO 객체 감지 + CLIP/ChromaDB 저장 + 로컬 파일 삭제
    indexed_frames = []

    for i, frame in enumerate(uploaded_frames, start=1):
        print(
            f"YOLO 객체 감지 시작: {i}/{len(uploaded_frames)} - {frame['frame_path']}",
            flush=True,
        )

        object_labels = detect_objects(frame["frame_path"])

        print(
            f"YOLO 객체 감지 완료: {i}/{len(uploaded_frames)} - {object_labels}",
            flush=True,
        )

        print(
            f"CLIP/ChromaDB 인덱싱 시작: {i}/{len(uploaded_frames)} - {frame['frame_path']}",
            flush=True,
        )

        frame_id = index_frame_to_chromadb(
            frame["frame_path"],
            video_id=video_id,
            s3_key=frame["s3_key"],
            object_labels=object_labels,
        )

        print(f"ChromaDB 저장 완료: {i}/{len(uploaded_frames)} - {frame_id}", flush=True)

        Path(frame["frame_path"]).unlink()
        print(f"로컬 프레임 삭제: {frame['frame_path']}", flush=True)

        indexed_frames.append({
            **frame,
            "frame_id": frame_id,
            "object_labels": object_labels,
        })

    print(f"프레임 {len(indexed_frames)}개 ChromaDB 저장 및 로컬 삭제 완료")

    # 5. SQLite에 메타데이터 저장
    init_db()

    for frame in indexed_frames:
        object_labels_json = json.dumps(
            frame.get("object_labels", []),
            ensure_ascii=False,
        )

        insert_scene(
            video_id=video_id,
            start_time=frame["timestamp"],
            end_time=frame["timestamp"] + 1.0,
            object_labels=object_labels_json,
            s3_key=frame["s3_key"],
        )

    print(f"메타데이터 {len(indexed_frames)}개 저장 완료!")

    # 6. 임시 파일 삭제
    os.remove(local_path)
    print(f"임시 파일 삭제: {local_path}")

    return {
        "chunk_path": chunk_path,
        "frame_count": len(indexed_frames),
        "frames": indexed_frames,
    }