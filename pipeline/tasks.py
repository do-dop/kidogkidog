from celery import Celery
from celery.signals import worker_ready
from pipeline.video.frame_extractor import extract_frames
from pipeline.storage.gcs_uploader import download_video, upload_frame
from pipeline.storage.vector_store import index_frame
from pipeline.models.yolo_detector import detect_objects
from pipeline.events.behavior_event_extractor import extract_behavior_events
from db.behavior_events import insert_behavior_events
from db.scenes import insert_scene
from db.schema import init_db
from db.processed_chunks import commit_chunk_results, is_chunk_processed
import os
import json
import time
from datetime import timedelta
from pathlib import Path
from pipeline.storage.gcs_uploader import get_object_metadata
from pipeline.observability.metrics import CHUNKS, FRAMES, STAGE_SECONDS, measure_stage, start_worker_metrics

FRAME_ROOT = Path("pipeline/frames")

celery_app = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@localhost:5672/')
)
celery_app.conf.update(
    # 작업이 끝난 뒤에 ack한다. worker가 중간에 죽으면 메시지가 큐로 돌아가 다시 처리된다.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # 긴 작업을 미리 받아 두지 않게 해서 죽은 worker가 다른 메시지까지 쥐고 있지 않게 한다.
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    # solo 풀은 작업 중 heartbeat를 보내지 못해, 수 분짜리 청크를 처리하는 동안 RabbitMQ가
    # 연결을 끊고 ack가 실패한다(실습 VM에서 확인). 끊김 감지는 TCP에 맡긴다.
    broker_heartbeat=0,
)
worker_ready.connect(start_worker_metrics, weak=False)


def upload_frame_to_object_storage(local_frame_path, video_id):
    """프레임을 GCS에 업로드"""
    return upload_frame(local_frame_path, video_id)


def index_frame_to_chromadb(local_frame_path, video_id, s3_key=None, object_labels=None, recorded_at=None, timestamp=None):
    """프레임을 CLIP 임베딩 후 ChromaDB에 저장"""
    return index_frame(
        local_frame_path,
        frame_root=FRAME_ROOT,
        video_id=video_id,
        s3_key=s3_key,
        object_labels=object_labels,
        recorded_at=recorded_at,
        timestamp=timestamp,
    )


def get_video_id_from_object_key(object_key):
    key = Path(object_key)
    parts = key.parts

    if len(parts) >= 2 and parts[0] == "chunks":
        return parts[1]

    if len(parts) >= 2 and parts[0] == "videos":
        return key.stem

    return key.stem


def delete_local_frames(frames):
    """
    행동 분석까지 끝난 뒤 로컬 프레임 파일을 삭제한다.

    기존에는 YOLO/ChromaDB 저장 직후 프레임을 삭제했지만,
    이제는 behavior_event_extractor가 대표 프레임 이미지를 사용해야 하므로
    행동 이벤트 분석 이후에 삭제한다.
    """
    deleted_count = 0
    seen_paths = set()

    for frame in frames:
        frame_path = frame.get("frame_path")

        if not frame_path:
            continue

        if frame_path in seen_paths:
            continue

        seen_paths.add(frame_path)

        path = Path(frame_path)

        if path.exists():
            path.unlink()
            deleted_count += 1
            print(f"로컬 프레임 삭제: {frame_path}", flush=True)

    return deleted_count


# 재실행은 processed_chunks 완료 표시와 결정적인 GCS/ChromaDB 키 덕분에 안전하다.
# 존재하지 않는 객체는 다시 시도해도 같은 결과이므로 재시도하지 않는다.
@celery_app.task(
    name='pipeline.tasks.process_chunk',
    acks_late=True,
    reject_on_worker_lost=True,
    autoretry_for=(Exception,),
    dont_autoretry_for=(FileNotFoundError,),
    max_retries=int(os.getenv("CHUNK_MAX_RETRIES", "3")),
    retry_backoff=10,
    retry_backoff_max=300,
    retry_jitter=True,
)
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
    → 행동 이벤트 분석
    → MySQL scenes·behavior_events·완료 표시를 한 트랜잭션으로 저장
    → 로컬 프레임 삭제
    """
    print(f"[Task 시작] {chunk_path}")
    task_started = time.monotonic()

    video_id = get_video_id_from_object_key(chunk_path)

    # 재전달·중복 요청으로 이미 끝난 청크는 다시 처리하지 않는다.
    init_db()
    if is_chunk_processed(chunk_path):
        print(f"[Task 스킵] 이미 처리된 청크: {chunk_path}", flush=True)
        CHUNKS.labels(outcome="duplicate").inc()
        STAGE_SECONDS.labels(stage="chunk_total").observe(time.monotonic() - task_started)
        return {"chunk_path": chunk_path, "video_id": video_id, "status": "duplicate"}

    frame_output_dir = FRAME_ROOT / video_id

    local_path = f"/tmp/{os.path.basename(chunk_path)}"
    indexed_frames = []
    behavior_events = []

    try:
        chunk_metadata = get_object_metadata(chunk_path)
        chunk_recorded_at = chunk_metadata.get("LastModified")

        # 1. GCS에서 영상/청크 다운로드
        with measure_stage("video_download"):
            download_video(chunk_path, local_path)
        print(f"다운로드 완료: {local_path}")

        # 2. 프레임 추출 (motion 감지 포함)
        with measure_stage("frame_extraction"):
            frames = extract_frames(
                local_path,
                output_dir=str(frame_output_dir),
                frame_prefix=Path(local_path).stem,
            )
        print(f"프레임 {len(frames)}개 추출 완료")

        # 3. 프레임을 먼저 GCS에 모두 업로드
        uploaded_frames = []

        for frame in frames:
            with measure_stage("frame_upload"):
                frame_s3_key = upload_frame_to_object_storage(
                    frame["frame_path"],
                    video_id,
                )

            uploaded_frames.append({
                **frame,
                "video_id": video_id,
                "s3_key": frame_s3_key,
            })

        print(f"프레임 {len(uploaded_frames)}개 GCS 업로드 완료")

        # 4. 업로드된 프레임에 YOLO 객체 감지 + CLIP/ChromaDB 저장
        #
        # 중요:
        # 기존에는 여기서 로컬 프레임을 바로 삭제했지만,
        # 이제는 행동 이벤트 분석에서 대표 프레임 이미지를 사용해야 하므로
        # 삭제하지 않고 indexed_frames에 모아둔다.
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
                timestamp=frame["timestamp"],
                recorded_at=(
                    chunk_recorded_at + timedelta(seconds=float(frame["timestamp"]))
                    if chunk_recorded_at else None
                ),
            )

            print(f"ChromaDB 저장 완료: {i}/{len(uploaded_frames)} - {frame_id}", flush=True)

            indexed_frames.append({
                **frame,
                "frame_id": frame_id,
                "object_labels": object_labels,
            })

        print(f"프레임 {len(indexed_frames)}개 ChromaDB 저장 완료")

        # 5. 행동 이벤트 분석
        #
        # indexed_frames에는 frame_path, timestamp, motion_start, motion_end,
        # segment_id, video_id, s3_key, object_labels, frame_id가 들어있다.
        # 따라서 같은 motion segment의 프레임들을 묶어서 행동 이벤트를 만들 수 있다.
        if indexed_frames:
            print("행동 이벤트 분석 시작...", flush=True)

            with measure_stage("behavior_analysis"):
                behavior_events = extract_behavior_events(
                    indexed_frames=indexed_frames,
                    video_id=video_id,
                    limit=5,
                )
        else:
            print("행동 이벤트 분석 생략: indexed_frames가 없습니다.", flush=True)

        # 6. MySQL 저장 (scenes + behavior_events + 완료 표시를 한 트랜잭션으로)
        #
        # GCS 프레임 키와 ChromaDB 프레임 ID는 청크마다 결정적이라 재실행해도 덮어쓰거나
        # 건너뛰지만, MySQL은 insert라 중복될 수 있다. 완료 표시와 함께 커밋해
        # 재실행 시 전부 저장되었거나 전혀 저장되지 않은 상태만 남게 한다.
        def write_results(conn):
            for frame in indexed_frames:
                insert_scene(
                    video_id=video_id,
                    start_time=frame["timestamp"],
                    end_time=frame["timestamp"] + 1.0,
                    object_labels=json.dumps(
                        frame.get("object_labels", []),
                        ensure_ascii=False,
                    ),
                    s3_key=frame["s3_key"],
                    conn=conn,
                )

            if behavior_events:
                insert_behavior_events(behavior_events, conn=conn)

        with measure_stage("scene_write"):
            committed = commit_chunk_results(chunk_path, video_id, write_results)

        if committed:
            print(
                f"scenes {len(indexed_frames)}개, 행동 이벤트 {len(behavior_events)}개 저장 완료!",
                flush=True,
            )
        else:
            print(f"다른 worker가 먼저 저장 완료: {chunk_path}", flush=True)

        # 7. 행동 분석까지 끝난 뒤 로컬 프레임 삭제
        deleted_frame_count = delete_local_frames(indexed_frames)
        print(f"로컬 프레임 {deleted_frame_count}개 삭제 완료")

        # 8. 임시 영상 파일 삭제
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"임시 파일 삭제: {local_path}")

        if committed:
            FRAMES.inc(len(indexed_frames))
        CHUNKS.labels(outcome="success" if committed else "duplicate").inc()
        return {
            "status": "processed" if committed else "duplicate",
            "chunk_path": chunk_path,
            "video_id": video_id,
            "frame_count": len(indexed_frames),
            "behavior_event_count": len(behavior_events),
            "frames": indexed_frames,
            "behavior_events": behavior_events,
        }

    except Exception as exc:
        CHUNKS.labels(outcome="failure").inc()
        print(f"[Task 실패] {chunk_path} - {exc}", flush=True)

        # 실패하더라도 가능한 로컬 파일은 정리
        if indexed_frames:
            delete_local_frames(indexed_frames)

        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"임시 파일 삭제: {local_path}")

        raise
    finally:
        STAGE_SECONDS.labels(stage="chunk_total").observe(time.monotonic() - task_started)
