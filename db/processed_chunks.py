import pymysql

from db.connection import connect


DUPLICATE_ENTRY = 1062


def is_chunk_processed(chunk_path):
    conn = connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM processed_chunks WHERE chunk_path = ?",
            (chunk_path,),
        )
        return bool(cursor.fetchall())
    finally:
        conn.close()


def is_chunk_ready(chunk_path):
    """True only after database rows and vector frames were verified together."""
    conn = connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM chunk_processing_status WHERE chunk_path = ? AND state = 'ready'",
            (chunk_path,),
        )
        return bool(cursor.fetchall())
    finally:
        conn.close()


def get_chunk_processing_statuses(video_id=None):
    """Return machine-readable ingestion integrity for the recordings API."""
    conn = connect(dict_rows=True)
    try:
        cursor = conn.cursor()
        if video_id:
            cursor.execute(
                "SELECT chunk_path, video_id, state, scene_count, vector_frame_count, "
                "behavior_event_count, error_message, verified_at FROM chunk_processing_status "
                "WHERE video_id = ? ORDER BY verified_at DESC",
                (video_id,),
            )
        else:
            cursor.execute(
                "SELECT chunk_path, video_id, state, scene_count, vector_frame_count, "
                "behavior_event_count, error_message, verified_at FROM chunk_processing_status "
                "ORDER BY verified_at DESC"
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def record_chunk_integrity(chunk_path, video_id, integrity):
    """Attach a ready status to a legacy completion marker after read-back.

    Older workers created `processed_chunks` without an integrity row. The
    caller has just re-extracted and read every vector back, so the old MySQL
    transaction already guarantees scene/event rows while this fills the
    missing observability record. Existing ready records are left untouched.
    """
    conn = connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO chunk_processing_status (
                chunk_path, video_id, state, scene_count, vector_frame_count, behavior_event_count
            ) VALUES (?, ?, 'ready', ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                state = 'ready',
                scene_count = VALUES(scene_count),
                vector_frame_count = VALUES(vector_frame_count),
                behavior_event_count = VALUES(behavior_event_count),
                error_message = NULL
            ''',
            (
                chunk_path,
                video_id,
                int(integrity["scene_count"]),
                int(integrity["vector_frame_count"]),
                int(integrity["behavior_event_count"]),
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def commit_chunk_results(chunk_path, video_id, write_results, integrity):
    """
    청크 결과(scenes, behavior_events)와 완료 표시를 한 트랜잭션으로 저장한다.

    완료 표시를 가장 먼저 insert하므로, 같은 청크를 동시에 처리한 다른 worker가
    먼저 커밋했다면 중복 키 오류로 롤백된다. 이 경우 False를 반환한다.
    중간에 실패하면 전부 롤백되어 재실행해도 부분 결과가 남지 않는다.
    """
    conn = connect()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO processed_chunks (chunk_path, video_id) VALUES (?, ?)",
                (chunk_path, video_id),
            )
        except pymysql.err.IntegrityError as exc:
            if exc.args and exc.args[0] == DUPLICATE_ENTRY:
                conn.rollback()
                return False
            raise

        write_results(conn)
        cursor.execute(
            '''
            INSERT INTO chunk_processing_status (
                chunk_path, video_id, state, scene_count, vector_frame_count, behavior_event_count
            ) VALUES (?, ?, 'ready', ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                state = 'ready',
                scene_count = VALUES(scene_count),
                vector_frame_count = VALUES(vector_frame_count),
                behavior_event_count = VALUES(behavior_event_count),
                error_message = NULL
            ''',
            (
                chunk_path,
                video_id,
                int(integrity["scene_count"]),
                int(integrity["vector_frame_count"]),
                int(integrity["behavior_event_count"]),
            ),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# queued -> processing -> ready | failed. ready는 commit_chunk_results/record_chunk_integrity만 쓴다.
PROGRESS_STATES = ("queued", "processing", "failed")


def mark_chunk_state(chunk_path, video_id, state, error=None):
    """청크의 처리 진행 상태를 남긴다. 상태 기록 실패가 영상 처리를 막으면 안 되므로 예외를 내지 않는다.

    이미 ready인 청크는 덮어쓰지 않는다(재전달된 중복 작업이 완료 기록을 되돌리면 안 된다).
    """
    if state not in PROGRESS_STATES:
        raise ValueError(f"알 수 없는 상태: {state}")
    message = str(error)[:1000] if error else None
    try:
        conn = connect()
    except Exception as exc:
        print(f"처리 상태 기록 실패(연결): {chunk_path} - {exc}", flush=True)
        return False
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO chunk_processing_status (chunk_path, video_id, state, error_message)
            VALUES (?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                error_message = IF(state = 'ready', error_message, VALUES(error_message)),
                state = IF(state = 'ready', state, VALUES(state))
            ''',
            (chunk_path, video_id, state, message),
        )
        conn.commit()
        return True
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"처리 상태 기록 실패: {chunk_path} - {exc}", flush=True)
        return False
    finally:
        conn.close()
