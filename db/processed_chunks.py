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


def commit_chunk_results(chunk_path, video_id, write_results):
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
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
