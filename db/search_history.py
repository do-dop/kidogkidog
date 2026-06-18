import re

from db.connection import connect, using_mysql


def normalize_query(query: str) -> str:
    """
    검색어 정규화 함수
    """
    if not query:
        return ""

    query = query.strip().lower()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"^[^\w가-힣]+|[^\w가-힣]+$", "", query)

    return query[:255]


def insert_search_log(
    user_id,
    query_raw,
    video_id=None,
    top_k=None,
    result_count=None,
    latency_ms=None
):
    """
    검색 요청 이벤트 로그 저장
    """
    query_norm = normalize_query(query_raw)

    conn = connect()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO search_logs (
            user_id,
            query_raw,
            query_norm,
            video_id,
            top_k,
            result_count,
            latency_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        query_raw,
        query_norm,
        video_id,
        top_k,
        result_count,
        latency_ms
    ))

    conn.commit()
    conn.close()


def upsert_user_frequent_query(user_id, query_raw):
    """
    사용자별 빈출 검색어 저장/업데이트
    """
    query_norm = normalize_query(query_raw)

    if not query_norm:
        return

    query_display = query_raw.strip()

    conn = connect()
    cursor = conn.cursor()

    if using_mysql():
        cursor.execute('''
            INSERT INTO user_frequent_queries (
                user_id,
                query_norm,
                query_display,
                count,
                last_searched_at,
                updated_at
            )
            VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE
                count = count + 1,
                query_display = VALUES(query_display),
                last_searched_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            user_id,
            query_norm,
            query_display
        ))
    else:
        cursor.execute('''
            INSERT INTO user_frequent_queries (
                user_id,
                query_norm,
                query_display,
                count,
                last_searched_at,
                updated_at
            )
            VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, query_norm)
            DO UPDATE SET
                count = count + 1,
                query_display = excluded.query_display,
                last_searched_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            user_id,
            query_norm,
            query_display
        ))

    conn.commit()
    conn.close()


def get_user_top_queries(user_id, limit=5):
    """
    사용자별 자주 찾는 검색어 조회
    """
    conn = connect()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT query_display, count, last_searched_at
        FROM user_frequent_queries
        WHERE user_id = ?
        ORDER BY count DESC, last_searched_at DESC
        LIMIT ?
    ''', (user_id, limit))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "query": row[0],
            "count": row[1],
            "last_searched_at": row[2]
        }
        for row in rows
    ]


def delete_user_frequent_query(user_id, query_raw):
    """
    사용자별 자주 찾는 검색어에서 특정 검색어를 삭제한다.
    """
    query_norm = normalize_query(query_raw)

    if not query_norm:
        return 0

    conn = connect()
    cursor = conn.cursor()

    cursor.execute('''
        DELETE FROM user_frequent_queries
        WHERE user_id = ?
          AND query_norm = ?
    ''', (user_id, query_norm))

    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()

    return deleted_count


def get_user_recent_queries(user_id, limit=5):
    """
    사용자별 최근 검색어 조회
    """
    conn = connect()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT query_raw, created_at
        FROM search_logs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (user_id, limit))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "query": row[0],
            "created_at": row[1]
        }
        for row in rows
    ]


def get_global_top_queries(limit=5):
    """
    전체 사용자 기준 자주 검색된 검색어 조회
    """
    conn = connect()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT
            query_raw,
            query_norm,
            COUNT(*) AS query_count,
            MAX(created_at) AS last_searched_at
        FROM search_logs
        WHERE query_norm != ''
        GROUP BY query_norm
        ORDER BY query_count DESC, last_searched_at DESC
        LIMIT ?
    ''', (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "query": row[0],
            "query_norm": row[1],
            "count": row[2],
            "last_searched_at": row[3],
        }
        for row in rows
    ]


def get_video_top_queries(video_id, limit=5):
    """
    특정 영상 기준 자주 검색된 검색어 조회
    """
    conn = connect()
    cursor = conn.cursor()

    if video_id:
        cursor.execute('''
            SELECT
                query_raw,
                query_norm,
                COUNT(*) AS query_count,
                MAX(created_at) AS last_searched_at
            FROM search_logs
            WHERE video_id = ?
              AND query_norm != ''
            GROUP BY query_norm
            ORDER BY query_count DESC, last_searched_at DESC
            LIMIT ?
        ''', (video_id, limit))
    else:
        cursor.execute('''
            SELECT
                query_raw,
                query_norm,
                COUNT(*) AS query_count,
                MAX(created_at) AS last_searched_at
            FROM search_logs
            WHERE query_norm != ''
            GROUP BY query_norm
            ORDER BY query_count DESC, last_searched_at DESC
            LIMIT ?
        ''', (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "query": row[0],
            "query_norm": row[1],
            "count": row[2],
            "last_searched_at": row[3],
        }
        for row in rows
    ]
