import sqlite3
import os
import re
import json
from collections import Counter

DB_PATH = "db/kidogkidog.db"


def normalize_query(query: str) -> str:
    """
    검색어 정규화 함수

    예:
    - " 강아지가  밥 먹는 장면? " -> "강아지가 밥 먹는 장면"
    - "Dog Eating Food!!" -> "dog eating food"
    """
    if not query:
        return ""

    query = query.strip().lower()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"^[^\w가-힣]+|[^\w가-힣]+$", "", query)

    return query


def init_db():
    """DB 및 테이블 초기화"""
    os.makedirs("db", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scenes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            object_labels TEXT,
            s3_key TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            query_raw TEXT NOT NULL,
            query_norm TEXT NOT NULL,
            video_id TEXT,
            top_k INTEGER,
            result_count INTEGER,
            latency_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_frequent_queries (
            user_id TEXT NOT NULL,
            query_norm TEXT NOT NULL,
            query_display TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 1,
            last_searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, query_norm)
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_search_logs_user_created_at
        ON search_logs(user_id, created_at DESC)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_search_logs_query_norm
        ON search_logs(query_norm)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_frequent_queries_user_count
        ON user_frequent_queries(user_id, count DESC)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_scenes_video_time
        ON scenes(video_id, start_time)
    ''')

    conn.commit()
    conn.close()

    print("DB 초기화 완료!")


def insert_scene(video_id, start_time, end_time, object_labels=None, s3_key=None):
    """프레임 메타데이터 저장"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO scenes (video_id, start_time, end_time, object_labels, s3_key)
        VALUES (?, ?, ?, ?, ?)
    ''', (video_id, start_time, end_time, object_labels, s3_key))

    conn.commit()
    conn.close()


def get_scenes(video_id):
    """특정 영상의 모든 scene 조회"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM scenes WHERE video_id = ?', (video_id,))
    rows = cursor.fetchall()

    conn.close()
    return rows


def get_scene_records(video_id=None):
    """
    scenes 테이블을 dict 형태로 조회한다.
    scene_event_extractor.py에서 사용한다.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if video_id:
        cursor.execute('''
            SELECT id, video_id, start_time, end_time, object_labels, s3_key, created_at
            FROM scenes
            WHERE video_id = ?
            ORDER BY start_time ASC, id ASC
        ''', (video_id,))
    else:
        cursor.execute('''
            SELECT id, video_id, start_time, end_time, object_labels, s3_key, created_at
            FROM scenes
            ORDER BY video_id ASC, start_time ASC, id ASC
        ''')

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


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

    검색 성공/실패와 관계없이 사용자가 검색을 시도했다는 사실을 저장한다.
    """
    query_norm = normalize_query(query_raw)

    conn = sqlite3.connect(DB_PATH)
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

    같은 사용자가 같은 정규화 검색어를 다시 검색하면 count를 1 증가시킨다.
    """
    query_norm = normalize_query(query_raw)

    if not query_norm:
        return

    query_display = query_raw.strip()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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

    count가 높은 순서대로 반환한다.
    """
    conn = sqlite3.connect(DB_PATH)
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


def get_user_recent_queries(user_id, limit=5):
    """
    사용자별 최근 검색어 조회
    """
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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


def _parse_object_labels(raw_labels):
    """
    scenes.object_labels에 저장된 JSON 문자열을 list[str]로 변환한다.

    예:
    '["dog", "bed"]' -> ["dog", "bed"]
    'dog,bed' -> ["dog", "bed"]
    None -> []
    """
    if not raw_labels:
        return []

    if isinstance(raw_labels, list):
        return raw_labels

    try:
        parsed = json.loads(raw_labels)

        if isinstance(parsed, list):
            return [str(label) for label in parsed]

        if isinstance(parsed, str):
            return [parsed]

    except json.JSONDecodeError:
        return [
            label.strip()
            for label in str(raw_labels).split(",")
            if label.strip()
        ]

    return []


def get_video_object_labels(video_id=None):
    """
    특정 영상 또는 전체 영상의 scene object_labels를 펼쳐서 반환한다.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if video_id:
        cursor.execute('''
            SELECT object_labels
            FROM scenes
            WHERE video_id = ?
              AND object_labels IS NOT NULL
        ''', (video_id,))
    else:
        cursor.execute('''
            SELECT object_labels
            FROM scenes
            WHERE object_labels IS NOT NULL
        ''')

    rows = cursor.fetchall()
    conn.close()

    labels = []

    for row in rows:
        labels.extend(_parse_object_labels(row[0]))

    return labels


def get_video_top_object_labels(video_id=None, limit=10):
    """
    특정 영상 또는 전체 영상에서 많이 감지된 객체 라벨 조회
    """
    labels = get_video_object_labels(video_id=video_id)
    counter = Counter(labels)

    return [
        {
            "label": label,
            "count": count,
        }
        for label, count in counter.most_common(limit)
    ]