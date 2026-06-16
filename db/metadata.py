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


def _to_json_text(value):
    """
    list/dict 값을 DB에 저장하기 위한 JSON 문자열로 변환한다.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False)


def _from_json_text(value, default=None):
    """
    DB에 저장된 JSON 문자열을 다시 Python 객체로 변환한다.
    """
    if default is None:
        default = []

    if not value:
        return default

    if isinstance(value, (list, dict)):
        return value

    try:
        return json.loads(value)
    except Exception:
        return default


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
        CREATE TABLE IF NOT EXISTS behavior_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            subject TEXT,
            action TEXT,
            target_object TEXT,
            summary TEXT,
            duration REAL,
            repeat_count INTEGER DEFAULT 0,
            confidence REAL,
            interestingness REAL,
            evidence_json TEXT,
            source_frames_json TEXT,
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

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_behavior_events_video_time
        ON behavior_events(video_id, start_time)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_behavior_events_video_score
        ON behavior_events(video_id, interestingness DESC, confidence DESC)
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


def delete_behavior_events(video_id):
    """
    특정 영상의 기존 행동 이벤트를 삭제한다.

    같은 영상을 다시 처리할 때 behavior_events가 중복 저장되는 것을 막기 위해 사용한다.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        DELETE FROM behavior_events
        WHERE video_id = ?
    ''', (video_id,))

    conn.commit()
    conn.close()


def insert_behavior_event(
    video_id,
    start_time,
    end_time,
    subject=None,
    action=None,
    target_object=None,
    summary=None,
    duration=None,
    repeat_count=0,
    confidence=None,
    interestingness=None,
    evidence=None,
    source_frames=None,
):
    """
    행동 이벤트 1개 저장

    Args:
        video_id: 영상 ID
        start_time: 행동 구간 시작 시간
        end_time: 행동 구간 종료 시간
        subject: 행동 주체 예: dog
        action: 행동 예: 그릇을 앞발로 반복해서 건드림
        target_object: 행동 대상 예: bowl
        summary: 행동 설명 요약
        duration: 행동 지속 시간
        repeat_count: 반복 횟수
        confidence: 분석 신뢰도
        interestingness: 추천질문으로 만들 가치 점수
        evidence: 근거 설명 리스트
        source_frames: 행동 분석에 사용된 프레임 정보 리스트
    """
    if duration is None:
        duration = round(float(end_time) - float(start_time), 2)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO behavior_events (
            video_id,
            start_time,
            end_time,
            subject,
            action,
            target_object,
            summary,
            duration,
            repeat_count,
            confidence,
            interestingness,
            evidence_json,
            source_frames_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        video_id,
        float(start_time),
        float(end_time),
        subject,
        action,
        target_object,
        summary,
        float(duration) if duration is not None else None,
        int(repeat_count or 0),
        float(confidence) if confidence is not None else None,
        float(interestingness) if interestingness is not None else None,
        _to_json_text(evidence or []),
        _to_json_text(source_frames or []),
    ))

    event_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return event_id


def insert_behavior_events(events):
    """
    행동 이벤트 여러 개 저장

    events 예시:
    [
        {
            "video_id": "dog_escape",
            "start_time": 18.0,
            "end_time": 42.0,
            "subject": "dog",
            "action": "그릇을 앞발로 반복해서 건드림",
            "target_object": "bowl",
            "summary": "...",
            "duration": 24.0,
            "repeat_count": 5,
            "confidence": 0.82,
            "interestingness": 0.91,
            "evidence": [...],
            "source_frames": [...]
        }
    ]
    """
    event_ids = []

    for event in events:
        event_id = insert_behavior_event(
            video_id=event.get("video_id"),
            start_time=event.get("start_time"),
            end_time=event.get("end_time"),
            subject=event.get("subject"),
            action=event.get("action"),
            target_object=event.get("target_object") or event.get("target"),
            summary=event.get("summary"),
            duration=event.get("duration"),
            repeat_count=event.get("repeat_count", 0),
            confidence=event.get("confidence"),
            interestingness=event.get("interestingness"),
            evidence=event.get("evidence", []),
            source_frames=event.get("source_frames", []),
        )

        event_ids.append(event_id)

    return event_ids


def _row_to_behavior_event(row):
    """
    sqlite Row를 행동 이벤트 dict로 변환한다.
    """
    event = dict(row)

    event["evidence"] = _from_json_text(event.pop("evidence_json", None), default=[])
    event["source_frames"] = _from_json_text(event.pop("source_frames_json", None), default=[])

    return event


def get_behavior_events(video_id=None, limit=5):
    """
    특정 영상 또는 전체 영상의 행동 이벤트 조회

    interestingness와 confidence가 높은 순서로 반환한다.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if video_id:
        cursor.execute('''
            SELECT
                id,
                video_id,
                start_time,
                end_time,
                subject,
                action,
                target_object,
                summary,
                duration,
                repeat_count,
                confidence,
                interestingness,
                evidence_json,
                source_frames_json,
                created_at
            FROM behavior_events
            WHERE video_id = ?
            ORDER BY
                COALESCE(interestingness, 0) DESC,
                COALESCE(confidence, 0) DESC,
                start_time ASC
            LIMIT ?
        ''', (video_id, limit))
    else:
        cursor.execute('''
            SELECT
                id,
                video_id,
                start_time,
                end_time,
                subject,
                action,
                target_object,
                summary,
                duration,
                repeat_count,
                confidence,
                interestingness,
                evidence_json,
                source_frames_json,
                created_at
            FROM behavior_events
            ORDER BY
                COALESCE(interestingness, 0) DESC,
                COALESCE(confidence, 0) DESC,
                start_time ASC
            LIMIT ?
        ''', (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [_row_to_behavior_event(row) for row in rows]


def get_top_behavior_events(video_id=None, limit=5):
    """
    추천질문 생성에 사용할 주요 행동 이벤트 조회
    """
    return get_behavior_events(video_id=video_id, limit=limit)


def get_behavior_events_overlapping(video_id, start_time, end_time, limit=5):
    """
    특정 시간 구간과 겹치는 행동 이벤트 조회

    나중에 RAG 답변에서 검색된 프레임 timestamp 주변의 행동 이벤트를 함께 보여줄 때 사용한다.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT
            id,
            video_id,
            start_time,
            end_time,
            subject,
            action,
            target_object,
            summary,
            duration,
            repeat_count,
            confidence,
            interestingness,
            evidence_json,
            source_frames_json,
            created_at
        FROM behavior_events
        WHERE video_id = ?
          AND start_time <= ?
          AND end_time >= ?
        ORDER BY
            COALESCE(interestingness, 0) DESC,
            COALESCE(confidence, 0) DESC,
            start_time ASC
        LIMIT ?
    ''', (
        video_id,
        float(end_time),
        float(start_time),
        limit,
    ))

    rows = cursor.fetchall()
    conn.close()

    return [_row_to_behavior_event(row) for row in rows]


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

    except Exception:
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