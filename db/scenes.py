import json
import sqlite3
from collections import Counter

from db.connection import connect


def insert_scene(video_id, start_time, end_time, object_labels=None, s3_key=None):
    """프레임 메타데이터 저장"""
    conn = connect()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO scenes (video_id, start_time, end_time, object_labels, s3_key)
        VALUES (?, ?, ?, ?, ?)
    ''', (video_id, start_time, end_time, object_labels, s3_key))

    conn.commit()
    conn.close()


def get_scene_records(video_id=None):
    """
    scenes 테이블을 dict 형태로 조회한다.
    """
    conn = connect()
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


def _parse_object_labels(raw_labels):
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
    conn = connect()
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
