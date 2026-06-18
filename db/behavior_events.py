import sqlite3

from db.connection import connect
from db.json_utils import from_json_text, to_json_text


def delete_behavior_events(video_id):
    """
    특정 영상의 기존 행동 이벤트를 삭제한다.
    """
    conn = connect()
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
    """
    if duration is None:
        duration = round(float(end_time) - float(start_time), 2)

    conn = connect()
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
        to_json_text(evidence or []),
        to_json_text(source_frames or []),
    ))

    event_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return event_id


def insert_behavior_events(events):
    """
    행동 이벤트 여러 개 저장
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
    event = dict(row)

    event["evidence"] = from_json_text(event.pop("evidence_json", None), default=[])
    event["source_frames"] = from_json_text(event.pop("source_frames_json", None), default=[])

    return event


def get_behavior_events(video_id=None, limit=5):
    """
    특정 영상 또는 전체 영상의 행동 이벤트 조회
    """
    conn = connect()
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


def get_behavior_event_by_id(event_id):
    """
    추천 질문과 연결된 단일 행동 이벤트 조회
    """
    conn = connect()
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
        WHERE id = ?
        LIMIT 1
    ''', (event_id,))

    row = cursor.fetchone()
    conn.close()

    return _row_to_behavior_event(row) if row else None


def get_behavior_events_overlapping(video_id, start_time, end_time, limit=5):
    """
    특정 시간 구간과 겹치는 행동 이벤트 조회
    """
    conn = connect()
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
