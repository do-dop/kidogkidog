import os
import sqlite3

import pymysql

from db.metadata import init_db


SQLITE_SOURCE_DB_PATH = os.getenv("SQLITE_SOURCE_DB_PATH", "db/kidogkidog.db")

TABLE_COLUMNS = {
    "scenes": [
        "id",
        "video_id",
        "start_time",
        "end_time",
        "object_labels",
        "s3_key",
        "created_at",
    ],
    "behavior_events": [
        "id",
        "video_id",
        "start_time",
        "end_time",
        "subject",
        "action",
        "target_object",
        "summary",
        "duration",
        "repeat_count",
        "confidence",
        "interestingness",
        "evidence_json",
        "source_frames_json",
        "created_at",
    ],
    "search_logs": [
        "id",
        "user_id",
        "query_raw",
        "query_norm",
        "video_id",
        "top_k",
        "result_count",
        "latency_ms",
        "created_at",
    ],
    "user_frequent_queries": [
        "user_id",
        "query_norm",
        "query_display",
        "count",
        "last_searched_at",
        "updated_at",
    ],
}


def mysql_connect():
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.environ["MYSQL_DATABASE"],
        charset="utf8mb4",
        autocommit=False,
    )


def read_sqlite_rows(table, columns):
    source = sqlite3.connect(SQLITE_SOURCE_DB_PATH)
    source.row_factory = sqlite3.Row
    cursor = source.cursor()
    cursor.execute(f"SELECT {', '.join(columns)} FROM {table}")
    rows = [dict(row) for row in cursor.fetchall()]
    source.close()
    return rows


def insert_mysql_rows(conn, table, columns, rows):
    if not rows:
        return 0

    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(columns)
    sql = f"INSERT IGNORE INTO {table} ({column_sql}) VALUES ({placeholders})"
    values = [tuple(row.get(column) for column in columns) for row in rows]

    cursor = conn.cursor()
    cursor.executemany(sql, values)
    return cursor.rowcount


def main():
    if not os.path.exists(SQLITE_SOURCE_DB_PATH):
        raise FileNotFoundError(f"SQLite DB가 없습니다: {SQLITE_SOURCE_DB_PATH}")

    init_db()
    conn = mysql_connect()

    try:
        for table, columns in TABLE_COLUMNS.items():
            rows = read_sqlite_rows(table, columns)
            inserted = insert_mysql_rows(conn, table, columns, rows)
            print(f"{table}: sqlite {len(rows)}개, mysql 신규 {inserted}개")

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
