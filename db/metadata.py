import sqlite3
import os

DB_PATH = "db/kidogkidog.db"

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


if __name__ == "__main__":
    # DB 초기화 테스트
    init_db()

    # 더미 데이터 삽입 테스트
    insert_scene("test_video", 1.0, 5.0, "dog", "chunks/test.mp4")
    insert_scene("test_video", 10.0, 15.0, "cat", "chunks/test.mp4")

    # 조회 테스트
    scenes = get_scenes("test_video")
    print(f"저장된 scene 수: {len(scenes)}")
    for scene in scenes:
        print(f"  {scene}")