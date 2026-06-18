from db.connection import connect, using_mysql


def init_db():
    """DB 및 테이블 초기화"""
    conn = connect()
    cursor = conn.cursor()

    if using_mysql():
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenes (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                video_id VARCHAR(255) NOT NULL,
                start_time DOUBLE NOT NULL,
                end_time DOUBLE NOT NULL,
                object_labels TEXT,
                s3_key TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_scenes_video_time (video_id, start_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavior_events (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                video_id VARCHAR(255) NOT NULL,
                start_time DOUBLE NOT NULL,
                end_time DOUBLE NOT NULL,
                subject VARCHAR(255),
                action TEXT,
                target_object VARCHAR(255),
                summary TEXT,
                duration DOUBLE,
                repeat_count INTEGER DEFAULT 0,
                confidence DOUBLE,
                interestingness DOUBLE,
                evidence_json JSON,
                source_frames_json JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_behavior_events_video_time (video_id, start_time),
                INDEX idx_behavior_events_video_score (video_id, interestingness, confidence)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                user_id VARCHAR(255) NOT NULL,
                query_raw TEXT NOT NULL,
                query_norm VARCHAR(255) NOT NULL,
                video_id VARCHAR(255),
                top_k INTEGER,
                result_count INTEGER,
                latency_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_search_logs_user_created_at (user_id, created_at),
                INDEX idx_search_logs_query_norm (query_norm)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_frequent_queries (
                user_id VARCHAR(255) NOT NULL,
                query_norm VARCHAR(255) NOT NULL,
                query_display TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                last_searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, query_norm),
                INDEX idx_user_frequent_queries_user_count (user_id, count)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
    else:
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
