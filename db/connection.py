import os
import sqlite3


DB_BACKEND = os.getenv("KIDOGKIDOG_DB_BACKEND", "").lower()
DB_PATH = os.getenv("KIDOGKIDOG_DB_PATH", "db/kidogkidog.db")
MYSQL_HOST = os.getenv("MYSQL_HOST") or os.getenv("RDS_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT") or os.getenv("RDS_PORT") or "3306")
MYSQL_USER = os.getenv("MYSQL_USER") or os.getenv("RDS_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD") or os.getenv("RDS_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE") or os.getenv("RDS_DATABASE") or os.getenv("MYSQL_DB")
MYSQL_SSL_CA = os.getenv("MYSQL_SSL_CA") or os.getenv("RDS_SSL_CA")


def using_mysql():
    return DB_BACKEND in {"mysql", "rds"} or bool(MYSQL_HOST)


def _translate_sql(sql):
    if not using_mysql():
        return sql

    return sql.replace("?", "%s")


class _Cursor:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, sql, params=None):
        return self._cursor.execute(_translate_sql(sql), params or ())

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @property
    def rowcount(self):
        return self._cursor.rowcount


class _Connection:
    def __init__(self, raw_connection, backend):
        self._raw_connection = raw_connection
        self._backend = backend
        self._dict_rows = False

    @property
    def row_factory(self):
        return None

    @row_factory.setter
    def row_factory(self, value):
        self._dict_rows = value is not None

        if self._backend == "sqlite":
            self._raw_connection.row_factory = value

    def cursor(self):
        if self._backend == "mysql":
            import pymysql

            return _Cursor(
                self._raw_connection.cursor(
                    pymysql.cursors.DictCursor if self._dict_rows else pymysql.cursors.Cursor
                )
            )

        return _Cursor(self._raw_connection.cursor())

    def commit(self):
        return self._raw_connection.commit()

    def close(self):
        return self._raw_connection.close()


def connect():
    if using_mysql():
        if not MYSQL_HOST or not MYSQL_USER or not MYSQL_DATABASE:
            raise RuntimeError(
                "MySQL/RDS 사용 시 MYSQL_HOST, MYSQL_USER, MYSQL_DATABASE 환경변수가 필요합니다."
            )

        import pymysql

        ssl_config = {"ca": MYSQL_SSL_CA} if MYSQL_SSL_CA else None

        return _Connection(
            pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD or "",
                database=MYSQL_DATABASE,
                charset="utf8mb4",
                autocommit=False,
                ssl=ssl_config,
            ),
            "mysql",
        )

    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    return _Connection(sqlite3.connect(DB_PATH), "sqlite")
