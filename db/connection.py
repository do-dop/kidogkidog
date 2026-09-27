import os

import pymysql


MYSQL_HOST = os.getenv("MYSQL_HOST") or os.getenv("RDS_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT") or os.getenv("RDS_PORT") or "3306")
MYSQL_USER = os.getenv("MYSQL_USER") or os.getenv("RDS_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD") or os.getenv("RDS_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE") or os.getenv("RDS_DATABASE") or os.getenv("MYSQL_DB")
MYSQL_SSL_CA = os.getenv("MYSQL_SSL_CA") or os.getenv("RDS_SSL_CA")
MYSQL_UNIX_SOCKET = os.getenv("MYSQL_UNIX_SOCKET")


class _Cursor:
    """쿼리의 `?` placeholder를 pymysql 형식(`%s`)으로 바꿔 실행한다."""

    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, sql, params=None):
        # params가 없으면 None으로 넘겨 LIKE '%...%' 같은 리터럴 %를 포맷 문자로 해석하지 않게 한다.
        return self._cursor.execute(sql.replace("?", "%s"), params or None)

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @property
    def rowcount(self):
        return self._cursor.rowcount


class _Connection:
    def __init__(self, raw_connection, dict_rows):
        self._raw_connection = raw_connection
        self._cursor_class = pymysql.cursors.DictCursor if dict_rows else pymysql.cursors.Cursor

    def cursor(self):
        return _Cursor(self._raw_connection.cursor(self._cursor_class))

    def commit(self):
        return self._raw_connection.commit()

    def close(self):
        return self._raw_connection.close()


def connect(dict_rows=False):
    """
    MySQL 연결을 연다.
    dict_rows=True면 조회 결과를 컬럼명 기반 dict로 반환한다.
    """
    if not (MYSQL_HOST or MYSQL_UNIX_SOCKET) or not MYSQL_USER or not MYSQL_DATABASE:
        raise RuntimeError(
            "MYSQL_HOST(또는 MYSQL_UNIX_SOCKET), MYSQL_USER, MYSQL_DATABASE 환경변수가 필요합니다."
        )

    # GCP Cloud SQL 서버 인증서는 IP가 아닌 인스턴스 UID로 발급되어 hostname 검증이
    # 항상 실패하므로 check_hostname을 꺼둔다. CA 검증 자체는 그대로 수행된다.
    ssl_config = {"ca": MYSQL_SSL_CA, "check_hostname": False} if MYSQL_SSL_CA else None

    connection_options = {
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD or "",
        "database": MYSQL_DATABASE,
        "charset": "utf8mb4",
        "autocommit": False,
        "ssl": ssl_config,
    }
    if MYSQL_UNIX_SOCKET:
        connection_options["unix_socket"] = MYSQL_UNIX_SOCKET
    else:
        connection_options["host"] = MYSQL_HOST
        connection_options["port"] = MYSQL_PORT

    return _Connection(pymysql.connect(**connection_options), dict_rows)
