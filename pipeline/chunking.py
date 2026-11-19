"""청크 길이 설정의 단일 출처.

청크 파일명에는 번호만 들어 있고 길이는 없다. 그래서 `청크번호 * CHUNK_SECONDS`로
영상 안 위치를 계산하는 모든 곳(분할, API, 검색 답변)이 같은 값을 써야 한다.
이 값을 바꾸면 이미 색인된 데이터의 시각 해석이 달라지므로, 새 값으로 처리한 영상은
새 video_id로 올리거나 다시 색인한다.

환경변수 CHUNK_SECONDS로 조절하며 기본은 60초다. 잘못된 값은 기본값으로 되돌린다.
"""

import os

DEFAULT_CHUNK_SECONDS = 60


def _read_chunk_seconds():
    try:
        value = int(os.getenv("CHUNK_SECONDS", str(DEFAULT_CHUNK_SECONDS)))
    except ValueError:
        return DEFAULT_CHUNK_SECONDS
    return value if value > 0 else DEFAULT_CHUNK_SECONDS


CHUNK_SECONDS = _read_chunk_seconds()
