"""행동 이벤트를 시간순으로 연결하는 시퀀스 질문 경로."""

import re
import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from pipeline.chunking import CHUNK_SECONDS
from pipeline.storage.capture_time import trusted_capture_time

MAX_STEP_FRAMES = 4
# 순서 질문임을 알려 주는 낱말. 짧아서 다른 낱말 속에 섞이기 쉬운 '전/후/뒤' 계열은 아래 경계 정규식으로 따로 본다.
_SEQUENCE_WORDS = ("순서", "시간순", "차례", "흐름", "타임라인", "어떤 행동", "뭐했", "무엇을 했", "그다음", "그 다음", "먼저", "처음", "마지막")
_FRAME_KEY = re.compile(r"petcam_(\d{8})_(\d{6})_(\d{3})_frame_([0-9.]+)\.jpg$")
_ANCHOR_STOPWORDS = {"강아지", "고양이", "반려동물", "우리", "아이가", "아이", "무엇", "뭐", "어떤", "그", "행동", "장면", "한", "했", "하는"}

# '전체', '오후', '뒤섞여', '뒤뜰'처럼 다른 낱말 안에 든 '전/후/뒤'를 시간 표지로 읽지 않도록,
# 표지는 앞뒤가 한글·영숫자로 이어지지 않는 독립된 말(조사만 허용)일 때만 인정한다.
_W = "가-힣A-Za-z0-9"
_START = rf"(?<![{_W}])"
_END = rf"(?![{_W}])"
_AFTER = re.compile(
    rf"{_START}(?:그\s*)?(?:다음|이후|뒤|후)(?:에는|에|엔|으로|로|은|는)?{_END}"
    rf"|(?:{_START}|(?<=고))나서(?:는|도)?{_END}"
)
_BEFORE = re.compile(rf"(?:{_START}|(?<=기))(?:직전|이전|전)(?:에는|에|엔)?{_END}")
_WHOLE_RANGE = re.compile(r"처음부터|처음에서|시작부터|끝까지|전부|모두|전체")
_FIRST = re.compile(r"처음|첫\s*번째|첫째|(?:제일|가장|맨)\s*먼저")
_EARLY = re.compile(r"먼저")
_ORDER_WORDS = re.compile(r"순서|시간순|차례|흐름|타임라인")
_SUMMARY_WORDS = re.compile(r"(?:오늘|하루|전체).{0,15}요약|(?:행동|뭐했|무엇을 했).{0,15}(?:요약|정리)|(?:요약|정리).{0,15}행동")
_BROAD_DAY_ACTIONS = re.compile(r"(?:오늘|하루|어제).{0,25}(?:뭐했|무엇을 했|어떤 행동|행동했|행동을 했)")
_LAST = re.compile(rf"마지막|맨\s*끝|(?:제일|가장)\s*나중|끝(?:에는|에|엔){_END}")
_ASKS_ABOUT_ACTION = re.compile(r"뭐|무엇|무슨|행동|했|어땠|어떻")


def _temporal_marker(query: str) -> bool:
    if _BEFORE.search(query) or _AFTER.search(query) or _FIRST.search(query) or _EARLY.search(query):
        return True
    last = _LAST.search(query)
    # '책상 끝에 있어?'처럼 위치를 묻는 말과 구분하려고, '끝에'는 행동을 묻는 말과 함께 있을 때만 인정한다.
    return bool(last and (not last.group().startswith("끝") or _ASKS_ABOUT_ACTION.search(query)))


# 물건이 넘어지거나 쓰러지는 등 장면이 바뀐 일. 같은 일을 다르게 적은 설명끼리 맞도록 낱말 묶음으로 본다.
_SCENE_CHANGE_GROUP = ("넘어지", "넘어졌", "넘어진", "넘어져", "넘어짐", "넘어가", "넘어갔", "쓰러", "젖혀", "젖혔", "무너", "엎어", "뒤집", "자빠", "떨어지", "떨어졌", "떨어진", "떨어져", "떨어짐", "떨어뜨")
_SCENE_CHANGE = re.compile("|".join(_SCENE_CHANGE_GROUP))
_WHEN = re.compile(r"언제")
_WHY = re.compile(r"왜|어째서|어쩌다|이유|원인")


def _normalize_spelling(query: str) -> str:
    """같은 물건의 흔한 표기 차이를 맞춘다('쇼파'→'소파'). 이벤트 설명은 '소파'로 적힌다."""
    return query.replace("쇼파", "소파")


def _is_scene_change_statement(query: str) -> bool:
    """'소파가 넘어진 장면 보여줘'처럼 '언제' 없이 물건이 넘어진 일을 묻는 말인가.

    '강아지가 넘어졌어?'는 반려동물 행동이라 제외한다. 반려동물·질문 틀 낱말을 뗀 뒤 대상 낱말(물건)과 서술어가
    함께 남을 때만 장면 변화로 본다.
    """
    if not _SCENE_CHANGE.search(query):
        return False
    terms, _ = _condition_terms(query)
    return len(terms) >= 2


def is_sequence_query(query: str) -> bool:
    query = _normalize_spelling(query)
    compact = re.sub(r"\s+", "", query.lower())
    if any(word in compact for word in ("몇번", "몇회", "횟수", "몇개", "얼마나자주")):
        return False
    # '소파가 언제 넘어졌어?'처럼 장면이 바뀐 시각을 묻는 질문은 저장된 이벤트에서 찾는다.
    if _WHEN.search(compact) and _SCENE_CHANGE.search(compact):
        return True
    if _is_scene_change_statement(query):
        return True
    if _SUMMARY_WORDS.search(query):
        return True
    if any(re.sub(r"\s+", "", word) in compact for word in _SEQUENCE_WORDS):
        return True
    return _temporal_marker(query)


def _recorded_at(event: dict[str, Any]) -> datetime | None:
    for frame in event.get("source_frames") or []:
        if not isinstance(frame, dict):
            continue
        captured_at = trusted_capture_time(frame)
        if captured_at:
            frame_offset = float(frame.get("timestamp") or event.get("start_time") or 0)
            event_offset = float(event.get("start_time") or 0)
            return captured_at + timedelta(seconds=event_offset - frame_offset)
    return None


def _in_filter(recorded_at: datetime | None, recording_date: str | None, time_range: dict | None) -> bool:
    if recording_date and (recorded_at is None or recorded_at.date().isoformat() != recording_date):
        return False
    if not time_range:
        return True
    start = time_range.get("start_hour")
    end = time_range.get("end_hour")
    if start is None or end is None or (start <= 0 and end >= 24):
        return True
    if recorded_at is None:
        return False
    hour = recorded_at.hour + recorded_at.minute / 60
    return start <= hour < end if start <= end else hour >= start or hour < end


def _step_frames(frames: list[dict[str, Any]], anchor: float) -> list[dict[str, Any]]:
    """재생 구간(앵커 -3초~+5초) 안의 프레임을 시간순으로 최대 MAX_STEP_FRAMES장 고른다."""
    window_start, window_end = anchor - 3, anchor + 5
    seen: set[str] = set()
    in_window = []
    for item in sorted(frames, key=lambda f: float(f.get("timestamp") or 0)):
        key = str(item.get("s3_key") or "")
        timestamp = float(item.get("timestamp") or 0)
        if not key or key in seen or not window_start <= timestamp <= window_end:
            continue
        seen.add(key)
        in_window.append({"s3_key": key, "timestamp": timestamp})
    if len(in_window) <= MAX_STEP_FRAMES:
        return in_window
    # 처음과 끝을 포함해 균등하게 솎아낸다.
    last = len(in_window) - 1
    indexes = sorted({round(i * last / (MAX_STEP_FRAMES - 1)) for i in range(MAX_STEP_FRAMES)})
    return [in_window[i] for i in indexes]


def _description(event: dict[str, Any]) -> str:
    return str(event.get("summary") or event.get("action") or "움직임 감지").strip()


# 무슨 일이 있었는지 말해 주지 않는 설명. 사용자에게 보여 줘도 도움이 안 되고 타임라인만 길어진다.
# 예: "반려동물이 화면에서 사라져서 더 이상 확인할 수 없어요", "같은 개체의 행동 전후 상태가 20초에 확인돼요".
# '화면 밖으로 나감'처럼 실제 행동을 말하는 설명은 걸러내지 않도록, 확인 불가를 말하는 표현만 본다.
_UNINFORMATIVE = re.compile(
    r"확인할\s*수\s*없|확인되지\s*않|보이지\s*않|알\s*수\s*없|판단할\s*수\s*없"
    r"|행동\s*전후\s*상태|전후\s*상태가"
)


def _is_informative(event: dict[str, Any]) -> bool:
    """설명이 비어 있거나 '확인할 수 없음'류이면 시퀀스 답변에서 뺀다."""
    description = _description(event)
    return description != "움직임 감지" and not _UNINFORMATIVE.search(description)


# 저장된 설명의 문장 끝을 답변 말투(해요체)로 바꾸는 규칙. 위에서부터 처음 맞는 하나만 적용한다.
_SPOKEN_ENDINGS = (
    ("모습입니다", "모습이 보여요"),
    ("모습이다", "모습이 보여요"),
    ("장면입니다", "장면이 보여요"),
    ("장면이다", "장면이 보여요"),
    ("장면으로 보임", "장면으로 보여요"),
    ("것으로 보임", "것으로 보여요"),
    ("것으로 추정되는 구간입니다", "것으로 보여요"),
    ("중입니다", "중이에요"),
    ("보입니다", "보여요"),
    ("보인다", "보여요"),
    ("보임", "보여요"),
    ("관찰됨", "관찰돼요"),
    ("관찰된다", "관찰돼요"),
    ("있습니다", "있어요"),
    ("있음", "있어요"),
    ("있다", "있어요"),
    ("없음", "없어요"),
    ("잠을 잠", "잠을 자요"),
    ("내려옴", "내려와요"),
    ("올라감", "올라가요"),
    ("내밈", "내밀어요"),
    ("마심", "마셔요"),
    ("걸음", "걸어요"),
    ("달림", "달려요"),
    ("나타남", "나타나요"),
    ("움직임", "움직여요"),
    ("합니다", "해요"),
    ("한다", "해요"),
    ("됩니다", "돼요"),
    ("된다", "돼요"),
)
_NOUN_IM = re.compile(r"(행동|모습|장면|상황|상태|것)임$")


def _spoken_description(event: dict[str, Any]) -> str:
    """저장된 설명의 의미를 유지하면서 답변에 어울리는 말투로 바꾼다. 내용은 더하거나 빼지 않는다."""
    description = _description(event).rstrip(".。 ")
    for ending, replacement in _SPOKEN_ENDINGS:
        if description.endswith(ending):
            return description[: -len(ending)] + replacement
    return _NOUN_IM.sub(r"\1이에요", description)


def _clock_text(moment: datetime | None) -> str:
    """'오전 7시 34분쯤'처럼 말하듯 시각을 적는다."""
    if moment is None:
        return ""
    period = "오전" if moment.hour < 12 else "오후"
    return f"{period} {moment.hour % 12 or 12}시 {moment.minute}분 {moment.second}초쯤"


# 촬영 시각이 검증되지 않은 영상은 청크 번호와 청크 안 시각으로 영상 안의 위치를 말한다.
_CHUNK_FRAME_KEY = re.compile(r"_(\d{3})_frame_[0-9.]+\.jpg$")


def _video_seconds(event: dict[str, Any]) -> float | None:
    """촬영 시각이 없는 영상의 청크 번호와 청크 안 이벤트 시각을 합친다."""
    for frame in event.get("source_frames") or []:
        if not isinstance(frame, dict):
            continue
        match = _CHUNK_FRAME_KEY.search(str(frame.get("s3_key") or ""))
        if match:
            return int(match.group(1)) * CHUNK_SECONDS + float(event.get("start_time") or 0)
    return None


def _video_time_text(event: dict[str, Any]) -> str:
    """'영상 21초쯤'처럼 영상 안에서의 위치. 청크 번호를 알 수 없으면 청크 안 시각을 그대로 쓴다."""
    chunk_index = 0
    for frame in event.get("source_frames") or []:
        match = _CHUNK_FRAME_KEY.search(str((frame or {}).get("s3_key") or "")) if isinstance(frame, dict) else None
        if match:
            chunk_index = int(match.group(1))
            break
    seconds = int(chunk_index * CHUNK_SECONDS + float(event.get("start_time") or 0))
    minutes, rest = divmod(seconds, 60)
    return f"영상 {minutes}분 {rest}초쯤" if minutes else f"영상 {rest}초쯤"


def _video_range_text(event: dict[str, Any]) -> str | None:
    """변화가 있었던 구간이 3초 이상 벌어졌으면 '영상 22~25초 사이'처럼 범위로 적는다. 아니면 None.

    장면 변화는 변화 직전 프레임과 변화가 처음 보이는 프레임 사이 어딘가에서 일어났다. 시작 시각만 쓰면 실제보다 이르게 들린다.
    """
    start = float(event.get("start_time") or 0)
    end = float(event.get("end_time") or start)
    if end - start < 3.0:
        return None
    chunk_index = 0
    for frame in event.get("source_frames") or []:
        match = _CHUNK_FRAME_KEY.search(str((frame or {}).get("s3_key") or "")) if isinstance(frame, dict) else None
        if match:
            chunk_index = int(match.group(1))
            break
    low, high = int(chunk_index * CHUNK_SECONDS + start), int(chunk_index * CHUNK_SECONDS + end)
    if low // 60 != high // 60:
        return None
    minutes = low // 60
    prefix = f"영상 {minutes}분 " if minutes else "영상 "
    return f"{prefix}{low % 60}~{high % 60}초 사이"


# 장면 설명에 반려동물이 한 일이 원인으로 이미 적혀 있는가: '강아지가 줄을 당기면서 선반이 떨어졌어요'.
_PET_SUBJECT = re.compile(r"강아지|반려동물|반려견|고양이|아이가")
_PET_CAUSE_VERB = re.compile(r"당기|당겨|잡아당|끌|밀[어고며었]|물[어고며었]|잡아|부딪|치[고며면어]|건드|올라[가타]|뛰어|들이받")


def _states_pet_cause(event: dict[str, Any]) -> bool:
    text = f"{event.get('summary') or ''} {event.get('action') or ''}"
    return bool(_PET_SUBJECT.search(text) and _PET_CAUSE_VERB.search(text))


# 물건이 바뀐 일의 원인으로 볼 수 있는 직전 행동: 몇 초 안에 이어지고, 같은 물건에 대한 반려동물의 행동이어야 한다.
CAUSE_WINDOW_SECONDS = 5.0
_OBJECT_CORE = re.compile(r"[가-힣A-Za-z0-9]{2,}?(?=의|와|과|이|가|$)")


def _is_scene_change_event(event: dict[str, Any]) -> bool:
    subject = str(event.get("subject") or "").strip()
    return bool(subject) and subject == str(event.get("target_object") or "").strip()


def _likely_cause(scene_event: dict[str, Any], rows: list[tuple[datetime | None, dict]]) -> dict[str, Any] | None:
    """물건이 바뀌기 직전에 같은 물건에 대해 반려동물이 한 행동. 있으면 원인으로 추정할 수 있다."""
    match = _OBJECT_CORE.match(str(scene_event.get("target_object") or scene_event.get("subject") or ""))
    if not match:
        return None
    noun = match.group()
    started = float(scene_event.get("start_time") or 0)
    candidates = [
        event for _, event in rows
        if event is not scene_event
        and event.get("video_id") == scene_event.get("video_id")
        and not _is_scene_change_event(event)
        and float(event.get("start_time") or 0) < started
        and started - float(event.get("end_time") or event.get("start_time") or 0) <= CAUSE_WINDOW_SECONDS
        and noun in f"{event.get('summary') or ''} {event.get('action') or ''} {event.get('target_object') or ''}"
    ]
    return max(candidates, key=lambda event: float(event.get("end_time") or 0)) if candidates else None


def _cause_observation(event: dict[str, Any]) -> str:
    """저장된 장면 설명을 원인 추정 문장에 자연스럽게 잇는다."""
    description = _spoken_description(event).rstrip(". ")
    if description.endswith("고 있어요"):
        return f"{description[:-5]}고 있었던 점을 보면"
    if description.endswith("고 있었어요"):
        return f"{description[:-6]}고 있었던 점을 보면"
    return f"‘{description}’라는 장면이 나온 점을 보면"


def _place_particle(place: str) -> str:
    """장소 이름 뒤에 붙일 '과/와'를 고른다."""
    last = place[-1]
    code = ord(last) - 0xAC00
    return "과" if 0 <= code < 11172 and code % 28 else "와"


@lru_cache(maxsize=1)
def _reviewed_events() -> dict[str, dict[str, dict[str, Any]]]:
    """원본 영상과 대조한 시연 영상의 승인 이벤트만 읽는다."""
    path = Path(__file__).with_name("sequence_reviewed_events.json")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _reviewed_sequence_notes() -> dict[str, str]:
    path = Path(__file__).with_name("sequence_review_notes.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _sequence_intent(query: str) -> tuple[str, str]:
    """전체·첫/마지막·특정 행동의 앞/뒤 질문을 구분한다."""
    before = _BEFORE.search(query)
    after = _AFTER.search(query)
    if before and (not after or before.start() < after.start()):
        return "before", query[:before.start()].strip()
    if after:
        return "after", query[:after.start()].strip()
    if (_WHEN.search(query) or _is_scene_change_statement(query)) and not (
        _FIRST.search(query) or _LAST.search(query) or _WHOLE_RANGE.search(query)
    ):
        return "when", ""
    if _WHOLE_RANGE.search(query):
        return "timeline", ""
    first = _FIRST.search(query) or (None if _ORDER_WORDS.search(query) else _EARLY.search(query))
    last = _LAST.search(query)
    if first and last:
        return "timeline", ""
    if first:
        return "first", ""
    if last:
        return "last", ""
    return "timeline", ""


# 조건·기준 문구에서 행동/대상이 아닌 질문 틀 낱말. 이 낱말만 남으면 조건이 없는 것이다.
_CONDITION_FILLER = {
    "행동", "장면", "모습", "동작", "상황", "순서", "일", "건", "거", "것", "게", "걸", "한", "했", "하는", "하던", "하고",
    "본", "봤어", "있었어", "있었던", "적", "있어", "있나", "있니", "있나요", "있어요", "있었나", "있었니", "뭐", "뭘", "뭐야", "무엇", "무슨", "어떤", "어디", "언제", "어땠어", "어땠나", "왜", "어째서", "어쩌다", "이유", "원인",
    "장소", "위치", "시간", "시각", "때", "처음", "첫", "번째", "마지막", "맨", "제일", "가장", "먼저", "끝", "나중",
    "오늘", "어제", "아까", "하루", "오전", "오후", "아침", "점심", "저녁", "밤", "새벽", "최근", "강아지", "고양이",
    "반려동물", "우리", "아이", "그", "알려줘", "알려", "줘", "보여줘", "보여", "말해줘", "해줘",
}
_CONDITION_FILLER_PREFIX = re.compile(r"^(?:했|알려|보여|말해|뭐|어땠|어떻|언제)\S*$")
# 조사는 한 번만, 겹조사까지 한꺼번에 뗀다. 두 번 떼면 '사과를'→'사과'→'사'처럼 낱말이 깎인다.
_CONDITION_PARTICLE = re.compile(
    r"(?:에서는|에서도|으로는|으로도|에서|에게|에는|에도|으로|하고|이랑|까지|부터|이나|은|는|이|가|을|를|에|도|만|랑|로|과|와|야|요)$"
)
_SINGLE_PARTICLE = r"(?:에서|에게|으로|까지|부터|이랑|을|를|이|가|은|는|에|로|도|와|과|랑|만|의)?"
_SUBJECT_WORDS = re.compile(r"반려동물|강아지|고양이")

Term = tuple[str, ...]  # 같은 낱말의 대체 표기. 하나라도 설명에 맞으면 맞은 것이다.


def _content_terms(text: str) -> tuple[list[Term], str]:
    """'물 마신'처럼 행동·대상·장소 낱말만 남긴다. 첫/마지막 조건과 앞/뒤 기준 장면이 같은 규칙을 쓴다.

    조사를 뗀 형태만 쓰면 '사과'가 '사'로 깎이므로, 원래 낱말과 뗀 낱말을 함께 둔다.
    """
    terms: list[Term] = []
    raw: list[str] = []
    for token in re.findall(r"[가-힣A-Za-z0-9]+", text.lower()):
        if _CONDITION_PARTICLE.fullmatch(token):  # 표지를 지우고 남은 조사('처음에'→'에')
            continue
        stripped = _CONDITION_PARTICLE.sub("", token) if len(token) > 1 else token
        if any(form in _CONDITION_FILLER or _CONDITION_FILLER_PREFIX.match(form) for form in (token, stripped)):
            continue
        if stripped:
            terms.append(tuple(dict.fromkeys((token, stripped))))
            raw.append(token)
    return terms, " ".join(raw)


def _condition_terms(query: str) -> tuple[list[Term], str]:
    """'마지막으로 물 마신 건 언제야?'에서 순서 표지를 지우고 행동 조건(물, 마신)만 남긴다."""
    text = _EARLY.sub(" ", _FIRST.sub(" ", _LAST.sub(" ", query)))
    return _content_terms(text)


# '소파에서 뭐했어?'·'소파 위에서'처럼 장소 표지가 붙은 낱말. 질문이 특정 장소의 행동을 묻는다는 뜻이다.
_PLACE_MARKED = re.compile(
    r"([가-힣A-Za-z0-9]{2,})\s*(?:위에서|위에|아래에서|아래에|옆에서|옆에|근처에서|근처에|앞에서|앞에|속에서|속에|밑에서|밑에|에서)(?![가-힣])"
)
# 장소가 아니라 시간·범위·화면을 가리키는 말이라, 이벤트 설명에 없어도 되는 낱말.
_GENERIC_PLACES = {"집", "방", "여기", "거기", "화면", "영상", "전체", "평소", "하루", "오늘", "어제", "아까", "밤", "낮", "아침", "저녁", "새벽", "어디", "언제", "사이"}


def _place_terms(query: str) -> list[str]:
    """질문에서 장소 표지가 붙은 낱말만 뽑는다. 반려동물·질문 틀 낱말과 일반적인 말은 뺀다."""
    places = []
    for match in _PLACE_MARKED.finditer(query):
        word = match.group(1)
        if word in _CONDITION_FILLER or word in _GENERIC_PLACES or _SUBJECT_WORDS.fullmatch(word):
            continue
        if word not in places:
            places.append(word)
    return places


def _variants(term: str) -> list[str]:
    """'마신'→'마시', '뛰어다닌'→'뛰어다니', '마시기'→'마시'처럼 활용 어미를 뗀 어간 후보를 함께 만든다. 한 글자 어간은 오매칭이 커서 만들지 않는다."""
    variants = [term]
    code = ord(term[-1]) - 0xAC00
    if 0 <= code < 11172 and code % 28 in (4, 8):
        stem = term[:-1] + chr(ord(term[-1]) - code % 28)
        if len(stem) >= 2:
            variants.append(stem)
            if code % 28 == 4:  # '내려온'과 저장 설명 '내려옴'처럼 ㄴ/ㅁ 명사형을 연결한다.
                variants.append(term[:-1] + chr(ord(term[-1]) - 4 + 16))
    elif term[-1] in "기고" and len(term) >= 3:  # '마시기'·'마시고' → '마시'
        variants.append(term[:-1])
    return variants


def _term_in(term: str, text: str) -> bool:
    # '넘어졌어'는 '젖혀지면서'·'쓰러져'로 적힌 같은 일에도 맞아야 한다.
    if any(word in term for word in _SCENE_CHANGE_GROUP) and any(word in text for word in _SCENE_CHANGE_GROUP):
        return True
    for variant in _variants(term):
        if len(variant) == 1:
            # 한 글자 낱말('물', '사')은 부분 문자열이면 '물그릇'·'사료'에도 맞으므로, 조사까지 붙은 독립된 말일 때만 인정한다.
            if re.search(rf"{_START}{re.escape(variant)}{_SINGLE_PARTICLE}{_END}", text):
                return True
        elif variant in text:
            return True
    return False


def _term_hit(term: Term, text: str) -> bool:
    return any(_term_in(form, text) for form in term)


def _match_text(event: dict[str, Any]) -> str:
    # 모든 설명에 들어 있는 주어 낱말은 지운다. 그렇지 않으면 '물'이 '반려동물'에도 맞아 모든 이벤트가 같은 점수가 된다.
    return _SUBJECT_WORDS.sub(" ", _description(event).lower())


def _match_indices(rows: list[tuple[datetime | None, dict]], terms: list[Term]) -> list[int]:
    """조건에 맞는 이벤트 위치. 한국어는 서술어가 끝에 오므로 마지막 낱말(예: '마신')만 활용형 차이를 감안해
    느슨하게 보고, 그 앞의 대상·장소 낱말(예: '물', '소파')은 모두 설명에 있어야 한다.
    그래야 '간식 먹은'이 '사료 먹는' 이벤트에 걸리지 않는다."""
    texts = [_match_text(event) for _, event in rows]
    required = terms[:-1] if len(terms) > 1 else terms
    candidates = [i for i, text in enumerate(texts) if all(_term_hit(term, text) for term in required)]
    if len(terms) > 1:
        with_predicate = [i for i in candidates if _term_hit(terms[-1], texts[i])]
        candidates = with_predicate or candidates
    return candidates


def _pick_by_condition(rows: list[tuple[datetime | None, dict]], terms: list[Term], last: bool) -> int | None:
    candidates = _match_indices(rows, terms)
    if not candidates:
        return None
    return candidates[-1] if last else candidates[0]


def _unsupported_anchor_claim(anchor: str, event: dict[str, Any]) -> bool:
    """장소만 일치할 때 질문 속 행동까지 확인된 것으로 취급하지 않는다."""
    description = _description(event)
    claims = (
        (r"나간|나갔|나가기|나왔|나온|나와|나오|빠져나", r"나가|나왔|나온|나와|나오|빠져나|통과"),
        (r"수영하|수영했|헤엄", r"수영하|수영했|헤엄"),
        (r"먹|마시", r"먹|마시"),
        (r"놀", r"놀"),
    )
    return any(re.search(claim, anchor) and not re.search(evidence, description) for claim, evidence in claims)


# 같은 행동이 연달아 기록된 이벤트를 한 단계로 묶기 위한 보수적인 행동 분류.
# 어휘에 없거나 "하지 않음/하려는/것 같음"처럼 실제 행동이 불확실한 설명은 분류하지 않아 묶이지 않는다.
_ACTION_CATEGORIES = (
    ("drink", re.compile(r"마시|마셔|수분을 섭취|음수")),
    ("eat", re.compile(r"먹(?!이)|섭취|식사")),
    ("hide", re.compile(r"숨|웅크")),
    ("rest", re.compile(r"누워|낮잠|잠을 자|휴식|엎드려")),
    ("explore", re.compile(r"탐색|돌아다니|배회")),
)
# "멈추"류: "먹기를 멈추고 고개를 든다"는 먹는 행동이 아니라 먹기를 멈춘 장면이다. 먹기로 분류하면
# 앞뒤의 먹는 장면과 한 단계로 묶여 가운데 장면이 사라진다.
_UNCERTAIN_ACTION = re.compile(r"않|못|려고|려는|하려|것 같|듯|멈추|멈춰|멈춘|멈췄")
MAX_MERGE_GAP_SECONDS = 120


def _action_category(event: dict[str, Any]) -> str | None:
    description = _description(event)
    if _UNCERTAIN_ACTION.search(description):
        return None
    return next((name for name, pattern in _ACTION_CATEGORIES if pattern.search(description)), None)


def _labels(event: dict[str, Any]) -> tuple[str | None, bool | None, str | None]:
    """모델이 붙인 (action_type, continues_previous, call_id). 이벤트 최상위 또는 근거 프레임의 event_labels에서 읽는다.

    behavior_events 테이블에 라벨 컬럼이 없어 추출 단계가 source_frames_json에 함께 저장한다.
    call_id는 같은 모델 호출(같은 청크의 같은 Window)에서 나온 이벤트를 알아보는 식별자다.
    라벨이 없는 기존 이벤트는 (None, None, None)이다.
    """
    labels: dict[str, Any] = {}
    if any(key in event for key in ("action_type", "continues_previous", "call_id")):
        labels = event
    else:
        for frame in event.get("source_frames") or []:
            found = frame.get("event_labels") if isinstance(frame, dict) else None
            if isinstance(found, dict):
                labels = found
                break
    action_type, continues, call_id = labels.get("action_type"), labels.get("continues_previous"), labels.get("call_id")
    return (
        action_type if isinstance(action_type, str) else None,
        continues if isinstance(continues, bool) else None,
        call_id if isinstance(call_id, str) else None,
    )


def _same_action(previous: dict[str, Any], event: dict[str, Any]) -> bool:
    """두 이웃 이벤트가 '같은 행동이 이어진' 것인가.

    - 같은 모델 호출에서 나온 둘 다 라벨이 있으면 모델 판단을 따른다: action_type이 같고(other 제외)
      continues_previous가 true일 때만 같다. 타입만 같은 것은 같다고 보지 않는다(서로 다른 개체·방향 전환 구분 불가).
    - 호출이 다르면(청크·Window 경계) continues_previous는 알 수 없다. 둘 다 라벨이 있고 타입이 다르면 다른 행동이고,
      그 밖에는 기존 키워드 분류(_action_category)가 같을 때만 묶는다. 라벨이 없는 기존 이벤트도 같다.
    """
    prev_type, _, prev_call = _labels(previous)
    next_type, next_continues, next_call = _labels(event)
    both_labeled = prev_type is not None and next_type is not None
    if both_labeled and prev_call and prev_call == next_call and next_continues is not None:
        return next_type == prev_type and next_type != "other" and next_continues is True
    if both_labeled and prev_type != next_type:
        return False
    category = _action_category(event)
    return category is not None and _action_category(previous) == category


def _event_end(recorded_at: datetime | None, event: dict[str, Any]) -> datetime | None:
    if recorded_at is None:
        return None
    span = max(float(event.get("end_time") or 0) - float(event.get("start_time") or 0), 0)
    return recorded_at + timedelta(seconds=span)


def _group_consecutive(rows: list[tuple[datetime | None, dict]]) -> list[list[tuple[datetime | None, dict]]]:
    """같은 영상에서 같은 행동이 시간상 바로 이어지면 한 묶음으로 만든다."""
    groups: list[list[tuple[datetime | None, dict]]] = []
    for row in rows:
        recorded_at, event = row
        if groups:
            last_recorded, last_event = groups[-1][-1]
            last_end = _event_end(last_recorded, last_event)
            same_video = last_event.get("video_id") == event.get("video_id")
            close = (
                last_end is not None
                and recorded_at is not None
                and (recorded_at - last_end).total_seconds() <= MAX_MERGE_GAP_SECONDS
            )
            if last_recorded is None and recorded_at is None:
                last_seconds = _video_seconds(last_event)
                next_seconds = _video_seconds(event)
                if last_seconds is not None and next_seconds is not None:
                    last_duration = max(float(last_event.get("end_time") or 0) - float(last_event.get("start_time") or 0), 0)
                    close = next_seconds - (last_seconds + last_duration) <= MAX_MERGE_GAP_SECONDS
            if same_video and close and _same_action(last_event, event):
                groups[-1].append(row)
                continue
        groups.append([row])
    return groups


def _collapse_same_moment(rows: list[tuple[datetime | None, dict]]) -> list[tuple[datetime | None, dict]]:
    """Show one representative event per video second in a broad timeline."""
    from pipeline.search.query_suggester import _event_score

    chosen: dict[tuple[str, int], tuple[datetime | None, dict]] = {}
    for row in rows:
        recorded_at, event = row
        seconds = _video_seconds(event)
        if seconds is None and recorded_at is None:
            # No reliable instant: keep the event rather than inventing a duplicate.
            chosen[(str(event.get("video_id")), -int(event.get("id") or id(event)))] = row
            continue
        moment = int(recorded_at.timestamp()) if recorded_at is not None else int(seconds)
        key = (str(event.get("video_id")), moment)
        previous = chosen.get(key)
        if previous is None or (
            _event_score(event), len(_description(event))
        ) > (
            _event_score(previous[1]), len(_description(previous[1]))
        ):
            chosen[key] = row
    kept = {id(row[1]) for row in chosen.values()}
    return [row for row in rows if id(row[1]) in kept]


def _summary_groups(groups: list[list[tuple[datetime | None, dict]]], limit: int = 5):
    """Pick diverse, well supported actions across the whole timeline."""
    from pipeline.search.query_suggester import _dedupe_events_by_category, _event_score

    representatives = [group[0][1] for group in groups]
    selected = {id(event) for event in _dedupe_events_by_category(representatives, limit)}
    # When fewer than five broad categories occur, show other distinct moments
    # rather than leaving useful summary slots empty.
    for event in sorted(representatives, key=_event_score, reverse=True):
        if len(selected) >= min(limit, len(representatives)):
            break
        selected.add(id(event))
    return [group for group in groups if id(group[0][1]) in selected]


def _episode_spans(rows: list[tuple[datetime | None, dict]]) -> list[tuple[int, int]]:
    """FULL 타임라인과 같은 묶음(에피소드)의 [시작, 끝] 위치. 앞/뒤 질문도 이 단위로 이웃을 정한다."""
    spans, start = [], 0
    for group in _group_consecutive(rows):
        spans.append((start, start + len(group) - 1))
        start += len(group)
    return spans


def _find_anchor(
    rows: list[tuple[datetime | None, dict]], spans: list[tuple[int, int]], anchor: str
) -> tuple[int, int] | None:
    """기준 장면이 속한 에피소드. 같은 에피소드 안의 여러 이벤트는 하나로 보고, 서로 다른 에피소드에 걸리면 모호하다."""
    terms, _ = _content_terms(anchor)
    if not terms:
        return None
    # 장소만 맞는 입수·가장자리 장면을 '수영장에서 나온' 기준으로 고르지 않는다.
    all_indices = _match_indices(rows, terms)
    indices = [index for index in all_indices if not _unsupported_anchor_claim(anchor, rows[index][1])]
    matched = {span for index in (indices or all_indices) for span in spans if span[0] <= index <= span[1]}
    return next(iter(matched)) if len(matched) == 1 else None


def _playable(recorded_at: datetime | None, event: dict[str, Any]) -> bool:
    """시각을 알고 원본 프레임(s3_key)으로 다시 열 수 있는 이벤트인가."""
    return _has_frame(recorded_at, event) and (recorded_at is not None or _video_seconds(event) is not None)


def _has_frame(_recorded_at: datetime | None, event: dict[str, Any]) -> bool:
    """촬영 시각은 몰라도 원본 프레임(s3_key)으로 다시 열 수 있는 이벤트인가."""
    return any(isinstance(frame, dict) and frame.get("s3_key") for frame in event.get("source_frames") or [])


def _card(event: dict[str, Any], merged_count: int = 1, role: str | None = None) -> dict[str, Any]:
    frames = event.get("source_frames") or []
    valid_frames = [item for item in frames if isinstance(item, dict)]
    reviewed_at = event.get("reviewed_timestamp")
    frame = (
        min(valid_frames, key=lambda item: abs(float(item.get("timestamp") or 0) - float(reviewed_at)))
        if reviewed_at is not None and valid_frames
        else valid_frames[0] if valid_frames else {}
    )
    timestamp = float(frame.get("timestamp") or event.get("start_time") or 0)
    card = {
        "summary": _spoken_description(event),
        "merged_count": merged_count,
        "step_frames": _step_frames(valid_frames, timestamp),
        "frame_id": frame.get("frame_id") or f"event-{event.get('id')}",
        "frame_path": frame.get("frame_path"),
        "s3_key": frame.get("s3_key"),
        "timestamp": timestamp,
        "video_id": event.get("video_id"),
        "object_labels": frame.get("object_labels") or "",
        "score": float(event.get("confidence") or 0),
        "source_event_id": event.get("id"),
        "event_start": event.get("start_time"),
        "event_end": event.get("end_time"),
        "representative_timestamp": timestamp,
        "playback_start": max(timestamp, 0),
        "playback_end": timestamp + 5,
    }
    if role:
        card["evidence_role"] = role
    return card


# 모션·장면 변화로 만든 이벤트는 "순간"(0.1초)만 기록하는 경우가 많다. 그 상태는 실제로는 그 앞뒤로 이어지므로
# 길이가 INSTANT_EVENT_SECONDS 이하인 순간 이벤트만 카드 구간(event_start/event_end)을 MIN_STEP_SPAN_SECONDS까지
# 넓힌다. 이미 구간을 가진 이벤트는 건드리지 않는다. 이웃 단계와는 겹치지 않게 중간에서 끊는다.
INSTANT_EVENT_SECONDS = 0.6
MIN_STEP_SPAN_SECONDS = 3.0


def _widen_step_spans(cards: list[dict[str, Any]], groups: list[list[tuple[datetime | None, dict]]]) -> None:
    """순간 이벤트의 구간을 앞뒤로 넓힌다. 원본 이벤트 값은 raw_event_start/raw_event_end에 남긴다."""
    spans: list[tuple[float, float] | None] = []
    for card, group in zip(cards, groups):
        try:
            start = float(card.get("event_start"))
            end = max(float(member.get("end_time") if member.get("end_time") is not None else start) for _, member in group)
        except (TypeError, ValueError):
            spans.append(None)
            continue
        spans.append((start, max(end, start)))
    for index, (card, span) in enumerate(zip(cards, spans)):
        if span is None:
            continue
        start, end = span
        card["raw_event_start"], card["raw_event_end"] = card.get("event_start"), card.get("event_end")
        new_start, new_end = start, end
        missing = MIN_STEP_SPAN_SECONDS - (end - start)
        if end - start <= INSTANT_EVENT_SECONDS and missing > 0:
            new_start, new_end = start - missing / 2, end + missing / 2
        # 같은 영상의 이웃 단계와 겹치면 두 구간 사이 중간에서 끊는다.
        if index > 0 and spans[index - 1] and cards[index - 1].get("video_id") == card.get("video_id"):
            new_start = max(new_start, min(start, (spans[index - 1][1] + start) / 2))
        if index + 1 < len(cards) and spans[index + 1] and cards[index + 1].get("video_id") == card.get("video_id"):
            new_end = min(new_end, max(end, (end + spans[index + 1][0]) / 2))
        card["event_start"] = round(max(new_start, 0.0), 2)
        card["event_end"] = round(max(new_end, card["event_start"]), 2)


def run_sequence_query(
    query: str,
    video_id: str | None = None,
    top_k: int = 5,
    recording_date: str | None = None,
    time_range: dict | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict:
    """전체 후보를 시간순으로 정렬하고 근거 프레임을 함께 돌려준다.

    이벤트의 start_time은 영상 전체가 아니라 60초 청크 안의 상대 시간이다(청크마다 따로 처리된다).
    DB의 `ORDER BY video_id, start_time`은 시간순이 아니므로, 개수 제한을 DB 순서로 자르면
    최신 이벤트가 빠진다. 그래서 여기서는 자르지 않고 전부 받아 청크 번호로 시각을 복원한 뒤 정렬한다.
    """
    query_as_asked = query
    query = _normalize_spelling(query)
    if events is None:
        from db.behavior_events import get_behavior_events_for_aggregation
        events = get_behavior_events_for_aggregation(video_id=video_id)
    truncated = False

    rows = []
    excluded_event_count = 0
    uninformative_event_count = 0
    reviewed = _reviewed_events()
    for event in events:
        recorded_at = _recorded_at(event)
        if _in_filter(recorded_at, recording_date, time_range):
            video_review = reviewed.get(str(event.get("video_id") or ""))
            if video_review is not None:
                approval = video_review.get(str(event.get("id")))
                if not approval:
                    excluded_event_count += 1
                    continue
                event = {
                    **event,
                    "summary": approval["summary"],
                    "action": approval["summary"],
                    "reviewed_timestamp": float(approval["timestamp"]),
                    "review_status": "approved",
                }
            if not _is_informative(event):
                uninformative_event_count += 1
                continue
            rows.append((recorded_at, event))

    rows.sort(key=lambda row: (
        row[0] is None,
        row[0] or datetime.max,
        str(row[1].get("video_id") or ""),
        _video_seconds(row[1]) if row[0] is None and _video_seconds(row[1]) is not None
        else float(row[1].get("start_time") or 0),
        int(row[1].get("id") or 0),
    ))
    intent, anchor = _sequence_intent(query)
    summary_request = intent == "timeline" and bool(
        re.search(r"요약|주요\s*행동|대표\s*행동", query)
        or (_BROAD_DAY_ACTIONS.search(query) and not _ORDER_WORDS.search(query))
    )
    if intent == "timeline":
        rows = _collapse_same_moment(rows)
    all_rows = rows
    unplayable_count = 0
    if intent != "timeline":
        # 처음/마지막/앞/뒤 답은 시각을 알고 다시 재생할 수 있는 이벤트로만 한다.
        # 프레임이 없는 레거시 이벤트는 시각이 없어 맨 뒤로 정렬되므로 '마지막'이나 이웃으로 내면 안 된다.
        # 다만 '언제' 질문은 시각을 묻는 것이라, 촬영 시각이 없어도 프레임이 있으면 영상 안의 위치로 답할 수 있다.
        keep = _has_frame if intent == "when" else _playable
        rows = [row for row in all_rows if keep(*row)]
        unplayable_count = len(all_rows) - len(rows)
    # '소파에서 뭐했어?'처럼 장소가 붙은 시간순 질문은 그 장소가 나온 장면만 모은다.
    # 조건을 무시하고 전체 타임라인을 보여 주면 소파와 상관없는 행동이 답으로 나간다.
    place_terms = _place_terms(query) if intent == "timeline" else []
    if place_terms:
        rows = [row for row in rows if all(_term_in(term, _match_text(row[1])) for term in place_terms)]
    selected_groups = None
    step_total = len(rows)
    anchor_span = None
    anchor_claim_unsupported = False
    condition_text = ""
    condition_status = None  # None | "missing"(조건에 맞는 장면 없음) | "unsupported"(장소만 맞고 행동은 확인 안 됨)
    condition_near = None
    when_total = 0
    needs_video = intent in {"before", "after"} and not video_id
    evidence_rows: list[tuple[datetime | None, dict, int, str | None]] = []  # (row, 묶음 크기, 역할)
    if needs_video:
        selected = []
    elif intent in {"before", "after"}:
        spans = _episode_spans(rows)
        anchor_span = _find_anchor(rows, spans, anchor)
        if anchor_span is not None:
            anchor_claim_unsupported = all(
                _unsupported_anchor_claim(anchor, rows[k][1]) for k in range(anchor_span[0], anchor_span[1] + 1)
            )
        selected = []
        if anchor_span is not None and not anchor_claim_unsupported:
            neighbor = next(
                (span for span in spans if span[1] == anchor_span[0] - 1) if intent == "before"
                else (span for span in spans if span[0] == anchor_span[1] + 1),
                None,
            )
            anchor_row = rows[anchor_span[0]]
            anchor_item = (*anchor_row, anchor_span[1] - anchor_span[0] + 1, "anchor")
            if neighbor is None:
                evidence_rows = [anchor_item]
                selected_groups = [rows[anchor_span[0]:anchor_span[1] + 1]]
            else:
                selected = [rows[neighbor[0]]]
                neighbor_item = (*rows[neighbor[0]], neighbor[1] - neighbor[0] + 1, "answer")
                ordered = [neighbor_item, anchor_item] if intent == "before" else [anchor_item, neighbor_item]
                evidence_rows = ordered
                spans_in_order = [neighbor, anchor_span] if intent == "before" else [anchor_span, neighbor]
                selected_groups = [rows[lo:hi + 1] for lo, hi in spans_in_order]
    elif intent == "when":
        terms, condition_text = _condition_terms(query)
        matched = _match_indices(rows, terms) if terms else []
        if len(terms) > 1:
            # 시각을 묻는 질문은 '소파'만 맞고 '넘어짐'은 없는 장면을 답으로 내면 안 되므로 서술어도 반드시 맞아야 한다.
            matched = [i for i in matched if _term_hit(terms[-1], _match_text(rows[i][1]))]
        if not terms:
            intent = "timeline"
            groups = _group_consecutive(rows)
            selected_groups = _summary_groups(groups) if summary_request else groups[:top_k]
            selected = [group[0] for group in selected_groups]
            step_total = len(groups)
        else:
            when_rows = [rows[i] for i in matched]
            selected = when_rows[:3]
            when_total = len(when_rows)
    elif intent in {"first", "last"}:
        terms, condition_text = _condition_terms(query)
        if not terms:
            selected = rows[:1] if intent == "first" else rows[-1:]
        else:
            picked = _pick_by_condition(rows, terms, last=intent == "last")
            if picked is None:
                condition_status = "missing"
                selected = []
            elif _unsupported_anchor_claim(
                condition_text + " " + " ".join(_variants(word)[-1] for word in condition_text.split()), rows[picked][1]
            ):
                condition_status = "unsupported"
                condition_near = rows[picked][1]
                selected = []
            else:
                selected = [rows[picked]]
    else:
        groups = _group_consecutive(rows)
        selected_groups = _summary_groups(groups) if summary_request else groups[:top_k]
        selected = [group[0] for group in selected_groups]
        step_total = len(groups)
    if selected_groups is None:
        selected_groups = [[row] for row in selected]
    merged_counts = {id(group[0][1]): len(group) for group in selected_groups}
    if intent in {"before", "after"}:
        results = [_card(event, size, role) for _, event, size, role in evidence_rows if role == "answer"]
        evidence_items = [_card(event, size, role) for _, event, size, role in evidence_rows]
    else:
        results = [_card(event, merged_counts.get(id(event), 1)) for _, event in selected]
        for card, group in zip(results, selected_groups):
            if len(group) > 1:
                card["merged_summaries"] = [_spoken_description(member) for _, member in group]
        _widen_step_spans(results, selected_groups)
        evidence_items = results

    anchor_event = rows[anchor_span[0]][1] if anchor_span is not None else None
    skipped_note = f" 재생할 수 있는 근거가 없는 장면 {unplayable_count}개는 제외했어요." if unplayable_count else ""
    if needs_video:
        answer = "어느 영상의 행동인지 선택한 뒤 다시 질문해 주세요."
    elif intent in {"before", "after"} and rows and anchor_span is None:
        answer = "어떤 장면을 기준으로 할지 하나로 좁히지 못했어요. 행동이나 장소를 조금 더 구체적으로 적어 주세요."
    elif intent in {"before", "after"} and anchor_claim_unsupported:
        answer = f"가까운 장면은 ‘{_spoken_description(anchor_event)}’인데, 질문하신 행동까지는 보이지 않아요. 그래서 그 행동의 앞뒤는 말씀드리기 어려워요."
    elif intent in {"before", "after"} and anchor_span is not None:
        if selected:
            first, second = (anchor_event, selected[0][1]) if intent == "after" else (selected[0][1], anchor_event)
            answer = f"먼저 {_spoken_description(first)}. 그다음에는 {_spoken_description(second)}."
        else:
            where = "그 뒤로는" if intent == "after" else "그 앞에는"
            answer = f"{_spoken_description(anchor_event)}. {where} 기록된 장면이 없어요."
        if selected and excluded_event_count:
            answer += " 빠진 구간이 있을 수 있어요."
        elif not selected and excluded_event_count:
            answer += " 다만 제외된 구간이 있어서, 실제로 다른 행동이 없었다는 뜻은 아니에요."
        if video_id in reviewed:
            answer += " 이 영상은 여러 장면을 이어 붙인 거라 같은 동물의 연속 행동인지는 알 수 없어요."
        answer += skipped_note
    elif place_terms and not rows:
        place = "·".join(place_terms)
        answer = f"분석된 장면에서는 {place}{_place_particle(place)} 관련된 행동을 찾지 못했어요."
        if excluded_event_count:
            answer += " 다만 제외된 구간이 있어서, 실제로 없었다는 뜻은 아니에요."
    elif not rows:
        if unplayable_count and not excluded_event_count:
            answer = "다시 재생할 수 있는 근거가 있는 행동 장면이 아직 없어요."
        else:
            answer = "선택하신 영상과 시간대에서는 보여드릴 수 있는 행동 장면이 없어요." if excluded_event_count else "선택하신 영상과 시간대에는 아직 분석된 행동이 없어요."
    elif intent == "when":
        if not selected:
            answer = "분석된 장면 중에서는 그 순간을 찾지 못했어요."
            if excluded_event_count:
                answer += " 다만 제외된 구간이 있어서, 실제로 없었다는 뜻은 아니에요."
        else:
            first_at, first_event = selected[0]
            clock = _clock_text(first_at) or _video_range_text(first_event) or _video_time_text(first_event)
            answer = f"{clock}에 {_spoken_description(first_event)}."
            if _WHY.search(query) and _states_pet_cause(first_event):
                # 장면 설명이 원인(반려동물의 행동)을 이미 담고 있다. 모른다고 덧붙이거나 관련 없는 직전 장면을 끌어오지 않는다.
                pass
            elif _WHY.search(query):
                # 이벤트에 원인이 직접 기록되지는 않으므로, 직전 행동이 조건에 맞을 때만 추정한다.
                cause = _likely_cause(first_event, rows)
                if cause is not None:
                    # 원인이 영상에 직접 찍힌 것은 아니므로 추정이라고 밝히고, 근거 장면을 함께 보여 준다.
                    answer += (
                        f" 직전에 {_cause_observation(cause)}, 그 행동이 영향을 준 것으로 보여요."
                    )
                else:
                    answer += " 왜 그랬는지는 영상만으로는 알 수 없어요."
                    started = float(first_event.get("start_time") or 0)
                    earlier = [
                        event for _, event in rows
                        if event.get("video_id") == first_event.get("video_id")
                        and float(event.get("start_time") or 0) < started
                    ]
                    if earlier:
                        answer += f" 다만 그 직전에는 ‘{_spoken_description(earlier[-1])}’ 장면이 있었어요."
            if when_total > 1:
                answer += f" 비슷한 장면이 모두 {when_total}번 있었어요."
        answer += skipped_note
    elif intent in {"first", "last"} and condition_status:
        position = "처음" if intent == "first" else "마지막"
        if condition_status == "unsupported":
            answer = f"가까운 장면은 ‘{_spoken_description(condition_near)}’인데, 질문하신 행동까지는 보이지 않아요. 그래서 그 행동을 {position}에 했는지는 말씀드리기 어려워요."
        else:
            answer = "분석된 장면 중에서는 질문하신 행동을 찾지 못했어요."
        if excluded_event_count:
            answer += " 다만 제외된 구간이 있어서, 실제로 없었다는 뜻은 아니에요."
        answer += skipped_note
    elif intent in {"first", "last"}:
        position = "처음" if intent == "first" else "마지막"
        answer = f"{position}에는 {_spoken_description(selected[0][1])}."
        if excluded_event_count:
            answer += " 제외된 구간이 있어서 영상 전체의 처음·마지막과는 다를 수 있어요."
        answer += skipped_note
    else:
        transitions = ("먼저", "그다음에는", "이어서")
        sentences = []
        for index, (_, event) in enumerate(selected):
            sentence = f"{transitions[min(index, 2)]} {_spoken_description(event)}."
            merged = merged_counts.get(id(event), 1)
            if merged > 1:
                sentence += f" 비슷한 장면이 모두 {merged}번 이어져요."
                # 묶음의 대표는 첫 장면이라, 끝에서 달라진 장면(예: 줄을 당기다 선반이 떨어짐)이 가려지지 않게 덧붙인다.
                last_description = _spoken_description(selected_groups[index][-1][1])
                if last_description != _spoken_description(event):
                    sentence += f" 마지막에는 {last_description}."
            sentences.append(sentence)
        answer = " ".join(sentences)
        if len(selected) < step_total:
            if summary_request:
                answer += f"\n전체 {step_total}개 장면 중 대표 행동 {len(selected)}개를 보여드려요."
            else:
                answer += f"\n전체 {step_total}개 장면 중 처음 {len(selected)}개만 보여드려요."
        if place_terms:
            place = "·".join(place_terms)
            answer = f"{place}{_place_particle(place)} 관련된 행동을 모았어요. " + answer

    if intent == "timeline" and results and video_id in _reviewed_sequence_notes():
        answer += " " + _reviewed_sequence_notes()[video_id]

    return {
        "query": query_as_asked,
        "query_type": "sequence",
        "video_id": video_id,
        "answer": answer,
        "results": results,
        "evidence_items": evidence_items,
        "behavior_events": [event for group in selected_groups for _, event in group],
        "used_llm": False,
        "total_event_count": len(all_rows),
        "total_step_count": step_total,
        "excluded_event_count": excluded_event_count,
        "uninformative_event_count": uninformative_event_count,
        "sequence_intent": intent,
        "truncated": truncated,
    }
