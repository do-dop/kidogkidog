"""행동 이벤트를 시간순으로 연결하는 시퀀스 질문 경로."""

import re
import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

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
_LAST = re.compile(rf"마지막|맨\s*끝|(?:제일|가장)\s*나중|끝(?:에는|에|엔){_END}")
_ASKS_ABOUT_ACTION = re.compile(r"뭐|무엇|무슨|행동|했|어땠|어떻")


def _temporal_marker(query: str) -> bool:
    if _BEFORE.search(query) or _AFTER.search(query) or _FIRST.search(query) or _EARLY.search(query):
        return True
    last = _LAST.search(query)
    # '책상 끝에 있어?'처럼 위치를 묻는 말과 구분하려고, '끝에'는 행동을 묻는 말과 함께 있을 때만 인정한다.
    return bool(last and (not last.group().startswith("끝") or _ASKS_ABOUT_ACTION.search(query)))


def is_sequence_query(query: str) -> bool:
    compact = re.sub(r"\s+", "", query.lower())
    if any(word in compact for word in ("몇번", "몇회", "횟수", "몇개", "얼마나자주")):
        return False
    if any(re.sub(r"\s+", "", word) in compact for word in _SEQUENCE_WORDS):
        return True
    return _temporal_marker(query)


def _recorded_at(event: dict[str, Any]) -> datetime | None:
    for frame in event.get("source_frames") or []:
        if not isinstance(frame, dict):
            continue
        match = _FRAME_KEY.search(str(frame.get("s3_key") or ""))
        if not match:
            continue
        chunk_start = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        chunk_start += timedelta(minutes=int(match.group(3)))
        event_start = float(event.get("start_time") or 0)
        return chunk_start + timedelta(seconds=event_start)
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


@lru_cache(maxsize=1)
def _reviewed_events() -> dict[str, dict[str, dict[str, Any]]]:
    """원본 영상과 대조한 시연 영상의 승인 이벤트만 읽는다."""
    path = Path(__file__).with_name("sequence_reviewed_events.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _sequence_intent(query: str) -> tuple[str, str]:
    """전체·첫/마지막·특정 행동의 앞/뒤 질문을 구분한다."""
    before = _BEFORE.search(query)
    after = _AFTER.search(query)
    if before and (not after or before.start() < after.start()):
        return "before", query[:before.start()].strip()
    if after:
        return "after", query[:after.start()].strip()
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
    "본", "봤어", "있었어", "있었던", "뭐", "뭘", "뭐야", "무엇", "무슨", "어떤", "어디", "언제", "어땠어", "어땠나",
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


def _variants(term: str) -> list[str]:
    """'마신'→'마시', '뛰어다닌'→'뛰어다니', '마시기'→'마시'처럼 활용 어미를 뗀 어간 후보를 함께 만든다. 한 글자 어간은 오매칭이 커서 만들지 않는다."""
    variants = [term]
    code = ord(term[-1]) - 0xAC00
    if 0 <= code < 11172 and code % 28 in (4, 8):
        stem = term[:-1] + chr(ord(term[-1]) - code % 28)
        if len(stem) >= 2:
            variants.append(stem)
    elif term[-1] in "기고" and len(term) >= 3:  # '마시기'·'마시고' → '마시'
        variants.append(term[:-1])
    return variants


def _term_in(term: str, text: str) -> bool:
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
        (r"나간|나갔|나가기|나왔|빠져나", r"나가|나왔|빠져나|통과"),
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
_UNCERTAIN_ACTION = re.compile(r"않|못|려고|려는|하려|것 같|듯")
MAX_MERGE_GAP_SECONDS = 120


def _action_category(event: dict[str, Any]) -> str | None:
    description = _description(event)
    if _UNCERTAIN_ACTION.search(description):
        return None
    return next((name for name, pattern in _ACTION_CATEGORIES if pattern.search(description)), None)


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
        category = _action_category(event)
        if groups and category is not None:
            last_recorded, last_event = groups[-1][-1]
            last_end = _event_end(last_recorded, last_event)
            same_video = last_event.get("video_id") == event.get("video_id")
            close = (
                last_end is not None
                and recorded_at is not None
                and (recorded_at - last_end).total_seconds() <= MAX_MERGE_GAP_SECONDS
            )
            if same_video and close and _action_category(last_event) == category:
                groups[-1].append(row)
                continue
        groups.append([row])
    return groups


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
    matched = {span for index in _match_indices(rows, terms) for span in spans if span[0] <= index <= span[1]}
    return next(iter(matched)) if len(matched) == 1 else None


def _playable(recorded_at: datetime | None, event: dict[str, Any]) -> bool:
    """시각을 알고 원본 프레임(s3_key)으로 다시 열 수 있는 이벤트인가."""
    return recorded_at is not None and any(
        isinstance(frame, dict) and frame.get("s3_key") for frame in event.get("source_frames") or []
    )


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
        "playback_start": max(timestamp - 3, 0),
        "playback_end": timestamp + 5,
    }
    if role:
        card["evidence_role"] = role
    return card


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
    if events is None:
        from db.behavior_events import get_behavior_events_for_aggregation
        events = get_behavior_events_for_aggregation(video_id=video_id)
    truncated = False

    rows = []
    excluded_event_count = 0
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
            rows.append((recorded_at, event))

    rows.sort(key=lambda row: (
        row[0] is None,
        row[0] or datetime.max,
        str(row[1].get("video_id") or ""),
        float(row[1].get("start_time") or 0),
        int(row[1].get("id") or 0),
    ))
    intent, anchor = _sequence_intent(query)
    all_rows = rows
    unplayable_count = 0
    if intent != "timeline":
        # 처음/마지막/앞/뒤 답은 시각을 알고 다시 재생할 수 있는 이벤트로만 한다.
        # 프레임이 없는 레거시 이벤트는 시각이 없어 맨 뒤로 정렬되므로 '마지막'이나 이웃으로 내면 안 된다.
        rows = [row for row in all_rows if _playable(*row)]
        unplayable_count = len(all_rows) - len(rows)
    selected_groups = None
    step_total = len(rows)
    anchor_span = None
    anchor_claim_unsupported = False
    condition_text = ""
    condition_status = None  # None | "missing"(조건에 맞는 장면 없음) | "unsupported"(장소만 맞고 행동은 확인 안 됨)
    condition_near = None
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
        selected_groups = groups[:top_k]
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
        direction = "다음" if intent == "after" else "이전"
        if selected:
            verb = "나와요" if intent == "after" else "있었어요"
            answer = f"‘{_spoken_description(anchor_event)}’ {direction}에는 ‘{_spoken_description(selected[0][1])}’ 장면이 {verb}."
        else:
            answer = f"‘{_spoken_description(anchor_event)}’ {direction}에는 이어지는 장면이 없어요." if intent == "after" else f"‘{_spoken_description(anchor_event)}’ {direction}에는 앞선 장면이 없어요."
        if selected and excluded_event_count:
            answer += " 빠진 구간이 있을 수 있어요."
        elif not selected and excluded_event_count:
            answer += " 다만 제외된 구간이 있어서, 실제로 다른 행동이 없었다는 뜻은 아니에요."
        if video_id in reviewed:
            answer += " 이 영상은 여러 장면을 이어 붙인 거라 같은 동물의 연속 행동인지는 알 수 없어요."
        answer += skipped_note
    elif not rows:
        if unplayable_count and not excluded_event_count:
            answer = "다시 재생할 수 있는 근거가 있는 행동 장면이 아직 없어요."
        else:
            answer = "선택하신 영상과 시간대에서는 보여드릴 수 있는 행동 장면이 없어요." if excluded_event_count else "선택하신 영상과 시간대에는 아직 분석된 행동이 없어요."
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
            sentences.append(sentence)
        answer = " ".join(sentences)
        if len(selected) < step_total:
            answer += f"\n전체 {step_total}개 장면 중 처음 {len(selected)}개만 보여드려요."

    return {
        "query": query,
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
        "sequence_intent": intent,
        "truncated": truncated,
    }
