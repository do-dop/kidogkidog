import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any

from pipeline.search.query_analyzer import build_prompt_hint
from db.behavior_events import (
    get_behavior_event_by_id,
    get_behavior_events_for_aggregation,
    get_top_behavior_events,
    get_behavior_events_overlapping,
)
from pipeline.search.aggregate_query import ACTION_PATTERNS, count_episodes, event_matches_action, parse_count_action
from pipeline.search.question_router import QuestionType, classify_with_rules
from pipeline.search.sequence_query import is_sequence_query, run_sequence_query

SEQUENCE_MIN_RESULTS = 10


DEFAULT_FALLBACK_MESSAGE = (
    "현재 OPENAI_API_KEY가 설정되어 있지 않아 LLM 답변은 생성하지 못했습니다. "
    "대신 분석된 행동 이벤트와 검색된 프레임 metadata를 기준으로 결과를 확인해주세요."
)

# 스테이징 영상으로 측정한 CLIP(ViT-L/14) 최상위 유사도: 관련 질문 0.249~0.285, 무관한 질문 0.161~0.214.
# 둘 사이의 중간값을 기본 기준으로 쓴다. 영상이나 모델이 바뀌면 SEARCH_MIN_SCORE로 조정한다.
DEFAULT_MIN_SEARCH_SCORE = 0.23

# 행동 이벤트가 질문과 맞는다고 보려면 _behavior_query_score의 키워드 그룹이 하나 이상 일치해야 한다.
# 0.3점짜리 단어 일치("장면" 같은 흔한 단어)만으로는 근거로 삼지 않는다.
_MIN_EVENT_KEYWORD_SCORE = 1.0


def _min_search_score() -> float:
    try:
        return float(os.getenv("SEARCH_MIN_SCORE", DEFAULT_MIN_SEARCH_SCORE))
    except ValueError:
        return DEFAULT_MIN_SEARCH_SCORE


def run_rag_query(
    query: str,
    video_id: str | None = None,
    top_k: int = 5,
    user_id: str | None = None,
    recording_date: str | None = None,
    time_range: dict[str, int] | None = None,
    source_event_id: int | None = None,
    event_start: float | None = None,
    event_end: float | None = None,
) -> dict:
    """
    사용자 질문을 받아 ChromaDB에서 관련 프레임을 검색하고,
    behavior_events와 검색 결과 metadata를 함께 사용해 답변한다.

    기존에는 검색된 프레임의 timestamp, object_labels, score만 근거로 답변했다.
    이제는 behavior_events를 우선 근거로 사용한다.

    즉:
    - 추천질문이 behavior_events에서 만들어졌다면
    - 답변도 behavior_events를 보고 할 수 있게 한다.
    """
    retrieved_frames = []

    source_event = _resolve_source_event(
        source_event_id=source_event_id,
        event_start=event_start,
        event_end=event_end,
        video_id=video_id,
    )

    if classify_with_rules(query).question_type == QuestionType.AGGREGATE:
        return _run_aggregate_query(
            query=query,
            video_id=video_id,
            top_k=top_k,
            recording_date=recording_date,
            time_range=time_range,
        )

    gate_scene_result = _run_gate_climb_scene_query(
        query=query,
        video_id=video_id or (source_event or {}).get("video_id"),
        top_k=top_k,
        recording_date=recording_date,
        time_range=time_range,
    )
    if gate_scene_result is not None:
        return gate_scene_result

    if source_event:
        return _run_event_grounded_query(
            query=query,
            source_event=source_event,
            top_k=top_k,
            user_id=user_id,
        )

    if is_sequence_query(query):
        # 순서 질문은 흐름이 끊기지 않게 요청한 개수와 상관없이 타임라인을 충분히 보여준다.
        return run_sequence_query(
            query=query,
            video_id=video_id,
            top_k=max(top_k, SEQUENCE_MIN_RESULTS),
            recording_date=recording_date,
            time_range=time_range,
        )

    search_succeeded = True
    candidate_count = 0

    try:
        from pipeline.storage.vector_store import search_with_query_expansion

        candidates = search_with_query_expansion(
            query=query,
            # Retrieve enough scene hits to offer different videos after the
            # API groups all hits from each video into one result card.
            top_k=top_k if video_id else min(30, max(12, top_k * 4)),
            video_id=video_id,
            recording_date=recording_date,
            time_range=time_range,
        )
        candidate_count = len(candidates)
        # CLIP은 무엇을 물어도 가장 가까운 프레임을 돌려준다. 점수가 낮으면 맞는 장면이 없다는 뜻이다.
        retrieved_frames = [
            frame for frame in candidates
            if float(frame.get("score") or 0.0) >= _min_search_score()
        ]
    except Exception as exc:
        # ChromaDB가 깨졌거나 인덱스 파일이 불안정할 때도
        # behavior_events 기반 답변은 가능하도록 앱 전체를 죽이지 않는다.
        print(f"ChromaDB 검색 실패: {exc}", flush=True)
        search_succeeded = False
        retrieved_frames = []

    behavior_events = _get_relevant_behavior_events(
        query=query,
        video_id=video_id,
        retrieved_frames=retrieved_frames,
        recording_date=recording_date,
        time_range=time_range,
        limit=5,
    )

    if search_succeeded and not retrieved_frames:
        # 관련 프레임이 없으면 질문과 단어가 맞는 이벤트만 근거로 쓴다.
        # 그렇지 않으면 "주요 행동 이벤트"가 무관한 질문에도 결과로 채워진다.
        behavior_events = [
            event for event in behavior_events
            if _behavior_query_score(query, event) >= _MIN_EVENT_KEYWORD_SCORE
        ]

    if not retrieved_frames and behavior_events:
        retrieved_frames = _frames_from_behavior_events(
            behavior_events=behavior_events,
            video_id=video_id,
            limit=top_k,
        )

    if not search_succeeded and not retrieved_frames and not behavior_events:
        # 검색 자체가 실패했을 때만 인덱스에 있는 프레임을 대신 보여준다.
        # 검색이 정상이었는데 맞는 장면이 없으면 임의의 프레임을 내놓지 않는다.
        retrieved_frames = _fallback_indexed_frames(
            video_id=video_id,
            limit=top_k,
            recording_date=recording_date,
            time_range=time_range,
        )

    if not retrieved_frames and not behavior_events:
        if _has_datetime_filter(recording_date, time_range):
            answer = (
                f"선택한 날짜와 시간대에서는 '{query}'과 맞는 장면을 찾지 못했어요. "
                "다른 날짜나 시간대를 선택해서 다시 검색해보세요."
            )
        elif candidate_count:
            answer = (
                f"'{query}'과 맞는 장면을 찾지 못했어요. "
                "다른 표현으로 물어보시거나 추천 질문을 눌러 보세요."
            )
        else:
            answer = (
                f"질문 '{query}'과 정확히 맞는 장면은 아직 찾지 못했습니다. "
                "다만 이 영상의 인덱스가 충분하지 않거나 분석 결과가 아직 생성 중일 수 있어요."
            )

        return {
            "query": query,
            "video_id": video_id,
            "answer": answer,
            "results": [],
            "evidence_items": [],
            "behavior_events": [],
            "used_llm": False,
        }

    retrieved_context = _build_retrieved_context(retrieved_frames)
    behavior_context = _build_behavior_event_context(behavior_events)
    prompt_hint = build_prompt_hint(user_id=user_id, video_id=video_id)

    answer = _generate_answer_with_langchain(
        query=query,
        retrieved_context=retrieved_context,
        behavior_context=behavior_context,
        prompt_hint=prompt_hint,
    )

    used_llm = answer != DEFAULT_FALLBACK_MESSAGE

    if not used_llm:
        answer = _build_fallback_answer(
            query=query,
            retrieved_frames=retrieved_frames,
            behavior_events=behavior_events,
        )

    return {
        "query": query,
        "video_id": video_id,
        "answer": answer,
        "results": retrieved_frames,
        "evidence_items": retrieved_frames,
        "behavior_events": behavior_events,
        "used_llm": used_llm,
    }


def _run_gate_climb_scene_query(
    query: str,
    video_id: str | None,
    top_k: int,
    recording_date: str | None,
    time_range: dict[str, int] | None,
) -> dict[str, Any] | None:
    """Use extracted behavior records for a gate-climbing scene question.

    CLIP object labels can call a small dog a cat. They must not override a
    matching behavior record or support a claim that the scene is absent.
    """
    if (
        not video_id
        or _has_datetime_filter(recording_date, time_range)
        or not re.search(ACTION_PATTERNS["gate_climb"][0], query, re.I)
    ):
        return None

    try:
        timeline = get_behavior_events_for_aggregation(video_id=video_id)
    except Exception as exc:
        print(f"울타리 행동 이벤트 조회 실패: {exc}", flush=True)
        return {
            "query": query,
            "video_id": video_id,
            "answer": "행동 기록을 조회하지 못해 울타리 장면을 확인할 수 없어요. 잠시 후 다시 시도해 주세요.",
            "results": [],
            "evidence_items": [],
            "behavior_events": [],
            "used_llm": False,
        }

    matched = [event for event in timeline if event_matches_action(event, "gate_climb")]
    episodes = count_episodes(matched)
    if not episodes:
        return {
            "query": query,
            "video_id": video_id,
            "answer": "울타리 올라타기를 확인할 행동 기록이 아직 없어요. 원본 영상을 확인해 주세요.",
            "results": [],
            "evidence_items": [],
            "behavior_events": [],
            "used_llm": False,
        }

    evidence = []
    for episode in episodes[:max(1, top_k)]:
        event = episode["events"][0]
        frames = _frames_from_behavior_events([event], video_id=video_id, limit=1)
        if frames:
            frame = _attach_event_window_to_frames(frames, event)[0]
            # YOLO labels are independent frame guesses, not proof of the action.
            frame["object_labels"] = ""
            evidence.append(frame)

    times = ", ".join(
        f"{episode['start_time']:.1f}~{episode['end_time']:.1f}초"
        for episode in episodes[:max(1, top_k)]
    )
    answer = (
        f"울타리를 올라가거나 넘는 것으로 분석된 장면이 {times}에 있어요. "
        "실제 동작은 영상을 재생해 확인해 주세요."
    )
    return {
        "query": query,
        "video_id": video_id,
        "answer": answer,
        "results": evidence,
        "evidence_items": evidence,
        "behavior_events": matched,
        "used_llm": False,
    }


def _run_aggregate_query(
    query: str,
    video_id: str | None,
    top_k: int,
    recording_date: str | None,
    time_range: dict[str, int] | None,
) -> dict[str, Any]:
    parsed = parse_count_action(query)
    if parsed is None:
        answer = "이 질문의 행동이나 계산 단위를 아직 정확히 셀 수 없어요. '물 몇 번 마셨어?'처럼 한 가지 행동의 횟수를 물어봐 주세요."
        return _empty_aggregate_result(query, video_id, answer)

    action, action_label = parsed
    if not recording_date:
        today = datetime.now(ZoneInfo("Asia/Seoul")).date()
        if "어제" in query:
            recording_date = (today - timedelta(days=1)).isoformat()
        elif "오늘" in query:
            recording_date = today.isoformat()

    try:
        timeline = get_behavior_events_for_aggregation(video_id=video_id)
    except Exception as exc:
        print(f"aggregate 행동 이벤트 조회 실패: {exc}", flush=True)
        return _empty_aggregate_result(query, video_id, "행동 기록을 조회하지 못해 횟수를 계산할 수 없어요.")

    matched = [event for event in timeline if event_matches_action(event, action)]
    if _has_datetime_filter(recording_date, time_range):
        # A missing capture time means the event may belong inside the requested period.
        # An incomplete timeline must not produce an exact count.
        if any(not any(
            _frame_recorded_at_from_key(frame.get("s3_key"))
            for frame in (event.get("source_frames") or [])
            if isinstance(frame, dict)
        ) for event in matched):
            return _empty_aggregate_result(
                query, video_id,
                "일부 행동의 촬영 시각을 확인할 수 없어 해당 기간의 횟수를 정확히 계산할 수 없어요.",
            )
        matched = _filter_events_by_datetime(matched, recording_date, time_range)

    episodes = count_episodes(matched)
    if not episodes:
        return _empty_aggregate_result(
            query, video_id,
            f"{action_label} 행동 구간이 기록에서 분리되지 않아 횟수를 계산할 수 없어요. 원본 영상을 확인해 주세요.",
        )

    evidence = []
    result_frames = []
    for episode in episodes:
        evidence.append({
            "video_id": episode["video_id"],
            "start_time": episode["start_time"],
            "end_time": episode["end_time"],
            "event_ids": episode["event_ids"],
        })
        if len(result_frames) < top_k:
            frames = _frames_from_behavior_events(episode["events"], episode["video_id"], limit=1)
            if frames:
                frame = frames[0]
                frame["event_start"] = episode["start_time"]
                frame["event_end"] = episode["end_time"]
                frame["source_event_id"] = episode["event_ids"][0]
                result_frames.append(frame)

    times = ", ".join(
        f"{item['video_id']} {item['start_time']:.1f}~{item['end_time']:.1f}초"
        for item in evidence[:3]
    )
    more = f" 외 {len(evidence) - 3}개 구간" if len(evidence) > 3 else ""
    answer = (
        f"기록에서 {action_label}로 표시된 구간 {len(evidence)}개를 찾았어요. "
        f"근거 구간: {times}{more}. 실제 행동 횟수는 영상을 확인해 주세요."
    )
    return {
        "query": query,
        "video_id": video_id,
        "answer": answer,
        "results": result_frames,
        "evidence_items": result_frames,
        "behavior_events": matched,
        "used_llm": False,
        "aggregate_result": {
            "action": action,
            "metric": "episode_count",
            "count": len(evidence),
            "count_basis": "detected_behavior_intervals",
            "recording_date": recording_date,
            "time_range": time_range,
            "intervals": evidence,
        },
    }


def _empty_aggregate_result(query: str, video_id: str | None, answer: str) -> dict[str, Any]:
    return {
        "query": query,
        "video_id": video_id,
        "answer": answer,
        "results": [],
        "evidence_items": [],
        "behavior_events": [],
        "used_llm": False,
        "aggregate_result": None,
    }


def _resolve_source_event(
    source_event_id: int | None,
    event_start: float | None,
    event_end: float | None,
    video_id: str | None,
) -> dict[str, Any] | None:
    if source_event_id is not None:
        try:
            event = get_behavior_event_by_id(source_event_id)
            if event:
                return event
        except Exception as exc:
            print(f"source_event_id 조회 실패: {exc}", flush=True)

    if event_start is None or event_end is None:
        return None

    return {
        "id": source_event_id,
        "video_id": video_id,
        "start_time": event_start,
        "end_time": event_end,
        "summary": "추천 질문과 연결된 행동 구간",
        "source_frames": [],
    }


def _run_event_grounded_query(
    query: str,
    source_event: dict[str, Any],
    top_k: int,
    user_id: str | None,
) -> dict:
    event_video_id = source_event.get("video_id")
    behavior_events = [source_event]
    retrieved_frames = []

    try:
        from pipeline.storage.vector_store import search_with_query_expansion

        candidate_frames = search_with_query_expansion(
            query=query,
            top_k=max(top_k * 5, 20),
            video_id=event_video_id,
        )
        retrieved_frames = _filter_frames_to_event_window(
            frames=candidate_frames,
            event=source_event,
            margin_seconds=3.0,
        )[:top_k]
    except Exception as exc:
        print(f"event 기반 CLIP 검색 실패: {exc}", flush=True)
        retrieved_frames = []

    if not retrieved_frames:
        retrieved_frames = _frames_from_behavior_events(
            behavior_events=behavior_events,
            video_id=event_video_id,
            limit=top_k,
        )

    retrieved_frames = _attach_event_window_to_frames(retrieved_frames, source_event)
    retrieved_context = _build_retrieved_context(retrieved_frames)
    behavior_context = _build_behavior_event_context(behavior_events)
    prompt_hint = build_prompt_hint(user_id=user_id, video_id=event_video_id)

    answer = _generate_answer_with_langchain(
        query=query,
        retrieved_context=retrieved_context,
        behavior_context=behavior_context,
        prompt_hint=prompt_hint,
    )

    used_llm = answer != DEFAULT_FALLBACK_MESSAGE

    if not used_llm:
        answer = _build_fallback_answer(
            query=query,
            retrieved_frames=retrieved_frames,
            behavior_events=behavior_events,
        )

    return {
        "query": query,
        "video_id": event_video_id,
        "answer": answer,
        "results": retrieved_frames,
        "evidence_items": retrieved_frames,
        "behavior_events": behavior_events,
        "used_llm": used_llm,
    }


def _get_relevant_behavior_events(
    query: str,
    video_id: str | None,
    retrieved_frames: list[dict[str, Any]],
    recording_date: str | None = None,
    time_range: dict[str, int] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    사용자 질문과 관련된 behavior_events를 가져온다.

    우선순위:
    1. 검색된 프레임 timestamp와 겹치는 행동 이벤트
    2. 영상 전체에서 중요한 행동 이벤트
    3. 질문 텍스트와 action/summary/evidence가 비슷한 행동 이벤트
    """
    candidates = []

    # 1. 검색된 프레임 주변의 behavior_events 조회
    for frame in retrieved_frames:
        frame_video_id = frame.get("video_id") or video_id
        timestamp = frame.get("timestamp")

        if frame_video_id is None or timestamp is None:
            continue

        try:
            timestamp = float(timestamp)

            scene_start = _safe_float(frame.get("scene_start"))
            scene_end = _safe_float(frame.get("scene_end"))
            overlapping_events = get_behavior_events_overlapping(
                video_id=frame_video_id,
                start_time=max((scene_start if scene_start is not None else timestamp) - 3.0, 0.0),
                end_time=(scene_end if scene_end is not None else timestamp) + 3.0,
                limit=5,
            )

            candidates.extend(_filter_events_by_datetime(overlapping_events, recording_date, time_range))

        except Exception as exc:
            print(f"겹치는 행동 이벤트 조회 실패: {exc}", flush=True)

    # 2. 영상 전체 주요 behavior_events 조회
    # 검색된 프레임과 겹치는 이벤트가 이미 있으면 그것만 근거로 쓴다.
    # 화면에 보이는 장면과 무관한 영상 전체 이벤트를 섞으면 답변의 시간대가 카드와 어긋난다.
    # 날짜/시간대 필터는 조회 후에 적용되므로, 필터가 있으면 후보를 넉넉히 가져온 뒤 거른다.
    if not candidates:
        try:
            has_filter = _has_datetime_filter(recording_date, time_range)
            top_events = get_top_behavior_events(
                video_id=video_id,
                limit=200 if has_filter else 10,
            )
            candidates.extend(_filter_events_by_datetime(top_events, recording_date, time_range)[:10])

        except Exception as exc:
            print(f"주요 행동 이벤트 조회 실패: {exc}", flush=True)

    candidates = _deduplicate_behavior_events(candidates)

    if not candidates:
        return []

    # 3. 질문과 관련성이 높은 순서로 정렬
    candidates = sorted(
        candidates,
        key=lambda event: (
            _behavior_query_score(query, event),
            float(event.get("interestingness") or 0.0),
            float(event.get("confidence") or 0.0),
            float(event.get("duration") or 0.0),
        ),
        reverse=True,
    )

    return candidates[:limit]


def _frame_recorded_at_from_key(s3_key: str | None):
    if not s3_key:
        return None

    match = re.search(
        r"petcam_(\d{8})_(\d{6})_(\d{3})_frame_([0-9.]+)\.jpg$",
        s3_key,
    )

    if not match:
        return None

    base = datetime.strptime(
        f"{match.group(1)}{match.group(2)}",
        "%Y%m%d%H%M%S",
    )
    return base + timedelta(seconds=int(match.group(3)) * 60 + float(match.group(4)))


def _has_datetime_filter(recording_date: str | None, time_range: dict[str, int] | None) -> bool:
    if recording_date:
        return True

    if not time_range:
        return False

    try:
        start_hour = int(time_range.get("start_hour"))
        end_hour = int(time_range.get("end_hour"))
    except (TypeError, ValueError):
        return False

    return not (start_hour <= 0 and end_hour >= 24)


def _matches_datetime_filter(recorded_at, recording_date: str | None, time_range: dict[str, int] | None):
    if recording_date:
        if not recorded_at or recorded_at.date().isoformat() != recording_date:
            return False

    if not time_range:
        return True

    start_hour = time_range.get("start_hour")
    end_hour = time_range.get("end_hour")

    if start_hour is None or end_hour is None:
        return True

    try:
        start_hour = int(start_hour)
        end_hour = int(end_hour)
    except (TypeError, ValueError):
        return True

    if start_hour <= 0 and end_hour >= 24:
        return True

    if not recorded_at:
        return False

    event_hour = recorded_at.hour + recorded_at.minute / 60 + recorded_at.second / 3600

    if start_hour <= end_hour:
        return start_hour <= event_hour < end_hour

    return event_hour >= start_hour or event_hour < end_hour


def _filter_events_by_datetime(
    events: list[dict[str, Any]],
    recording_date: str | None,
    time_range: dict[str, int] | None,
):
    if not recording_date and not time_range:
        return events

    filtered_events = []

    for event in events:
        source_frames = event.get("source_frames") or []
        recorded_times = [
            _frame_recorded_at_from_key(frame.get("s3_key"))
            for frame in source_frames
            if isinstance(frame, dict)
        ]

        if any(_matches_datetime_filter(recorded_at, recording_date, time_range) for recorded_at in recorded_times):
            filtered_events.append(event)

    return filtered_events


def _frames_from_behavior_events(
    behavior_events: list[dict[str, Any]],
    video_id: str | None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    frames = []

    for event in behavior_events:
        event_video_id = event.get("video_id") or video_id or "default"
        source_frames = event.get("source_frames") or []

        if not isinstance(source_frames, list):
            continue

        for frame in source_frames:
            if not isinstance(frame, dict):
                continue

            timestamp = frame.get("timestamp")
            if timestamp is None:
                timestamp = event.get("start_time")

            frames.append({
                "frame_id": frame.get("frame_id") or f"event-{event.get('id')}-{timestamp}",
                "frame_path": frame.get("frame_path"),
                "s3_key": frame.get("s3_key"),
                "timestamp": timestamp,
                "video_id": event_video_id,
                "object_labels": _normalize_labels(frame.get("object_labels")),
                "score": float(event.get("confidence") or event.get("interestingness") or 0.5),
            })

            if len(frames) >= limit:
                return frames

    return frames[:limit]


def _filter_frames_to_event_window(
    frames: list[dict[str, Any]],
    event: dict[str, Any],
    margin_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    start_time = _safe_float(event.get("start_time"))
    end_time = _safe_float(event.get("end_time"))

    if start_time is None or end_time is None:
        return frames

    lower_bound = max(start_time - margin_seconds, 0.0)
    upper_bound = end_time + margin_seconds

    return [
        frame
        for frame in frames
        if (timestamp := _safe_float(frame.get("timestamp"))) is not None
        and lower_bound <= timestamp <= upper_bound
    ]


def _attach_event_window_to_frames(
    frames: list[dict[str, Any]],
    event: dict[str, Any],
) -> list[dict[str, Any]]:
    start_time = _safe_float(event.get("start_time"))
    end_time = _safe_float(event.get("end_time"))
    output = []

    for frame in frames:
        timestamp = _safe_float(frame.get("timestamp"))
        item = dict(frame)
        item["source_event_id"] = event.get("id")
        item["event_start"] = start_time
        item["event_end"] = end_time

        representative_timestamp = timestamp if timestamp is not None else start_time
        item["representative_timestamp"] = representative_timestamp

        if representative_timestamp is not None:
            item["playback_start"] = max(representative_timestamp - 3.0, 0.0)
            item["playback_end"] = representative_timestamp + 5.0
        else:
            item["playback_start"] = start_time
            item["playback_end"] = end_time

        output.append(item)

    return output


def _fallback_indexed_frames(
    video_id: str | None,
    limit: int = 5,
    recording_date: str | None = None,
    time_range: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    try:
        from pipeline.storage.vector_store import get_indexed_frames

        frames = get_indexed_frames(
            video_id=video_id,
            recording_date=recording_date,
            time_range=time_range,
        )
    except Exception as exc:
        print(f"대체 프레임 조회 실패: {exc}", flush=True)
        return []

    fallback_frames = []

    for index, frame in enumerate(frames[:limit]):
        fallback_frames.append({
            "frame_id": frame.get("frame_id") or f"fallback-{index}",
            "frame_path": frame.get("frame_path"),
            "s3_key": frame.get("s3_key"),
            "timestamp": frame.get("timestamp"),
            "video_id": frame.get("video_id") or video_id or "default",
            "object_labels": frame.get("object_labels", ""),
            "score": 0.0,
        })

    # 날짜/시간대를 골랐는데 맞는 프레임이 없으면 다른 날짜의 0초 프레임을 보여주지 않는다.
    if not fallback_frames and video_id and not _has_datetime_filter(recording_date, time_range):
        fallback_frames.append({
            "frame_id": f"fallback-{video_id}-0",
            "frame_path": None,
            "s3_key": None,
            "timestamp": 0.0,
            "video_id": video_id,
            "object_labels": "",
            "score": 0.0,
        })

    return fallback_frames


def _normalize_labels(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# 한 글자 키워드("물", "공")가 다른 단어 안에서 우연히 일치하지 않도록 먼저 지운다.
# 예: "반려동물"의 "물"이 모든 이벤트를 물 마시기 질문과 일치시켰다.
_SUBSTRING_TRAPS = ("동물", "공간", "공원", "공격")


def _strip_substring_traps(text: str) -> str:
    for trap in _SUBSTRING_TRAPS:
        text = text.replace(trap, "")
    return text


def _behavior_query_score(query: str, event: dict[str, Any]) -> float:
    """
    질문 텍스트와 행동 이벤트가 얼마나 관련 있는지 간단히 점수화한다.

    복잡한 임베딩 검색이 아니라, 현재는 action/summary/evidence의 키워드 포함 여부로 판단한다.
    """
    if not query:
        return 0.0

    query_text = _strip_substring_traps(query.lower().replace(" ", ""))

    event_text_parts = [
        event.get("subject"),
        event.get("action"),
        event.get("target_object"),
        event.get("summary"),
    ]

    evidence = event.get("evidence") or []

    if isinstance(evidence, list):
        event_text_parts.extend(str(item) for item in evidence)
    else:
        event_text_parts.append(str(evidence))

    event_text = " ".join(str(item or "") for item in event_text_parts)
    event_text = _strip_substring_traps(event_text.lower().replace(" ", ""))

    score = 0.0

    keyword_groups = [
        ["횟수", "몇번", "몇회", "반복"],
        ["그릇", "사료", "밥", "먹", "물", "마시", "bowl", "cup", "bottle"],
        ["울타리", "펜스", "gate", "fence"],
        ["올라", "올랐", "타고", "넘", "기어", "climb"],
        ["공", "스포츠공", "sportsball", "ball"],
        ["불안", "서성", "돌아다니", "배회"],
        ["사람", "person"],
        ["강아지", "dog"],
        ["고양이", "cat"],
    ]

    for group in keyword_groups:
        query_has_keyword = any(keyword in query_text for keyword in group)
        event_has_keyword = any(keyword in event_text for keyword in group)

        if query_has_keyword and event_has_keyword:
            score += 1.0

    # 질문에 들어간 긴 단어가 이벤트 텍스트에 포함되면 가산점
    rough_tokens = [
        token.strip()
        for token in query.replace("?", " ").replace(",", " ").split()
        if len(token.strip()) >= 2
    ]

    for token in rough_tokens:
        normalized_token = token.lower().replace(" ", "")
        if normalized_token and normalized_token in event_text:
            score += 0.3

    return score


def _build_behavior_event_context(behavior_events: list[dict[str, Any]]) -> str:
    """
    behavior_events를 LLM에게 전달하기 좋은 텍스트로 변환한다.
    """
    if not behavior_events:
        return "분석된 행동 이벤트 없음"

    lines = []

    for idx, event in enumerate(behavior_events, start=1):
        start_time = event.get("start_time")
        end_time = event.get("end_time")

        if start_time is not None and end_time is not None:
            time_text = f"{float(start_time):.2f}초~{float(end_time):.2f}초"
        else:
            time_text = "시간 정보 없음"

        evidence = event.get("evidence") or []
        if isinstance(evidence, list):
            evidence_text = " / ".join(str(item) for item in evidence[:5]) or "근거 없음"
        else:
            evidence_text = str(evidence)

        source_frames = event.get("source_frames") or []

        frame_text_items = []
        for frame in source_frames[:5]:
            timestamp = frame.get("timestamp")
            timestamp_text = f"{float(timestamp):.2f}초" if timestamp is not None else "시간 없음"
            labels = frame.get("object_labels") or []
            frame_text_items.append(f"{timestamp_text}: {labels}")

        source_frame_text = " / ".join(frame_text_items) or "대표 프레임 정보 없음"

        lines.append(
            f"{idx}. behavior_event_id: {event.get('id')}\n"
            f"   time_range: {time_text}\n"
            f"   subject: {event.get('subject') or 'unknown'}\n"
            f"   action: {event.get('action') or 'unknown'}\n"
            f"   target_object: {event.get('target_object') or '없음'}\n"
            f"   summary: {event.get('summary') or '없음'}\n"
            f"   duration: {event.get('duration') or 0}초\n"
            f"   repeat_count: {event.get('repeat_count') or 0}\n"
            f"   confidence: {event.get('confidence') or 0}\n"
            f"   interestingness: {event.get('interestingness') or 0}\n"
            f"   evidence: {evidence_text}\n"
            f"   source_frames: {source_frame_text}"
        )

    return "\n".join(lines)


def _format_clock(value) -> str:
    seconds = max(0, int(round(float(value))))
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _format_scene_range(frame: dict[str, Any]) -> str:
    """결과 카드에 표시되는 '재생 구간'과 같은 MM:SS 형식으로 장면 구간을 만든다."""
    start = _safe_float(frame.get("playback_start"))
    end = _safe_float(frame.get("playback_end"))
    representative = _safe_float(frame.get("representative_timestamp"))
    if representative is None:
        representative = _safe_float(frame.get("timestamp"))

    if start is None and representative is not None:
        start = max(representative - 3.0, 0.0)
    if end is None and representative is not None:
        end = representative + 5.0
    if start is None or end is None:
        return "알 수 없음"

    text = f"{_format_clock(start)}~{_format_clock(end)}"
    if representative is not None:
        text += f" (가장 비슷한 순간 {_format_clock(representative)})"
    return text


def _build_retrieved_context(retrieved_frames: list[dict[str, Any]]) -> str:
    """
    검색된 프레임 metadata를 LLM에게 전달하기 좋은 텍스트로 변환한다.
    """
    if not retrieved_frames:
        return "검색된 프레임 없음"

    lines = []

    for idx, frame in enumerate(retrieved_frames, start=1):
        timestamp = frame.get("timestamp")
        timestamp_text = f"{float(timestamp):.2f}초" if timestamp is not None else "알 수 없음"

        object_labels = frame.get("object_labels") or "감지 객체 없음"
        score = frame.get("score")
        score_text = f"{float(score):.4f}" if score is not None else "알 수 없음"

        scene_text = _format_scene_range(frame)

        lines.append(
            f"{idx}. video_id: {frame.get('video_id', 'unknown')}\n"
            f"   frame_id: {frame.get('frame_id', 'unknown')}\n"
            f"   화면에 표시되는 재생 구간: {scene_text}\n"
            f"   timestamp: {timestamp_text}\n"
            f"   object_labels: {object_labels}\n"
            f"   similarity_score: {score_text}\n"
            f"   s3_key: {frame.get('s3_key') or '없음'}"
        )

    return "\n".join(lines)


def _generate_answer_with_langchain(
    query: str,
    retrieved_context: str,
    behavior_context: str,
    prompt_hint: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    LangChain을 사용해서 behavior_events + 검색 결과 기반 답변을 생성한다.
    OPENAI_API_KEY가 없거나 호출 실패 시 fallback 메시지를 반환한다.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return DEFAULT_FALLBACK_MESSAGE

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
너는 반려동물 영상 검색 서비스의 답변 도우미다.

너에게는 두 종류의 근거가 주어진다.

1. 분석된 행동 이벤트 behavior_events
- 영상에서 미리 분석된 반려동물 행동 후보이다.
- action, summary, repeat_count, duration, evidence를 포함한다.
- 추천질문은 이 behavior_events를 바탕으로 생성되었을 수 있다.
- 따라서 답변할 때 behavior_events를 우선 근거로 사용한다.

2. 검색된 프레임 metadata
- ChromaDB 검색으로 찾은 관련 프레임이다.
- timestamp, object_labels, similarity_score를 포함한다.
- 행동 이벤트를 보조하는 근거로 사용한다.

답변 규칙:
- 사용자는 내부 분석 로그가 아니라 "어느 장면을 보면 되는지"를 알고 싶어 한다.
- 첫 문장은 질문에 직접 답한다. 예: "고양이가 보이는 장면은 00:02 근처에서 확인할 수 있어요."
- 시간은 반드시 "검색된 프레임 근거"의 "화면에 표시되는 재생 구간"(MM:SS)만 사용한다. 사용자 화면의 결과 카드와 같은 구간이어야 한다.
- 행동 이벤트의 time_range(영상 전체 길이에 걸친 넓은 구간)를 그대로 말하지 않는다. 2초~58초처럼 넓은 구간을 안내하지 않는다.
- 질문에 나온 대상이나 행동(예: 쓰레기통, 뒤지기)이 summary, evidence, target_object, object_labels 어디에도 직접 나오지 않으면, 있다고 말하지 않는다. 이때는 "정확히 그 장면이라고 확신하긴 어렵지만 가장 비슷한 후보는 ~예요"처럼 후보라고 말한다.
- 근거에 없는 해석(호기심, 감정, 의도 등)을 덧붙이지 않는다. 화면에 보이는 것만 말한다.
- 후보 장면이 여러 개면 비슷한 순서대로 구간을 짧게 나열한다.
- 중복된 구간은 반복해서 나열하지 말고 하나로 묶는다.
- confidence, similarity_score, repeat_count, frame count, object_labels 원문, evidence 원문 같은 내부 수치는 노출하지 않는다.
- "분석된 행동 이벤트", "프레임 기반", "metadata", "ChromaDB" 같은 기술 용어를 사용자 답변에 쓰지 않는다.
- 행동을 과하게 단정하지 않는다. "~로 보여요", "~장면으로 확인해보면 좋아요"처럼 자연스럽게 말한다.
- 답변은 2~4문장 또는 최대 3개 bullet로 짧게 작성한다.
- 마지막에는 필요할 때만 "정확한 행동 판단은 영상으로 한 번 확인해주세요."처럼 부드럽게 덧붙인다.
- 한국어로 답변한다.
""".strip(),
            ),
            (
                "human",
                """
사용자 질문:
{query}

사용자/영상 맥락:
{prompt_hint}

분석된 행동 이벤트:
{behavior_context}

검색된 프레임 근거:
{retrieved_context}

위 근거를 사용해서 답변해줘.
""".strip(),
            ),
        ])

        llm = ChatOpenAI(
            model=model,
            temperature=0,
        )

        chain = prompt | llm | StrOutputParser()

        return chain.invoke({
            "query": query,
            "prompt_hint": prompt_hint,
            "behavior_context": behavior_context,
            "retrieved_context": retrieved_context,
        }).strip()

    except Exception as exc:
        print(f"LangChain RAG 답변 생성 실패: {exc}", flush=True)
        return DEFAULT_FALLBACK_MESSAGE


def _build_fallback_answer(
    query: str,
    retrieved_frames: list[dict[str, Any]],
    behavior_events: list[dict[str, Any]],
) -> str:
    """
    LLM을 못 쓸 때도 behavior_events와 검색 결과 기반으로 간단한 답변을 생성한다.
    """
    if retrieved_frames:
        # 카드에 보이는 장면과 같은 구간·순서로 안내한다. 이벤트는 장면 설명에만 쓴다.
        top_items = []
        for frame in retrieved_frames[:3]:
            timestamp = _safe_float(frame.get("timestamp"))
            matching_event = next(
                (
                    candidate for candidate in behavior_events
                    if timestamp is not None
                    and _safe_float(candidate.get("start_time")) is not None
                    and _safe_float(candidate.get("end_time")) is not None
                    and _safe_float(candidate["start_time"]) - 3.0 <= timestamp <= _safe_float(candidate["end_time"]) + 3.0
                ),
                None,
            )
            description = (
                _friendly_event_summary(matching_event) if matching_event
                else _friendly_labels(frame.get("object_labels"))
            )
            top_items.append(f"- {_format_scene_range(frame)}: {description}")

        joined = "\n".join(top_items)
        return (
            f"'{query}'와 가장 비슷한 장면 순서로 보여드려요. "
            "질문 속 대상이 정확히 나오는지는 자동으로 확정하기 어려워요.\n\n"
            f"{joined}\n\n"
            "결과 영상을 눌러 해당 구간을 직접 확인해보세요."
        )

    if behavior_events:
        visible_events = _deduplicate_answer_events(behavior_events)[:3]
        top_items = []

        for event in visible_events:
            start_time = event.get("start_time")
            end_time = event.get("end_time")
            time_text = _format_user_time_range(start_time, end_time)
            summary = _friendly_event_summary(event)

            top_items.append(
                f"- {time_text}: {summary}"
            )

        joined = "\n".join(top_items)
        intro = _friendly_intro(query=query, count=len(top_items))

        return f"{intro}\n\n{joined}\n\n정확한 행동은 영상을 재생해서 한 번 확인해보면 좋아요."

    if retrieved_frames:
        top_items = []
        has_strong_match = any(float(frame.get("score") or 0.0) > 0 for frame in retrieved_frames)

        for frame in retrieved_frames[:3]:
            timestamp = frame.get("timestamp")
            timestamp_text = f"{float(timestamp):.2f}초" if timestamp is not None else "알 수 없는 시간"
            labels = _friendly_labels(frame.get("object_labels"))

            top_items.append(
                f"- {timestamp_text}: {labels}"
            )

        joined = "\n".join(top_items)

        if not has_strong_match:
            return (
                f"'{query}'와 완전히 딱 맞는 장면은 아직 뚜렷하지 않지만, 가까운 후보는 있어요.\n\n"
                f"{joined}\n\n"
                "검색 결과 화면에서 해당 구간을 확인해보세요."
            )

        return (
            f"'{query}'와 관련 있어 보이는 장면을 찾았어요.\n\n"
            f"{joined}\n\n"
            "필요하면 결과 영상을 눌러 바로 확인할 수 있어요."
        )

    return (
        f"'{query}'와 잘 맞는 장면을 아직 찾지 못했어요. "
        "다른 표현으로 검색하거나, 영상 분석이 끝난 뒤 다시 시도해보세요."
    )


def _deduplicate_answer_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []

    for event in events:
        key = (
            event.get("video_id"),
            round(float(event.get("start_time") or 0.0), 1),
            round(float(event.get("end_time") or 0.0), 1),
            _friendly_event_summary(event),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(event)

    return output


def _format_user_time_range(start_time, end_time) -> str:
    if start_time is None:
        return "확인 가능한 구간"

    start_text = _format_user_seconds(start_time)

    if end_time is None:
        return f"{start_text} 근처"

    duration = max(0.0, float(end_time) - float(start_time))

    if duration < 8:
        return f"{start_text} 근처"

    return f"{start_text}부터 약 {_format_duration(duration)}"


def _format_user_seconds(value) -> str:
    seconds = max(0, int(round(float(value))))
    minutes, remaining_seconds = divmod(seconds, 60)

    if minutes <= 0:
        return f"{remaining_seconds}초"

    return f"{minutes}분 {remaining_seconds:02d}초"


def _format_duration(seconds: float) -> str:
    rounded = max(1, int(round(seconds)))
    minutes, remaining_seconds = divmod(rounded, 60)

    if minutes <= 0:
        return f"{remaining_seconds}초"

    if remaining_seconds == 0:
        return f"{minutes}분"

    return f"{minutes}분 {remaining_seconds}초"


def _friendly_intro(query: str, count: int) -> str:
    if count <= 1:
        return f"'{query}'와 관련 있어 보이는 장면을 찾았어요."

    return f"'{query}'와 관련 있어 보이는 장면 {count}개를 찾았어요."


def _friendly_event_summary(event: dict[str, Any]) -> str:
    summary = str(event.get("summary") or "").strip()
    action = str(event.get("action") or "").strip().lower()
    target = str(event.get("target_object") or "").strip()
    labels = _labels_from_event(event)

    if target:
        return f"{target} 근처에서 움직임이 보여요."

    if "cat" in labels or "고양이" in summary:
        return "고양이가 화면에 보이는 구간이에요."

    if "dog" in labels or "강아지" in summary or "반려동물" in summary:
        return "반려동물이 움직이는 모습이 보여요."

    if action and action not in {"motion", "unknown"}:
        return f"{action} 행동으로 보이는 장면이에요."

    if summary:
        return _soften_summary(summary)

    return "움직임이 확인된 장면이에요."


def _labels_from_event(event: dict[str, Any]) -> set[str]:
    labels = set()

    for frame in event.get("source_frames") or []:
        if not isinstance(frame, dict):
            continue
        raw_labels = frame.get("object_labels") or []
        if isinstance(raw_labels, str):
            candidates = raw_labels.split(",")
        else:
            candidates = raw_labels

        for label in candidates:
            normalized = str(label).strip().lower()
            if normalized:
                labels.add(normalized)

    return labels


def _soften_summary(summary: str) -> str:
    cleaned = re.sub(r"\s+", " ", summary).strip()
    cleaned = re.sub(r"\([^)]*(반복|신뢰도|confidence|frame|프레임)[^)]*\)", "", cleaned).strip()
    cleaned = cleaned.replace("것으로 추정되는 구간입니다", "것으로 보여요")
    cleaned = cleaned.replace("구간입니다", "장면이에요")

    if len(cleaned) > 70:
        cleaned = cleaned[:67].rstrip() + "..."

    return cleaned or "확인해볼 만한 장면이에요."


def _friendly_labels(value) -> str:
    labels = _normalize_labels(value)
    normalized = labels.lower()

    if "cat" in normalized:
        return "고양이가 보이는 장면이에요."
    if "dog" in normalized:
        return "강아지가 보이는 장면이에요."
    if "person" in normalized:
        return "사람이 함께 보이는 장면이에요."

    return "검색어와 비슷한 장면이에요."


def _deduplicate_behavior_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    behavior_events 중복 제거
    """
    seen = set()
    result = []

    for event in events:
        event_id = event.get("id")

        if event_id is not None:
            key = ("id", event_id)
        else:
            key = (
                event.get("video_id"),
                round(float(event.get("start_time") or 0.0), 2),
                round(float(event.get("end_time") or 0.0), 2),
                event.get("action"),
                event.get("target_object"),
            )

        if key in seen:
            continue

        seen.add(key)
        result.append(event)

    return result
