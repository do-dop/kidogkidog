import os
from typing import Any

from pipeline.vector_store import get_indexed_frames, search_with_query_expansion
from pipeline.query_analyzer import build_prompt_hint
from db.metadata import (
    get_behavior_event_by_id,
    get_top_behavior_events,
    get_behavior_events_overlapping,
)


DEFAULT_FALLBACK_MESSAGE = (
    "현재 OPENAI_API_KEY가 설정되어 있지 않아 LLM 답변은 생성하지 못했습니다. "
    "대신 분석된 행동 이벤트와 검색된 프레임 metadata를 기준으로 결과를 확인해주세요."
)


def run_rag_query(
    query: str,
    video_id: str | None = None,
    top_k: int = 5,
    user_id: str | None = None,
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

    if source_event:
        return _run_event_grounded_query(
            query=query,
            source_event=source_event,
            top_k=top_k,
            user_id=user_id,
        )

    try:
        retrieved_frames = search_with_query_expansion(
            query=query,
            top_k=top_k,
            video_id=video_id,
        )
    except Exception as exc:
        # ChromaDB가 깨졌거나 인덱스 파일이 불안정할 때도
        # behavior_events 기반 답변은 가능하도록 앱 전체를 죽이지 않는다.
        print(f"ChromaDB 검색 실패: {exc}", flush=True)
        retrieved_frames = []

    behavior_events = _get_relevant_behavior_events(
        query=query,
        video_id=video_id,
        retrieved_frames=retrieved_frames,
        limit=5,
    )

    if not retrieved_frames and behavior_events:
        retrieved_frames = _frames_from_behavior_events(
            behavior_events=behavior_events,
            video_id=video_id,
            limit=top_k,
        )

    if not retrieved_frames and not behavior_events:
        retrieved_frames = _fallback_indexed_frames(
            video_id=video_id,
            limit=top_k,
        )

    if not retrieved_frames and not behavior_events:
        return {
            "query": query,
            "video_id": video_id,
            "answer": (
                f"질문 '{query}'과 정확히 맞는 장면은 아직 찾지 못했습니다. "
                "다만 이 영상의 인덱스가 충분하지 않거나 분석 결과가 아직 생성 중일 수 있어요."
            ),
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

            overlapping_events = get_behavior_events_overlapping(
                video_id=frame_video_id,
                start_time=max(timestamp - 3.0, 0.0),
                end_time=timestamp + 3.0,
                limit=5,
            )

            candidates.extend(overlapping_events)

        except Exception as exc:
            print(f"겹치는 행동 이벤트 조회 실패: {exc}", flush=True)

    # 2. 영상 전체 주요 behavior_events 조회
    try:
        top_events = get_top_behavior_events(
            video_id=video_id,
            limit=10,
        )
        candidates.extend(top_events)

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


def _fallback_indexed_frames(video_id: str | None, limit: int = 5) -> list[dict[str, Any]]:
    try:
        frames = get_indexed_frames(video_id=video_id)
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

    if not fallback_frames and video_id:
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


def _behavior_query_score(query: str, event: dict[str, Any]) -> float:
    """
    질문 텍스트와 행동 이벤트가 얼마나 관련 있는지 간단히 점수화한다.

    복잡한 임베딩 검색이 아니라, 현재는 action/summary/evidence의 키워드 포함 여부로 판단한다.
    """
    if not query:
        return 0.0

    query_text = query.lower().replace(" ", "")

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
    event_text = event_text.lower().replace(" ", "")

    score = 0.0

    keyword_groups = [
        ["횟수", "몇번", "몇회", "반복"],
        ["그릇", "사료", "밥", "먹", "물", "마시", "bowl", "cup", "bottle"],
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

        lines.append(
            f"{idx}. video_id: {frame.get('video_id', 'unknown')}\n"
            f"   frame_id: {frame.get('frame_id', 'unknown')}\n"
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
- behavior_events에 관련 정보가 있으면 "확인할 수 없습니다"라고 먼저 말하지 말고, 가능한 범위에서 답변한다.
- repeat_count가 있으면 "정확한 실제 횟수"가 아니라 "프레임 기반 반복 감지 후보 N회"라고 표현한다.
- 사용자가 "몇 번"을 물으면 repeat_count를 활용하되, 실제 행동 횟수가 아니라 분석 후보라는 점을 함께 말한다.
- 행동을 과하게 단정하지 않는다.
- "확실히 먹었다", "반드시 불안하다"처럼 단정하지 말고 "~로 보입니다", "~후보로 볼 수 있습니다"라고 표현한다.
- object_labels는 오탐 가능성이 있으므로, 객체 탐지 기반 정보는 조심스럽게 말한다.
- 답변에는 관련 시간대, 행동 요약, 반복 후보 횟수 또는 근거를 포함한다.
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
    if behavior_events:
        top_items = []

        for event in behavior_events[:3]:
            start_time = event.get("start_time")
            end_time = event.get("end_time")

            if start_time is not None and end_time is not None:
                time_text = f"{float(start_time):.2f}초~{float(end_time):.2f}초"
            else:
                time_text = "시간 정보 없음"

            summary = event.get("summary") or event.get("action") or "행동 설명 없음"
            repeat_count = event.get("repeat_count") or 0
            confidence = event.get("confidence")
            evidence = event.get("evidence") or []

            evidence_text = ""
            if isinstance(evidence, list) and evidence:
                evidence_text = f" 근거: {', '.join(str(item) for item in evidence[:2])}"

            confidence_text = ""
            if confidence is not None:
                confidence_text = f", 신뢰도 {float(confidence):.2f}"

            top_items.append(
                f"- {time_text}: {summary} "
                f"(반복 감지 후보 {repeat_count}회{confidence_text})."
                f"{evidence_text}"
            )

        joined = "\n".join(top_items)

        return (
            f"질문 '{query}'에 대해 분석된 행동 이벤트 기준으로는 아래 구간을 확인할 수 있습니다.\n\n"
            f"{joined}\n\n"
            "단, 반복 횟수는 실제 행동을 사람이 직접 센 값이 아니라 "
            "프레임 기반 분석에서 반복 감지된 후보 횟수로 보는 것이 적절합니다."
        )

    if retrieved_frames:
        top_items = []
        has_strong_match = any(float(frame.get("score") or 0.0) > 0 for frame in retrieved_frames)

        for frame in retrieved_frames[:3]:
            timestamp = frame.get("timestamp")
            timestamp_text = f"{float(timestamp):.2f}초" if timestamp is not None else "알 수 없는 시간"
            labels = frame.get("object_labels") or "감지 객체 없음"
            score = frame.get("score")
            score_text = f"{float(score):.4f}" if score is not None else "알 수 없음"

            top_items.append(
                f"- {timestamp_text}: 감지 객체 `{labels}`, 유사도 {score_text}"
            )

        joined = "\n".join(top_items)

        if not has_strong_match:
            return (
                f"질문 '{query}'과 정확히 일치하는 장면은 아직 뚜렷하게 찾지 못했습니다.\n\n"
                "다만 같은 영상에서 확인 가능한 유사 후보 구간은 아래와 같습니다.\n\n"
                f"{joined}\n\n"
                "영상 분석 결과가 더 쌓이면 추천 질문과 검색 매칭이 더 안정적으로 연결됩니다."
            )

        return (
            f"질문 '{query}'과 관련된 후보 장면은 아래 시간대에서 확인해볼 수 있습니다.\n\n"
            f"{joined}\n\n"
            "단, 현재 답변은 검색 metadata 기반이며, 실제 행동을 확정하는 것은 아닙니다."
        )

    return (
        f"질문 '{query}'과 정확히 맞는 장면은 아직 찾지 못했습니다. "
        "다만 영상 분석 결과가 생성되는 중일 수 있어 잠시 후 다시 검색해보면 더 가까운 후보가 나올 수 있습니다."
    )


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
