import base64
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gpt-4o-mini"
MAX_REPRESENTATIVE_FRAMES = 6
SEGMENT_GAP_SECONDS = 3.0


def extract_behavior_events(
    indexed_frames: list[dict] | None = None,
    video_id: str | None = None,
    limit: int = 5,
    model: str = DEFAULT_MODEL,
    max_segments: int = 8,
) -> list[dict]:
    """
    프레임 목록을 행동 구간 단위로 묶고, 각 구간에서 강아지 행동 이벤트를 추출한다.

    우선순위:
    1. OPENAI_API_KEY가 있고 로컬 프레임 이미지가 존재하면 VLM 기반 행동 분석
    2. API 키가 없거나 실패하면 object_labels 기반 fallback 행동 이벤트 생성

    Args:
        indexed_frames:
            tasks.py에서 생성된 프레임 정보 리스트.
            각 item 예:
            {
                "frame_path": "...",
                "timestamp": 18.0,
                "motion_start": 18.0,
                "motion_end": 42.0,
                "segment_id": 0,
                "video_id": "...",
                "s3_key": "...",
                "frame_id": "...",
                "object_labels": ["dog", "bowl"]
            }

        video_id:
            특정 영상 ID. indexed_frames가 없을 때 ChromaDB에서 조회할 수 있다.

        limit:
            최종 반환할 행동 이벤트 개수.

        model:
            VLM 분석에 사용할 OpenAI 모델.

        max_segments:
            너무 많은 구간을 분석하지 않도록 제한하는 값.

    Returns:
        behavior_events:
        [
            {
                "video_id": "...",
                "start_time": 18.0,
                "end_time": 42.0,
                "subject": "dog",
                "action": "초록색 그릇을 앞발로 반복해서 건드림",
                "target_object": "bowl",
                "summary": "...",
                "duration": 24.0,
                "repeat_count": 5,
                "confidence": 0.82,
                "interestingness": 0.91,
                "evidence": [...],
                "source_frames": [...]
            },
            ...
        ]
    """
    frames = indexed_frames or _load_indexed_frames(video_id=video_id)

    if video_id:
        frames = [
            frame for frame in frames
            if frame.get("video_id") == video_id
        ]

    if not frames:
        return []

    frame_groups = _group_frames_by_segment(frames)

    if not frame_groups:
        return []

    # 너무 짧거나 의미 없는 구간보다 프레임 수가 많은 구간을 먼저 분석한다.
    frame_groups = sorted(
        frame_groups,
        key=lambda group: (
            len(group),
            _get_group_duration(group),
        ),
        reverse=True,
    )[:max_segments]

    behavior_events = []

    for group in frame_groups:
        representative_frames = _select_representative_frames(
            group,
            max_count=MAX_REPRESENTATIVE_FRAMES,
        )

        if not representative_frames:
            continue

        event = None

        if _can_use_vlm(representative_frames):
            event = _analyze_segment_with_vlm(
                frames=representative_frames,
                all_segment_frames=group,
                model=model,
            )

        if not event:
            event = _build_fallback_behavior_event(group)

        if not event:
            continue

        normalized_event = _normalize_behavior_event(
            event=event,
            group=group,
            representative_frames=representative_frames,
        )

        if normalized_event:
            behavior_events.append(normalized_event)

    behavior_events = _deduplicate_behavior_events(behavior_events)

    behavior_events = sorted(
        behavior_events,
        key=lambda item: (
            float(item.get("interestingness") or 0.0),
            float(item.get("confidence") or 0.0),
            float(item.get("duration") or 0.0),
        ),
        reverse=True,
    )

    return behavior_events[:limit]


def format_behavior_events_for_prompt(events: list[dict]) -> str:
    """
    question_generator.py에서 LLM 프롬프트에 넣기 좋은 형태로 행동 이벤트를 변환한다.
    """
    if not events:
        return "- 행동 이벤트 없음"

    lines = []

    for idx, event in enumerate(events, start=1):
        start_time = event.get("start_time")
        end_time = event.get("end_time")

        if start_time is not None and end_time is not None:
            time_text = f"{float(start_time):.1f}초~{float(end_time):.1f}초"
        else:
            time_text = "시간 정보 없음"

        evidence = event.get("evidence") or []
        evidence_text = ", ".join(str(item) for item in evidence[:3]) or "근거 없음"

        lines.append(
            f"{idx}. 시간: {time_text}\n"
            f"   주체: {event.get('subject') or '알 수 없음'}\n"
            f"   행동: {event.get('action') or '알 수 없음'}\n"
            f"   대상: {event.get('target_object') or '없음'}\n"
            f"   요약: {event.get('summary') or '없음'}\n"
            f"   지속 시간: {event.get('duration') or 0}초\n"
            f"   반복 횟수: {event.get('repeat_count') or 0}\n"
            f"   신뢰도: {event.get('confidence') or 0}\n"
            f"   흥미도: {event.get('interestingness') or 0}\n"
            f"   근거: {evidence_text}"
        )

    return "\n".join(lines)


def summarize_behavior_events(events: list[dict], max_items: int = 5) -> str:
    """
    UI 또는 RAG context에 넣기 좋은 행동 이벤트 요약 문자열 생성
    """
    if not events:
        return "아직 분석된 주요 행동이 없습니다."

    summaries = []

    for event in events[:max_items]:
        start_time = event.get("start_time")
        end_time = event.get("end_time")
        summary = event.get("summary") or event.get("action") or "행동 설명 없음"

        if start_time is not None and end_time is not None:
            summaries.append(
                f"{float(start_time):.1f}~{float(end_time):.1f}초: {summary}"
            )
        else:
            summaries.append(summary)

    return " / ".join(summaries)


def _load_indexed_frames(video_id: str | None = None) -> list[dict]:
    """
    indexed_frames가 직접 전달되지 않았을 때 ChromaDB metadata에서 프레임을 가져온다.

    단, ChromaDB metadata에는 현재 segment_id, motion_start, motion_end가 없을 수 있다.
    그래서 tasks.py에서 indexed_frames를 직접 넘겨주는 방식이 더 좋다.
    """
    try:
        from pipeline.vector_store import get_indexed_frames

        return get_indexed_frames(video_id=video_id)
    except Exception as exc:
        print(f"인덱싱된 프레임 조회 실패: {exc}", flush=True)
        return []


def _group_frames_by_segment(frames: list[dict]) -> list[list[dict]]:
    """
    프레임들을 행동 분석 단위로 묶는다.

    우선순위:
    1. segment_id가 있으면 video_id + segment_id 기준으로 묶는다.
    2. segment_id가 없으면 timestamp 간격이 일정 이상 벌어질 때 새 구간으로 나눈다.
    """
    if not frames:
        return []

    has_segment_id = any(frame.get("segment_id") is not None for frame in frames)

    if has_segment_id:
        grouped = defaultdict(list)

        for frame in frames:
            key = (
                frame.get("video_id") or "unknown",
                frame.get("segment_id"),
                frame.get("motion_start"),
                frame.get("motion_end"),
            )
            grouped[key].append(frame)

        groups = []

        for group in grouped.values():
            groups.append(
                sorted(
                    group,
                    key=lambda item: float(item.get("timestamp") or 0.0),
                )
            )

        return groups

    sorted_frames = sorted(
        frames,
        key=lambda item: (
            item.get("video_id") or "",
            float(item.get("timestamp") or 0.0),
        ),
    )

    groups = []
    current_group = []
    previous_video_id = None
    previous_timestamp = None

    for frame in sorted_frames:
        current_video_id = frame.get("video_id")
        current_timestamp = float(frame.get("timestamp") or 0.0)

        should_start_new_group = False

        if not current_group:
            should_start_new_group = True
        elif current_video_id != previous_video_id:
            should_start_new_group = True
        elif previous_timestamp is not None and current_timestamp - previous_timestamp > SEGMENT_GAP_SECONDS:
            should_start_new_group = True

        if should_start_new_group:
            if current_group:
                groups.append(current_group)
            current_group = [frame]
        else:
            current_group.append(frame)

        previous_video_id = current_video_id
        previous_timestamp = current_timestamp

    if current_group:
        groups.append(current_group)

    return groups


def _select_representative_frames(
    frames: list[dict],
    max_count: int = MAX_REPRESENTATIVE_FRAMES,
) -> list[dict]:
    """
    행동 분석에 사용할 대표 프레임을 고른다.

    프레임이 많으면 처음, 중간, 끝이 골고루 들어가게 선택한다.
    """
    if not frames:
        return []

    frames = sorted(
        frames,
        key=lambda item: float(item.get("timestamp") or 0.0),
    )

    if len(frames) <= max_count:
        return frames

    if max_count <= 1:
        return [frames[0]]

    selected = []
    last_index = len(frames) - 1

    for i in range(max_count):
        index = round(i * last_index / (max_count - 1))
        selected.append(frames[index])

    return selected


def _can_use_vlm(frames: list[dict]) -> bool:
    """
    VLM 분석 가능 여부 확인

    조건:
    - OPENAI_API_KEY 존재
    - 대표 프레임 이미지 파일이 로컬에 존재
    """
    if not os.getenv("OPENAI_API_KEY"):
        return False

    for frame in frames:
        frame_path = frame.get("frame_path")
        if frame_path and Path(frame_path).exists():
            return True

    return False


def _analyze_segment_with_vlm(
    frames: list[dict],
    all_segment_frames: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict | None:
    """
    여러 프레임 이미지를 VLM에 전달하여 행동 이벤트를 생성한다.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        image_contents = []

        for frame in frames:
            frame_path = frame.get("frame_path")

            if not frame_path:
                continue

            path = Path(frame_path)

            if not path.exists():
                continue

            data_url = _image_to_data_url(path)

            image_contents.append({
                "type": "input_image",
                "image_url": data_url,
            })

        if not image_contents:
            return None

        segment_context = _build_segment_context(all_segment_frames)
        prompt = _build_vlm_prompt(segment_context)

        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        *image_contents,
                    ],
                }
            ],
        )

        output_text = response.output_text.strip()
        parsed = _parse_json_output(output_text)

        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else None

        if not isinstance(parsed, dict):
            return None

        return parsed

    except Exception as exc:
        print(f"행동 이벤트 VLM 분석 실패: {exc}", flush=True)
        return None


def _build_segment_context(frames: list[dict]) -> dict:
    """
    VLM 프롬프트에 전달할 구간 정보 생성
    """
    sorted_frames = sorted(
        frames,
        key=lambda item: float(item.get("timestamp") or 0.0),
    )

    start_time = _get_group_start(sorted_frames)
    end_time = _get_group_end(sorted_frames)
    duration = round(end_time - start_time, 2)

    label_counter = Counter()

    frame_summaries = []

    for frame in sorted_frames:
        labels = _parse_object_labels(frame.get("object_labels"))
        label_counter.update(labels)

        frame_summaries.append({
            "timestamp": frame.get("timestamp"),
            "object_labels": labels,
            "frame_id": frame.get("frame_id"),
        })

    return {
        "video_id": sorted_frames[0].get("video_id") if sorted_frames else None,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "frame_count": len(sorted_frames),
        "label_counts": dict(label_counter),
        "frames": frame_summaries,
    }


def _build_vlm_prompt(segment_context: dict) -> str:
    """
    VLM 행동 분석 프롬프트 생성

    동물 종류는 코드에서 dog/cat/bird로 고정하지 않고,
    이미지 자체를 보고 모델이 판단하게 한다.
    """
    context_json = json.dumps(
        segment_context,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
너는 반려동물 펫캠 영상 분석 도우미다.

아래 이미지는 같은 움직임 구간에서 시간 순서대로 추출된 프레임들이다.
사용자가 궁금해할 만한 반려동물의 행동 이벤트를 1개만 추출해라.

중요한 규칙:
- 이미지에서 확인 가능한 내용만 말한다.
- 확실하지 않은 행동은 단정하지 말고 "~로 보임", "~하는 것으로 보임"처럼 표현한다.
- 단순히 객체 라벨이 보인다는 설명은 피한다.
- 동물 종류는 코드나 metadata의 라벨을 그대로 따라 쓰지 말고, 이미지를 보고 판단한다.
- subject는 항상 "반려동물"이라고 적는다.
- action과 summary에서도 강아지, 고양이, 새처럼 특정 동물명을 쓰지 말고 "반려동물"이라고 표현한다.
- 동물 종류를 맞히는 것이 목적이 아니라, 행동과 상황을 설명하는 것이 목적이다.
- 이미지에서 특정 동물처럼 보여도 화면에 표시되는 설명에서는 "반려동물"로 일반화한다.
- 행동, 대상, 반복 여부, 지속 시간을 중심으로 설명한다.
- 보호자에게 유용한 행동일수록 interestingness를 높게 준다.
- 건강, 식사, 물 섭취, 불안, 반복 행동, 사람/물체와의 상호작용은 중요하게 본다.
- JSON 객체만 출력한다.
- markdown 코드블록을 쓰지 마라.

구간 metadata:
{context_json}

출력 형식:
{{
  "subject": "반려동물",
  "action": "반려동물이 무엇을 하는 것으로 보이는지 짧게 작성",
  "target_object": "행동 대상이 있으면 작성, 없으면 null",
  "summary": "반려동물의 행동을 사용자가 이해하기 쉬운 한 문장으로 요약",
  "repeat_count": 1,
  "confidence": 0.7,
  "interestingness": 0.8,
  "evidence": [
    "판단 근거 1",
    "판단 근거 2"
  ]
}}

""".strip()

def _build_fallback_behavior_event(frames: list[dict]) -> dict | None:
    """
    VLM을 사용할 수 없을 때 motion 구간 기반으로 최소 행동 이벤트를 만든다.

    fallback에서는 동물 종류를 코드로 판단하지 않는다.
    dog/cat/bird 같은 라벨을 subject로 쓰지 않고,
    항상 '반려동물'로 일반화한다.
    """
    if not frames:
        return None

    start_time = _get_group_start(frames)
    end_time = _get_group_end(frames)
    duration = round(end_time - start_time, 2)

    if duration <= 0:
        return None

    label_counter = Counter()

    for frame in frames:
        labels = set(_parse_object_labels(frame.get("object_labels")))
        label_counter.update(labels)

    detected_labels = [
        label
        for label, _ in label_counter.most_common(5)
    ]

    subject = "반려동물"
    target_object = None
    action = "움직임이 있는 구간으로 감지됨"
    summary = f"반려동물이 {duration:.1f}초 동안 움직임을 보인 것으로 추정되는 구간입니다."

    evidence = [
        f"motion 구간 {start_time:.1f}초~{end_time:.1f}초에서 프레임 {len(frames)}개 추출",
    ]

    if detected_labels:
        evidence.append(f"함께 감지된 객체 라벨: {detected_labels}")

    confidence = 0.45
    interestingness = 0.5

    if duration >= 10:
        interestingness += 0.05
        evidence.append(f"{duration:.1f}초 동안 지속된 구간")

    if len(frames) >= 4:
        interestingness += 0.05
        evidence.append("여러 프레임에서 움직임이 이어짐")

    return {
        "subject": subject,
        "action": action,
        "target_object": target_object,
        "summary": summary,
        "duration": duration,
        "repeat_count": len(frames),
        "confidence": round(min(confidence, 0.95), 2),
        "interestingness": round(min(interestingness, 0.95), 2),
        "evidence": evidence,
    }

def _normalize_behavior_event(
    event: dict,
    group: list[dict],
    representative_frames: list[dict],
) -> dict | None:
    """
    VLM 또는 fallback에서 나온 행동 이벤트를 DB 저장용 표준 형태로 정리한다.
    """
    if not group:
        return None

    start_time = _get_group_start(group)
    end_time = _get_group_end(group)
    duration = round(end_time - start_time, 2)

    first_frame = group[0]
    video_id = first_frame.get("video_id")

    subject = event.get("subject") or _guess_subject_from_frames(group)
    action = event.get("action")
    summary = event.get("summary") or action
        # 화면 표시에서는 특정 동물명을 쓰지 않고 반려동물로 일반화한다.
    subject = "반려동물"
    action = _generalize_subject_prefix(action)
    summary = _generalize_subject_prefix(summary)

    if not action and not summary:
        return None

    target_object = (
        event.get("target_object")
        or event.get("target")
        or event.get("object")
    )

    repeat_count = event.get("repeat_count", 0)
    confidence = _clamp_score(event.get("confidence"), default=0.65)
    interestingness = _clamp_score(event.get("interestingness"), default=0.7, treat_zero_as_default=False)

    evidence = event.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]

    source_frames = []

    for frame in representative_frames:
        source_frames.append({
            "frame_id": frame.get("frame_id"),
            "frame_path": frame.get("frame_path"),
            "s3_key": frame.get("s3_key"),
            "timestamp": frame.get("timestamp"),
            "object_labels": _parse_object_labels(frame.get("object_labels")),
        })

    return {
        "video_id": video_id,
        "start_time": start_time,
        "end_time": end_time,
        "subject": subject,
        "action": action,
        "target_object": target_object,
        "summary": summary,
        "duration": duration,
        "repeat_count": int(repeat_count or 0),
        "confidence": confidence,
        "interestingness": interestingness,
        "evidence": evidence,
        "source_frames": source_frames,
    }


def _deduplicate_behavior_events(events: list[dict]) -> list[dict]:
    """
    비슷한 시간대와 행동 설명을 가진 이벤트 중복 제거
    """
    seen = set()
    result = []

    for event in events:
        start_time = round(float(event.get("start_time") or 0.0), 1)
        end_time = round(float(event.get("end_time") or 0.0), 1)
        action = event.get("action") or ""
        target_object = event.get("target_object") or ""

        key = (
            event.get("video_id"),
            start_time,
            end_time,
            action[:30],
            target_object,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(event)

    return result


def _image_to_data_url(path: Path) -> str:
    """
    이미지 파일을 OpenAI input_image에 넣을 수 있는 data URL로 변환한다.
    """
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    elif suffix == ".webp":
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"

    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def _parse_json_output(output_text: str) -> Any:
    """
    LLM 출력에서 JSON 객체 또는 배열을 파싱한다.
    """
    if not output_text:
        return None

    cleaned = output_text.strip()

    # ```json ... ``` 형태 제거
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 앞뒤에 설명이 섞인 경우 JSON 부분만 추출
    object_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _parse_object_labels(raw_labels: Any) -> list[str]:
    """
    object_labels 값을 list[str]로 변환한다.

    지원 형태:
    - ["dog", "bowl"]
    - '["dog", "bowl"]'
    - "dog,bowl"
    - None
    """
    if not raw_labels:
        return []

    if isinstance(raw_labels, list):
        return [
            str(label).strip()
            for label in raw_labels
            if str(label).strip()
        ]

    try:
        parsed = json.loads(raw_labels)

        if isinstance(parsed, list):
            return [
                str(label).strip()
                for label in parsed
                if str(label).strip()
            ]

        if isinstance(parsed, str):
            parsed = parsed.strip()
            return [parsed] if parsed else []

    except Exception:
        pass

    return [
        label.strip()
        for label in str(raw_labels).split(",")
        if label.strip()
    ]

def _guess_subject_from_frames(frames: list[dict]) -> str | None:
    """
    fallback에서 프레임 라벨만 보고 동물 종류를 추정하지 않는다.
    동물 종류 판단은 VLM이 하도록 둔다.
    """
    return "반려동물"


def _get_group_start(frames: list[dict]) -> float:
    """
    그룹 시작 시간 계산
    """
    motion_starts = [
        frame.get("motion_start")
        for frame in frames
        if frame.get("motion_start") is not None
    ]

    if motion_starts:
        return round(float(min(motion_starts)), 2)

    timestamps = [
        float(frame.get("timestamp") or 0.0)
        for frame in frames
    ]

    return round(min(timestamps), 2) if timestamps else 0.0


def _get_group_end(frames: list[dict]) -> float:
    """
    그룹 종료 시간 계산
    """
    motion_ends = [
        frame.get("motion_end")
        for frame in frames
        if frame.get("motion_end") is not None
    ]

    if motion_ends:
        return round(float(max(motion_ends)), 2)

    timestamps = [
        float(frame.get("timestamp") or 0.0)
        for frame in frames
    ]

    return round(max(timestamps), 2) if timestamps else 0.0


def _get_group_duration(frames: list[dict]) -> float:
    """
    그룹 지속 시간 계산
    """
    if not frames:
        return 0.0

    return round(_get_group_end(frames) - _get_group_start(frames), 2)


def _clamp_score(value, default=0.5, treat_zero_as_default=True) -> float:
    """
    점수를 0.0~1.0 사이로 보정한다.

    VLM이 출력 예시를 그대로 따라 해서 0.0을 내는 경우가 있어,
    기본적으로 0 이하 값은 default로 보정한다.
    """
    try:
        score = float(value)
    except Exception:
        score = float(default)

    if treat_zero_as_default and score <= 0:
        score = float(default)

    return round(max(0.0, min(score, 1.0)), 2)

def _generalize_subject_prefix(text: str | None) -> str | None:
    """
    문장 맨 앞의 특정 주어를 '반려동물'로 일반화한다.

    예:
    - 고양이가 물을 마시는 것으로 보임
      -> 반려동물이 물을 마시는 것으로 보임

    - 강아지가 바닥을 긁는 것으로 보임
      -> 반려동물이 바닥을 긁는 것으로 보임

    동물 리스트를 코드에 두지 않고, 문장 앞 주어 패턴만 일반화한다.
    """
    if not text:
        return text

    text = str(text).strip()

    # 문장 맨 앞의 "OO이/가 " 패턴을 반려동물로 바꾼다.
    text = re.sub(r"^[^\s]+(이|가)\s+", "반려동물이 ", text)

    return text