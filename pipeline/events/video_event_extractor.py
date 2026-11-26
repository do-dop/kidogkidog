"""청크 영상을 Gemini에 직접 넣어 행동 이벤트를 뽑는다 (프레임 6장 방식의 대안).

설계: docs/video-event-extraction-design.md

- BEHAVIOR_EVENT_SOURCE=frames(기본) | video | both
  - frames: 기존 프레임 기반 추출만 쓴다. 이 모듈을 설정하지 않으면 동작이 바뀌지 않는다.
  - video: 영상 직접 입력을 쓰고, 호출이 실패하면(None) 프레임 방식으로 대체한다.
  - both: DB에는 프레임 방식 결과만 돌려주고, 영상 방식 결과는 VIDEO_EVENT_SHADOW_DIR에 JSON으로만 남긴다.
- Gemini REST를 httpx로 직접 호출한다(새 의존성 없음). 키는 URL이 아니라 헤더로 보낸다.
- 호출 실패는 예외를 내지 않는다. 청크 전체 처리가 이 단계 때문에 실패하면 안 된다.
- 영상은 Google로 전송된다. 기본값이 frames인 이유다.
"""

import base64
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from pipeline.events.behavior_event_extractor import (
    _deduplicate_behavior_events,
    _normalize_behavior_event,
    extract_behavior_events,
)

DEFAULT_VIDEO_EVENT_MODEL = "gemini-3.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
REQUEST_TIMEOUT_SECONDS = 180.0
MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 5.0
# base64로 부풀기 때문에 요청 한도(20MB)보다 작게 잡는다. 이보다 크면 Files API로 올린다.
INLINE_MAX_BYTES = 12 * 1024 * 1024
FILE_ACTIVE_WAIT_SECONDS = 60.0
# 업로드·생성·재시도를 모두 합친 호출 전체의 시간 상한. 요청 단위 타임아웃만으로는 응답이 멈추면 청크 하나가 수십 분 걸릴 수 있다
# (로컬 측정에서 호출 한 번이 1,193초 걸린 적이 있다). 넘기면 None을 돌려 프레임 방식으로 대체되게 한다.
TOTAL_DEADLINE_SECONDS = 300.0
# 이벤트 구간 안에 색인 프레임이 없을 때, 이 거리 안의 가장 가까운 프레임까지 구간을 넓혀 근거 프레임을 잡는다.
EVIDENCE_SNAP_SECONDS = 2.0

SOURCES = ("frames", "video", "both")

_PROMPT = """이 영상은 고정 카메라로 찍은 반려동물/가정 영상이다. 영상에서 실제로 보이는 행동을 시간순으로 빠짐없이 뽑아라.

규칙:
- 화면에서 직접 확인되는 행동만 쓴다. 보이지 않는 의도·원인·감정·화면 밖 상황은 추측하지 않는다.
- 행동이 바뀌면 이벤트를 나눈다. 이동 방향, 자세, 장소, 대상이 바뀌는 지점마다 새 이벤트로 쓰고, 서로 다른 행동을 한 이벤트로 합치지 않는다.
- 한 이벤트는 보통 2~6초 길이다. 7초를 넘는 이벤트는 그 안에서 행동이 정말 바뀌지 않았을 때만 허용한다.
- 정지해 있거나 머무는 상태(앉아 있음, 서 있음, 무언가 위에 있음)도 2초 이상 이어지면 이벤트로 쓴다.
- 대상이 화면에 있는 구간은 비우지 말고 이어서 쓴다. 대상이 화면에 없는 구간은 쓰지 않는다.
- 종료 화면, 자막, 광고만 보이는 구간은 이벤트가 아니다.
- 소파·의자·문 같은 가구나 물건이 움직이거나 넘어지는 등 장면 자체가 바뀌면 반려동물 행동과 별도의 이벤트로 쓰고 subject_type을 "scene", target_object를 그 물건으로 쓴다. 그 외에는 subject_type을 "pet"으로 쓴다.
- start_time, end_time은 초 단위 소수 첫째 자리이고 start_time < end_time 이다.
- action과 summary에서는 강아지·고양이 같은 특정 동물명을 쓰지 않고 "반려동물"이라고 쓴다. 다른 개체는 "다른 반려동물"로 구분한다.
- action은 행동을 한 문장으로 쓴다.
- summary는 보호자에게 말해 주듯 해요체 한두 문장으로 쓴다. 위치, 이동 방향, 함께 있는 개체나 물건처럼 영상에서 보이는 구체적인 내용을 넣는다. 확실하지 않은 것은 "~로 보여요"로 쓴다.
- 확인할 행동이 없으면 빈 배열을 반환한다.

JSON 배열만 출력한다. markdown 코드블록을 쓰지 마라. 각 원소의 필드:
start_time, end_time, subject_type("pet" 또는 "scene"), target_object(scene일 때만), action, summary,
appearance(보이는 개체의 털색·크기 등 외형. 종 이름은 여기에만 쓸 수 있다), count(화면에 보이는 개체 수, 정수),
location(화면 안 위치나 장소), direction(이동 방향. 이동이 없으면 빈 문자열), objects(상호작용한 물건 목록),
evidence_description(화면에서 본 근거), confidence(0~1), interestingness(0~1, 보호자에게 유용할수록 높게),
action_type(eat, drink, walk, run, jump, swim, sniff, look, rest, play, groom, hide, interact_person, leave_frame, other 중 하나),
continues_previous(앞 이벤트와 같은 행동이 끊기지 않고 이어지면 true, 아니면 false)
"""


def event_source() -> str:
    """BEHAVIOR_EVENT_SOURCE를 읽는다. 값이 이상하면 기본(frames)으로 되돌린다."""
    value = os.getenv("BEHAVIOR_EVENT_SOURCE", "frames").strip().lower()
    return value if value in SOURCES else "frames"


def is_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _model() -> str:
    return os.getenv("VIDEO_EVENT_MODEL", DEFAULT_VIDEO_EVENT_MODEL)


def _headers() -> dict[str, str]:
    return {"x-goog-api-key": os.getenv("GEMINI_API_KEY", "")}


def _upload_file(path: Path) -> dict | None:
    """Files API로 영상을 올리고 ACTIVE가 될 때까지 기다린다. 실패하면 None."""
    size = path.stat().st_size
    start = httpx.post(
        GEMINI_UPLOAD_URL,
        headers={
            **_headers(),
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": "video/mp4",
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": "chunk"}},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    upload_url = start.headers.get("x-goog-upload-url")
    if start.status_code != 200 or not upload_url:
        print(f"영상 업로드 시작 실패: HTTP {start.status_code}", flush=True)
        return None
    finish = httpx.post(
        upload_url,
        headers={
            "Content-Length": str(size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        content=path.read_bytes(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if finish.status_code != 200:
        print(f"영상 업로드 실패: HTTP {finish.status_code}", flush=True)
        return None
    info = finish.json().get("file") or {}
    deadline = time.monotonic() + FILE_ACTIVE_WAIT_SECONDS
    while info.get("state") == "PROCESSING" and time.monotonic() < deadline:
        time.sleep(2)
        poll = httpx.get(f"{GEMINI_BASE_URL}/{info['name']}", headers=_headers(), timeout=30.0)
        if poll.status_code != 200:
            break
        info = poll.json()
    return info if info.get("state") == "ACTIVE" else None


def _delete_file(name: str) -> None:
    try:
        httpx.delete(f"{GEMINI_BASE_URL}/{name}", headers=_headers(), timeout=30.0)
    except httpx.HTTPError:
        pass


def _describe_unusable(payload: dict, text: str) -> str:
    """해석하지 못한 응답의 원인 정보. 영상 내용이 담길 수 있는 응답 본문은 남기지 않는다."""
    candidate = (payload.get("candidates") or [{}])[0]
    usage = payload.get("usageMetadata") or {}
    if not text:
        reason = "본문 없음"
    else:
        try:
            json.loads(_strip_code_fence(text))
            reason = "이벤트 목록 아님"
        except ValueError as exc:
            reason = f"JSON 오류: {exc}"
    return (
        f"{reason}, finishReason={candidate.get('finishReason')}, "
        f"blockReason={(payload.get('promptFeedback') or {}).get('blockReason')}, 글자 수={len(text)}, "
        f"출력 토큰={usage.get('candidatesTokenCount')}, 추론 토큰={usage.get('thoughtsTokenCount')}"
    )


def _generate(video_part: dict[str, Any]) -> str | None:
    body = {
        "contents": [{"role": "user", "parts": [video_part, {"text": _PROMPT}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    url = f"{GEMINI_BASE_URL}/models/{_model()}:generateContent"
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = httpx.post(url, json=body, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                payload = response.json()
                candidates = payload.get("candidates") or []
                parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
                text = "".join(part.get("text", "") for part in parts).strip()
                if text and parse_events(text) is not None:
                    return text
                # HTTP는 성공했지만 쓸 수 없는 응답이다. 원인을 남기고 한 번 더 시도한 뒤 대체 방식에 맡긴다.
                print(f"영상 이벤트 응답 해석 실패: {_describe_unusable(payload, text)}", flush=True)
            else:
                print(f"영상 이벤트 추출 호출 실패: HTTP {response.status_code}", flush=True)
                if response.status_code not in (429, 500, 502, 503, 504):
                    return None
        except (httpx.HTTPError, ValueError) as exc:
            print(f"영상 이벤트 추출 호출 실패: {type(exc).__name__}", flush=True)
        if attempt + 1 < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS)
    return None


def _deadline_seconds() -> float:
    try:
        value = float(os.getenv("VIDEO_EVENT_DEADLINE_SECONDS", str(TOTAL_DEADLINE_SECONDS)))
    except ValueError:
        return TOTAL_DEADLINE_SECONDS
    return value if value > 0 else TOTAL_DEADLINE_SECONDS


def _call_gemini_video(video_path: str) -> str | None:
    """호출 전체에 시간 상한을 둔다. 넘기면 None. 남은 작업은 백그라운드에서 끝나며 올린 파일은 거기서 삭제된다."""
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["text"] = _call_gemini_video_unbounded(video_path)
        except Exception as exc:  # 이 단계가 청크 처리를 막으면 안 된다
            outcome["error"] = type(exc).__name__

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(_deadline_seconds())
    if thread.is_alive():
        print(f"영상 이벤트 추출 시간 초과({_deadline_seconds():.0f}초)", flush=True)
        return None
    if "error" in outcome:
        print(f"영상 이벤트 추출 실패: {outcome['error']}", flush=True)
        return None
    return outcome.get("text")


def _call_gemini_video_unbounded(video_path: str) -> str | None:
    path = Path(video_path)
    if not path.exists() or not is_available():
        return None
    try:
        inline_max = int(os.getenv("VIDEO_EVENT_INLINE_MAX_BYTES", str(INLINE_MAX_BYTES)))
    except ValueError:
        inline_max = INLINE_MAX_BYTES
    if path.stat().st_size <= inline_max:
        part = {"inline_data": {
            "mime_type": "video/mp4",
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }}
        return _generate(part)
    uploaded = None
    try:
        uploaded = _upload_file(path)
        if not uploaded:
            return None
        return _generate({"file_data": {"mime_type": "video/mp4", "file_uri": uploaded["uri"]}})
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        print(f"영상 업로드 실패: {type(exc).__name__}", flush=True)
        return None
    finally:
        if uploaded and uploaded.get("name"):
            _delete_file(uploaded["name"])


def _strip_code_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def parse_events(text: str) -> list[dict] | None:
    """모델 응답에서 이벤트 목록을 꺼낸다. 형식이 틀리면 None."""
    try:
        parsed = json.loads(_strip_code_fence(text))
    except ValueError:
        return None
    if isinstance(parsed, dict):
        parsed = parsed.get("events")
    if not isinstance(parsed, list):
        return None
    return [item for item in parsed if isinstance(item, dict)]


def _detail_line(event: dict) -> str | None:
    labels = (("appearance", "외형"), ("count", "개체 수"), ("location", "위치"), ("direction", "방향"))
    parts = [f"{label} {event[key]}" for key, label in labels if str(event.get(key) or "").strip()]
    objects = event.get("objects")
    if isinstance(objects, list) and objects:
        parts.append("물건 " + ", ".join(str(item) for item in objects if str(item).strip()))
    elif isinstance(objects, str) and objects.strip():
        parts.append(f"물건 {objects.strip()}")
    return "상세: " + "; ".join(parts) if parts else None


def _snap_to_frames(start: float, end: float, frames: list[dict]) -> tuple[float, float]:
    """구간 안에 색인 프레임이 없으면 가장 가까운 프레임까지 구간을 최소한으로 넓힌다."""
    times = [float(frame.get("timestamp") or 0.0) for frame in frames]
    if not times or any(start <= t <= end for t in times):
        return start, end
    midpoint = (start + end) / 2
    nearest = min(times, key=lambda t: abs(t - midpoint))
    if abs(nearest - midpoint) > EVIDENCE_SNAP_SECONDS:
        return start, end
    return min(start, nearest), max(end, nearest)


def _to_pipeline_events(raw_events: list[dict], frames: list[dict], video_id: str | None) -> list[dict]:
    ordered = sorted(frames, key=lambda frame: float(frame.get("timestamp") or 0.0))
    results = []
    for raw in raw_events:
        try:
            start = float(raw.get("start_time"))
            end = float(raw.get("end_time"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        start, end = _snap_to_frames(start, end, ordered)
        description = str(raw.get("evidence_description") or "").strip()
        evidence = [f"{start:.2f}초: {description}"] if description else []
        detail = _detail_line(raw)
        if detail:
            evidence.append(detail)
        event = {**raw, "start_time": start, "end_time": end, "evidence": evidence}
        normalized = _normalize_behavior_event(event=event, group=ordered, representative_frames=[])
        if normalized:
            results.append(normalized)
    return results


def extract_video_events(
    video_path: str,
    indexed_frames: list[dict],
    video_id: str | None,
    limit: int = 20,
) -> list[dict] | None:
    """영상 직접 입력으로 이벤트를 뽑는다. 호출 실패나 형식 오류는 None, 이벤트가 없으면 빈 목록."""
    if not indexed_frames:
        return None
    text = _call_gemini_video(video_path)
    raw_events = parse_events(text) if text else None
    if raw_events is None:
        return None
    events = _to_pipeline_events(raw_events, indexed_frames, video_id)
    events = _deduplicate_behavior_events(events)
    events.sort(
        key=lambda item: (
            float(item.get("interestingness") or 0.0),
            float(item.get("confidence") or 0.0),
            float(item.get("duration") or 0.0),
        ),
        reverse=True,
    )
    return events[:limit]


def _write_shadow(video_id: str | None, video_path: str, events: list[dict] | None) -> None:
    directory = Path(os.getenv("VIDEO_EVENT_SHADOW_DIR", "/tmp/video_event_shadow"))
    try:
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{video_id or 'unknown'}__{Path(video_path).stem}.json"
        payload = {"video_id": video_id, "chunk": Path(video_path).name, "events": events}
        (directory / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as exc:
        print(f"영상 이벤트 섀도 결과 저장 실패: {type(exc).__name__}", flush=True)


def extract_events_for_chunk(
    indexed_frames: list[dict],
    video_id: str | None,
    video_path: str | None,
    limit: int = 20,
) -> list[dict]:
    """BEHAVIOR_EVENT_SOURCE에 따라 이벤트 추출 방식을 고른다."""
    source = event_source()
    if source == "frames" or not video_path:
        return extract_behavior_events(indexed_frames=indexed_frames, video_id=video_id, limit=limit)

    if source == "video":
        events = extract_video_events(video_path, indexed_frames, video_id, limit)
        if events is None:
            print("영상 이벤트 추출 실패, 프레임 방식으로 대체합니다.", flush=True)
            return extract_behavior_events(indexed_frames=indexed_frames, video_id=video_id, limit=limit)
        return events

    frame_events = extract_behavior_events(indexed_frames=indexed_frames, video_id=video_id, limit=limit)
    try:
        _write_shadow(video_id, video_path, extract_video_events(video_path, indexed_frames, video_id, limit))
    except Exception as exc:  # 섀도 비교가 서비스 처리를 막으면 안 된다
        print(f"영상 이벤트 섀도 실행 실패: {type(exc).__name__}", flush=True)
    return frame_events
