"""반려동물이 아닌 장면 변화(소파가 넘어짐, 물건이 쓰러짐 등)를 앞뒤 두 프레임 비교로 찾는다.

행동 추출 프롬프트는 반려동물에 집중해서, 반려동물이 화면에서 사라질 때 같이 일어난 장면 변화를 놓친다.
그래서 분석 구간의 첫 프레임과 끝 프레임만 따로 보내 "물건이 달라졌는가"만 묻는다. 이미지가 두 장뿐이라 싸고 질문이 좁다.

- Gemini REST API를 httpx로 직접 호출한다(새 의존성 없음).
- GEMINI_API_KEY가 없거나 SCENE_CHANGE_DETECTION=0이면 아무것도 하지 않는다.
- 호출 실패(429 할당량 초과, 네트워크 오류, 잘못된 응답)는 예외를 내지 않고 None을 돌려준다.
  행동 추출 전체가 이 보조 호출 때문에 실패하면 안 되기 때문이다.
"""

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_SCENE_CHANGE_MODEL = "gemini-3.5-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT_SECONDS = 30.0
# 429 등 일시적 실패 때 한 번 더 시도한다. 그래도 실패하면 이 구간은 건너뛴다.
MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 5.0
SCENE_CHANGE_INTERESTINGNESS = 0.8

_PROMPT = """너는 펫캠 영상 분석 도우미다.

아래 이미지들은 같은 장면에서 시간 순서대로 가져온 프레임이다. 반려동물과 사람은 무시하고, 장면 속 가구·문·물건의 모양이나 위치가 앞뒤 프레임에서 달라졌는지만 판단해라.

규칙:
- 반려동물이나 사람이 움직이거나 나타나거나 사라지는 것은 변화가 아니다.
- 카메라 밝기, 초점, 화질, 자막 변화는 변화가 아니다.
- 카메라가 흔들리거나 화면 구도(보이는 범위, 확대)가 달라져서 물건이 다르게 보이는 것은 변화가 아니다. 벽·창문·문틀 같은 배경 물건의 위치가 같이 바뀌었다면 카메라가 움직인 것이다. 앞 프레임에 없던 물건이 뒤 프레임에 보이더라도 시야가 달라져서일 수 있으니 이때도 false로 한다.
- 그 물건이 앞 프레임과 뒤 프레임에서 분명히 다른 모양이거나 다른 자리에 있을 때만 changed를 true로 한다. 조금이라도 불확실하면 false로 한다.
- changed가 true이면 before에는 앞 프레임에서의 모습, after에는 뒤 프레임에서의 모습을 보이는 대로 쓴다.
- 앞/뒤/좌/우/위/아래 같은 방향은 before와 after를 비교해서 분명할 때만 summary에 쓴다. 분명하지 않으면 방향 없이 "넘어졌어요", "움직였어요"처럼 쓴다.
- summary는 보호자에게 말해 주듯 쉬운 해요체 한 문장으로 쓴다. 예: "화분이 옆으로 쓰러지면서 흙이 바닥에 쏟아졌어요". "확인됨", "관찰됨" 같은 보고서 표현은 쓰지 않는다.
- 프레임에 보이지 않는 것은 쓰지 않는다. 반려동물이 같이 움직였는지가 보이지 않으면 쓰지 않는다.
- JSON 객체만 출력한다. markdown 코드블록을 쓰지 마라.

출력 형식:
{"changed": false}
또는
{"changed": true, "object": "물건 이름", "before": "앞 프레임 모습", "after": "뒤 프레임 모습", "summary": "한 문장", "confidence": 0.0}
"""


def is_enabled() -> bool:
    """키가 있고 끄지 않았을 때만 켠다."""
    if os.getenv("SCENE_CHANGE_DETECTION", "1").strip() == "0":
        return False
    return bool(os.getenv("GEMINI_API_KEY"))


def _model() -> str:
    return os.getenv("SCENE_CHANGE_MODEL", DEFAULT_SCENE_CHANGE_MODEL)


def _call_gemini(frames: list[dict]) -> str | None:
    """프레임 이미지를 Gemini에 보내 응답 텍스트를 받는다. 실패하면 None."""
    parts: list[dict[str, Any]] = [{"text": _PROMPT}]
    for frame in frames:
        path = Path(str(frame.get("frame_path") or ""))
        if not path.exists():
            return None
        parts.append({"text": f"다음 프레임의 영상 시각: {float(frame.get('timestamp') or 0.0):.2f}초"})
        parts.append({"inline_data": {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }})

    url = f"{GEMINI_BASE_URL}/models/{_model()}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0},
    }
    # 키는 URL이 아니라 헤더로 보낸다.
    headers = {"x-goog-api-key": os.getenv("GEMINI_API_KEY", "")}

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = httpx.post(url, json=body, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                candidates = response.json().get("candidates") or []
                texts = [
                    part.get("text", "")
                    for part in ((candidates[0].get("content") or {}).get("parts") or [])
                ] if candidates else []
                return "".join(texts).strip() or None
            retryable = response.status_code in (429, 500, 502, 503, 504)
            print(f"장면 변화 비교 호출 실패: HTTP {response.status_code}", flush=True)
            if not retryable:
                return None
        except (httpx.HTTPError, ValueError) as exc:
            print(f"장면 변화 비교 호출 실패: {type(exc).__name__}", flush=True)
        if attempt + 1 < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS)
    return None


def _parse(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _compare(first: dict, last: dict) -> dict | None:
    """두 프레임을 비교해 물건이 달라졌으면 {object, summary, before, after, confidence}를, 아니면 None을 돌려준다."""
    text = _call_gemini([first, last])
    parsed = _parse(text) if text else None
    if not parsed or parsed.get("changed") is not True:
        return None
    obj = str(parsed.get("object") or "").strip()
    summary = str(parsed.get("summary") or "").strip()
    if not obj or not summary:
        return None
    return {
        "object": obj,
        "summary": summary,
        "before": str(parsed.get("before") or "").strip(),
        "after": str(parsed.get("after") or "").strip(),
        "confidence": parsed.get("confidence"),
    }


def _pixel_change(path_a: str, path_b: str) -> float | None:
    """두 프레임의 평균 픽셀 차이(0~255). 읽을 수 없으면 None."""
    try:
        import cv2

        a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return None
        size = (160, 90)
        a, b = cv2.resize(a, size), cv2.resize(b, size)
        return float(cv2.absdiff(a, b).mean())
    except Exception:
        return None


def _locate_change(frames: list[dict]) -> tuple[int, int]:
    """변화가 일어난 인접 프레임 쌍. 연속 프레임 사이 픽셀 변화가 가장 큰 쌍을 고른다.

    모델에게 다시 묻지 않는다. 좁혀 들어간 쌍에서는 이미 변화가 끝난 뒤의 프레임을 보게 되어 설명이 흔들리기 때문이다.
    프레임을 읽을 수 없으면 구간 전체를 돌려준다.
    """
    last = len(frames) - 1
    if last <= 1:
        return 0, last
    scores = [
        _pixel_change(frames[i].get("frame_path"), frames[i + 1].get("frame_path"))
        for i in range(last)
    ]
    if any(score is None for score in scores):
        return 0, last
    best = max(range(last), key=lambda i: scores[i])
    return best, best + 1


def detect_scene_change(frames: list[dict]) -> dict | None:
    """구간의 첫·끝 프레임을 비교해 장면 변화 이벤트를 돌려준다. 변화가 없거나 실패하면 None.

    설명(무엇이 어떻게 달라졌는지)은 구간 전체의 첫·끝 프레임 비교 한 번으로 정한다. 변화 전후의 대비가 가장 뚜렷한 쌍이다.
    시각은 프레임 사이 픽셀 변화가 가장 큰 인접 쌍으로 좁힌다("소파가 언제 넘어졌어?"에 구간 전체가 아니라 그 순간으로 답하기 위해서다).
    변화가 없는 구간은 호출 1번, 변화가 있어도 호출은 1번이다.
    돌려주는 dict는 _normalize_behavior_event가 그대로 받는 형식이다(subject_type='scene').
    """
    if not is_enabled() or len(frames) < 2:
        return None

    first, last = frames[0], frames[-1]
    first_time = float(first.get("timestamp") or 0.0)
    last_time = float(last.get("timestamp") or 0.0)
    if last_time <= first_time:
        return None

    found = _compare(first, last)
    if found is None:
        return None

    low, high = _locate_change(frames)
    evidence = [
        f"{first_time:.2f}초: {found['before']}" if found["before"] else f"{first_time:.2f}초: 변화 전 프레임",
        f"{last_time:.2f}초: {found['after']}" if found["after"] else f"{last_time:.2f}초: 변화 후 프레임",
    ]
    return {
        "start_time": float(frames[low].get("timestamp") or 0.0),
        "end_time": float(frames[high].get("timestamp") or 0.0),
        "subject_type": "scene",
        "target_object": found["object"],
        "action": found["summary"],
        "summary": found["summary"],
        "repeat_count": 1,
        "confidence": found["confidence"],
        "interestingness": SCENE_CHANGE_INTERESTINGNESS,
        "evidence": evidence,
    }
