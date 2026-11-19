"""로컬 UI에서 수정한 "왜/언제" 질의를 확인하기 위한 중계 서버.

- 대부분의 요청(영상, 썸네일, 목록 등)은 스테이징 API로 그대로 전달한다.
- 순서/장면 질의(POST /query, is_sequence_query가 참)이고 영상이 없거나 smartbench_0949(_v2)이면
  스테이징 v2의 이벤트에 "줄을 당겨 물건이 떨어짐" 장면 이벤트를 더해 로컬 run_sequence_query로 답한다.
  (이 장면 이벤트는 수정한 감지기로 시작할 때 한 번 만든다. GEMINI_API_KEY 필요)
- 스테이징 DB·워커는 쓰지 않는다. 읽기 전용 요청만 전달한다.

    python3.11 scripts/local_ui_shim.py            # 127.0.0.1:8099
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

STAGING = "https://kidog-api-staging-ptiwpgodca-du.a.run.app"
DEMO_VIDEO_IDS = {"", "smartbench_0949", "smartbench_0949_v2"}
LOCAL_VIDEO_ID = "smartbench_0949_v2"
PORT = int(os.getenv("SHIM_PORT", "8099"))


def staging_json(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    request = urllib.request.Request(
        STAGING + path, data=data, headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    return json.load(urllib.request.urlopen(request, timeout=120))


def build_scene_event() -> dict:
    """수정한 감지기로 0949의 줄 당기기 장면 이벤트를 만든다."""
    import local_why_when_demo as demo

    demo.load_key_from_env_file()
    from pipeline.events import scene_change_detector as detector
    from pipeline.events.behavior_event_extractor import _select_representative_frames, _split_long_group

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        video = tmp_dir / "video.mp4"
        urllib.request.urlretrieve(demo.STAGING_VIDEO, video)
        frames = demo.extract_frames(video, tmp_dir, 15.5, 14.0)
        for window in _split_long_group(frames):
            found = detector.detect_scene_change(_select_representative_frames(window, max_count=6))
            if found:
                end = found["end_time"]
                return {
                    "id": 9001, "video_id": LOCAL_VIDEO_ID, "start_time": found["start_time"], "end_time": end,
                    "subject": found["target_object"], "target_object": found["target_object"], "subject_type": "scene",
                    "action": found["summary"], "summary": found["summary"], "confidence": found["confidence"],
                    "interestingness": found["interestingness"],
                    "source_frames": [{
                        "frame_id": f"{LOCAL_VIDEO_ID}__{LOCAL_VIDEO_ID}_000_frame_{end:.2f}",
                        "timestamp": end,
                        "s3_key": f"frames/{LOCAL_VIDEO_ID}/{LOCAL_VIDEO_ID}_000_frame_{end:.2f}.jpg",
                    }],
                }
    raise SystemExit("장면 변화 이벤트를 만들지 못했습니다.")


def stored_events() -> list[dict]:
    """스테이징 v2의 저장 이벤트(시간순 질의 결과)를 질의 코드가 쓰는 형식으로 바꾼다."""
    timeline = staging_json("/query", {"query": "반려동물 행동을 시간순으로 알려줘", "top_k": 10, "video_id": LOCAL_VIDEO_ID})
    events = []
    for item in timeline.get("results", []):
        events.append({
            "id": item["source_event_id"], "video_id": LOCAL_VIDEO_ID,
            "start_time": item["event_start"], "end_time": item["event_end"],
            "summary": item["summary"], "action": item["summary"], "subject": "반려동물",
            "confidence": 0.8, "interestingness": 0.7,
            "source_frames": [{"frame_id": item["frame_id"], "timestamp": item["timestamp"], "s3_key": item["s3_key"]}],
        })
    return events


class Shim(BaseHTTPRequestHandler):
    events: list[dict] = []

    def log_message(self, fmt, *args):  # 조용히
        pass

    def _forward(self, body: bytes | None):
        headers = {k: v for k, v in self.headers.items() if k.lower() in {"content-type", "range", "accept"}}
        request = urllib.request.Request(STAGING + self.path, data=body, headers=headers, method=self.command)
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except HTTPError as error:
            response = error
        payload = response.read()
        self.send_response(response.status if hasattr(response, "status") else response.code)
        for key, value in response.headers.items():
            if key.lower() not in {"transfer-encoding", "connection", "content-encoding", "content-length"}:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._forward(None)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path.split("?")[0] == "/query":
            try:
                request = json.loads(body or b"{}")
                handled = self._local_query(request)
                if handled:
                    return
            except Exception as exc:  # 로컬 처리 실패는 스테이징으로 넘긴다.
                print("로컬 처리 실패, 스테이징으로 전달:", type(exc).__name__, exc, flush=True)
        self._forward(body)

    def _local_query(self, request: dict) -> bool:
        from pipeline.search.sequence_query import is_sequence_query, run_sequence_query

        query = str(request.get("query") or "")
        video_id = str(request.get("video_id") or "")
        if video_id not in DEMO_VIDEO_IDS or not is_sequence_query(query):
            return False
        top_k = max(1, min(int(request.get("top_k") or 5), 10))
        result = run_sequence_query(query, video_id=LOCAL_VIDEO_ID, top_k=top_k, events=self.events)
        response = {
            "status": "ok", "query": query, "video_id": request.get("video_id"),
            "recording_date": request.get("recording_date"), "top_k": top_k,
            "answer": result["answer"], "results": result.get("results", []),
            "evidence_items": result.get("evidence_items", result.get("results", [])),
            "behavior_events": result.get("behavior_events", []), "used_llm": result.get("used_llm", False),
            "query_type": result.get("query_type", "sequence"),
            "total_event_count": result.get("total_event_count"),
            "excluded_event_count": result.get("excluded_event_count", 0),
            "sequence_intent": result.get("sequence_intent"), "truncated": result.get("truncated", False),
        }
        payload = json.dumps(response, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        print(f"[로컬 처리] {query} → {result['answer'][:60]}", flush=True)
        return True


def main() -> None:
    print("장면 변화 이벤트 만드는 중… (영상 다운로드 + Gemini 호출, 약 20초)", flush=True)
    scene = build_scene_event()
    print("장면 이벤트:", scene["summary"], f"({scene['start_time']}~{scene['end_time']}초)", flush=True)
    Shim.events = [*stored_events(), scene]
    print(f"이벤트 {len(Shim.events)}개 준비. http://127.0.0.1:{PORT} 에서 대기합니다.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Shim).serve_forever()


if __name__ == "__main__":
    main()
