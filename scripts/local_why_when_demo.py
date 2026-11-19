"""줄 당기기 영상(smartbench_0949)으로 "왜/언제 물건이 떨어졌어?"를 로컬에서 확인한다.

1) 영상에서 15.5~29.5초 프레임을 1초 간격으로 뽑는다(스테이징이 저장한 프레임 시각과 같다).
2) 서비스와 같은 코드로 12초 창을 나누고 장면 변화 감지(Gemini)를 돌린다.
3) 나온 이벤트로 순서/장면 질의(run_sequence_query)를 돌려 답변을 본다.

스테이징 DB·워커는 건드리지 않는다. 필요한 것: ffmpeg, python3.11, GEMINI_API_KEY(환경변수 또는 프로젝트 .env).

    python3.11 scripts/local_why_when_demo.py                     # 스테이징 media에서 영상을 받아 사용
    python3.11 scripts/local_why_when_demo.py --video my.mp4      # 내 영상 파일 사용
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

STAGING_VIDEO = (
    "https://kidog-api-staging-ptiwpgodca-du.a.run.app/media/s3"
    "?key=chunks/smartbench_0949/smartbench_0949_000.mp4"
)
QUESTIONS = (
    "물건들이 왜 떨어졌어?",
    "물건들이 떨어진 건 언제야?",
    "물건들이 왜, 언제 떨어졌어?",
)


def load_key_from_env_file() -> None:
    if os.getenv("GEMINI_API_KEY"):
        return
    env_file = PROJECT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")


def extract_frames(video: Path, out_dir: Path, start: float, seconds: float) -> list[dict]:
    pattern = str(out_dir / "f_%03d.jpg")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-t", str(seconds), "-i", str(video),
         "-vf", "fps=1", pattern],
        check=True,
    )
    files = sorted(out_dir.glob("f_*.jpg"))
    return [
        {"timestamp": start + index, "frame_path": str(path), "motion_detected": True,
         "motion_start": start, "motion_end": start + seconds, "segment_id": 0}
        for index, path in enumerate(files)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, help="영상 파일. 없으면 스테이징 media에서 받는다")
    parser.add_argument("--start", type=float, default=15.5, help="분석할 구간 시작(초)")
    parser.add_argument("--seconds", type=float, default=14.0, help="분석할 구간 길이(초)")
    args = parser.parse_args()

    load_key_from_env_file()
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY가 없습니다. 환경변수로 넣거나 프로젝트 .env에 적어 주세요.")

    from pipeline.events import scene_change_detector as detector
    from pipeline.events.behavior_event_extractor import _select_representative_frames, _split_long_group
    from pipeline.search.sequence_query import run_sequence_query

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        video = args.video
        if video is None:
            video = tmp_dir / "video.mp4"
            print("스테이징에서 영상 받는 중…")
            urllib.request.urlretrieve(STAGING_VIDEO, video)
        frames = extract_frames(video, tmp_dir, args.start, args.seconds)
        print(f"프레임 {len(frames)}장 ({frames[0]['timestamp']:.1f}~{frames[-1]['timestamp']:.1f}초)\n")

        scene_events = []
        for window in _split_long_group(frames):
            sample = _select_representative_frames(window, max_count=6)
            times = [f["timestamp"] for f in sample]
            found = detector.detect_scene_change(sample)
            print(f"[장면 변화 감지] 창 {window[0]['timestamp']:.1f}~{window[-1]['timestamp']:.1f}초, 대표 프레임 {times}")
            print("   →", found["summary"] if found else "변화 없음", f"({found['start_time']}~{found['end_time']}초)" if found else "")
            if found:
                scene_events.append(found)

    if not scene_events:
        sys.exit("\n장면 변화 이벤트가 없어서 질의 단계는 건너뜁니다.")

    events = []
    for index, found in enumerate(scene_events):
        # source_frames의 키는 서비스가 영상 안 시각을 계산할 때 쓰는 형식(청크 번호 000 + 프레임 시각)이다.
        events.append({
            "id": 100 + index, "video_id": "demo", "start_time": found["start_time"], "end_time": found["end_time"],
            "subject": found["target_object"], "target_object": found["target_object"], "subject_type": "scene",
            "action": found["summary"], "summary": found["summary"],
            "confidence": found["confidence"], "interestingness": found["interestingness"],
            "source_frames": [{"frame_id": f"f{index}", "timestamp": found["end_time"],
                               "s3_key": f"frames/demo/demo_000_frame_{found['end_time']:.2f}.jpg"}],
        })

    print("\n[질의 결과] (정답: 줄을 당겨서 24~30초 부근에 물건이 떨어짐)")
    for question in QUESTIONS:
        result = run_sequence_query(question, video_id="demo", events=events)
        print(f"\nQ. {question}\nA. {result.get('answer')}")


if __name__ == "__main__":
    main()
