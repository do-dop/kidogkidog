import argparse
import time
import os
import requests
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.chunk_splitter import split_video_into_chunks
from pipeline.s3_uploader import upload_video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def iter_video_paths(source_path):
    source = Path(source_path)
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    raise FileNotFoundError(f"영상 파일 또는 폴더가 없습니다: {source}")


def run_edge_simulator(video_path, server_url="http://localhost:8000", interval=300):
    """
    펫캠 역할 시뮬레이터
    - 영상을 5분 청크로 분할
    - 주기적으로 FastAPI 서버에 전송
    - S3에 업로드

    Args:
        video_path: 원본 영상 경로
        server_url: FastAPI 서버 URL
        interval: 전송 간격 (초, 기본 5분)
    """
    video_path = Path(video_path)
    video_id = video_path.stem

    print("Edge Simulator 시작!")
    print(f"영상: {video_path}")
    print(f"서버: {server_url}")
    print(f"전송 간격: {interval}초")
    print("-" * 40)

    # 1. 청크 분할
    chunks = split_video_into_chunks(video_path)

    # 2. 청크 순서대로 전송
    for i, chunk_path in enumerate(chunks):
        print(f"\n[{i + 1}/{len(chunks)}] {chunk_path} 전송 중...")

        # S3 업로드
        s3_key = f"chunks/{video_id}/{os.path.basename(chunk_path)}"
        upload_video(chunk_path, s3_key)

        # FastAPI 서버에 알림
        try:
            response = requests.post(
                f"{server_url}/upload",
                json={"chunk_path": s3_key}
            )
            print(f"서버 응답: {response.json()}")
        except Exception as e:
            print(f"서버 전송 실패: {e}")

        # 다음 청크까지 대기 (마지막 청크면 스킵)
        if i < len(chunks) - 1:
            print(f"{interval}초 대기 중...")
            time.sleep(interval)

    print("\nEdge Simulator 완료!")


def run_edge_simulator_source(source_path, server_url="http://localhost:8000", interval=300):
    video_paths = iter_video_paths(source_path)
    if not video_paths:
        raise FileNotFoundError(f"처리할 영상 파일이 없습니다: {source_path}")

    print(f"총 {len(video_paths)}개 영상 처리 시작")
    for index, video_path in enumerate(video_paths, start=1):
        print(f"\n=== 영상 {index}/{len(video_paths)}: {video_path} ===")
        run_edge_simulator(video_path, server_url=server_url, interval=interval)

    print("\n모든 영상 처리 요청 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", nargs="?", default="data/videos")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    run_edge_simulator_source(args.source_path, server_url=args.server_url, interval=args.interval)
