import time
import os
import requests
import sys
sys.path.insert(0, '.')

from simulator.chunk_splitter import split_video_into_chunks
from pipeline.s3_uploader import upload_video


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
    print(f"Edge Simulator 시작!")
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
        s3_key = f"chunks/{os.path.basename(chunk_path)}"
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


if __name__ == "__main__":
    # 테스트용: interval 5초로 설정
    run_edge_simulator("test_video_2.mp4", interval=5)