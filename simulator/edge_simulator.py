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
from pipeline.storage.gcs_uploader import upload_video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def get_id_token(audience):
    """
    IAM 인증이 필요한 Cloud Run 서비스를 부르기 위한 Google ID 토큰을 만든다.

    IMPERSONATE_SERVICE_ACCOUNT가 있으면 로컬 ADC로 그 서비스 계정을 가장해 발급하고,
    없으면 현재 자격 증명(서비스 계정 키, 메타데이터 서버 등)으로 발급한다.
    """
    import google.auth
    from google.auth.transport.requests import Request

    target = os.getenv("IMPERSONATE_SERVICE_ACCOUNT")
    if target:
        from google.auth import impersonated_credentials

        source, _ = google.auth.default()
        credentials = impersonated_credentials.IDTokenCredentials(
            impersonated_credentials.Credentials(
                source_credentials=source,
                target_principal=target,
                target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
            ),
            target_audience=audience,
            include_email=True,
        )
        credentials.refresh(Request())
        return credentials.token

    from google.oauth2 import id_token

    return id_token.fetch_id_token(Request(), audience)


def build_upload_headers(server_url, iam_auth=False):
    headers = {}
    upload_token = os.getenv("UPLOAD_API_TOKEN")
    if upload_token:
        headers["X-Upload-Token"] = upload_token
    if iam_auth:
        headers["Authorization"] = f"Bearer {get_id_token(server_url.rstrip('/'))}"
    return headers


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


def run_edge_simulator(video_path, server_url="http://localhost:8000", interval=300, iam_auth=False):
    """
    펫캠 역할 시뮬레이터
    - 영상을 5분 청크로 분할
    - 주기적으로 FastAPI 서버에 전송
    - GCS에 업로드

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

        # GCS 업로드
        s3_key = f"chunks/{video_id}/{os.path.basename(chunk_path)}"
        upload_video(chunk_path, s3_key)

        # FastAPI 서버에 알림
        try:
            response = requests.post(
                f"{server_url}/upload",
                json={"chunk_path": s3_key},
                headers=build_upload_headers(server_url, iam_auth) or None,
                timeout=20,
            )
            response.raise_for_status()
            print(f"서버 응답: {response.json()}")
        except Exception as e:
            print(f"서버 전송 실패: {e}")
            raise

        # 다음 청크까지 대기 (마지막 청크면 스킵)
        if i < len(chunks) - 1:
            print(f"{interval}초 대기 중...")
            time.sleep(interval)

    print("\nEdge Simulator 완료!")


def run_edge_simulator_source(source_path, server_url="http://localhost:8000", interval=300, iam_auth=False):
    video_paths = iter_video_paths(source_path)
    if not video_paths:
        raise FileNotFoundError(f"처리할 영상 파일이 없습니다: {source_path}")

    print(f"총 {len(video_paths)}개 영상 처리 시작")
    for index, video_path in enumerate(video_paths, start=1):
        print(f"\n=== 영상 {index}/{len(video_paths)}: {video_path} ===")
        run_edge_simulator(video_path, server_url=server_url, interval=interval, iam_auth=iam_auth)

    print("\n모든 영상 처리 요청 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", nargs="?", default="data/videos")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument(
        "--iam-auth",
        action="store_true",
        help="IAM 인증이 필요한 Cloud Run 서비스용 Google ID 토큰을 Authorization 헤더로 보낸다.",
    )
    args = parser.parse_args()

    run_edge_simulator_source(
        args.source_path,
        server_url=args.server_url,
        interval=args.interval,
        iam_auth=args.iam_auth,
    )
