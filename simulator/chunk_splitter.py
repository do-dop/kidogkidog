import subprocess
from pathlib import Path
from datetime import datetime


def split_video_into_chunks(video_path, chunk_duration=60, output_dir="simulator/chunks"):
    """
    영상을 5분(300초) 단위로 청크 분할

    Args:
        video_path: 원본 영상 경로
        chunk_duration: 청크 길이 (초, 기본 5분)
        output_dir: 청크 저장 폴더
    """
    video_path = Path(video_path)
    video_id = video_path.stem
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / video_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(run_dir / f"petcam_{run_id}_%03d.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),

        # 아이폰 MOV에 들어있는 위치정보/메타데이터 stream은 제외하고
        # 첫 번째 video stream과 audio stream만 사용
        "-map", "0:v:0",
        "-map", "0:a?",

        # 브라우저/Streamlit/OpenCV에서 잘 읽히도록 H.264 mp4로 변환
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",

        # 오디오는 있으면 AAC로 변환, 없으면 무시
        "-c:a", "aac",
        "-b:a", "128k",

        "-segment_time", str(chunk_duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        str(output_pattern),
    ]

    subprocess.run(cmd, check=True)

    chunks = sorted(run_dir.glob("*.mp4"))
    chunks = [str(chunk) for chunk in chunks]

    print(f"총 {len(chunks)}개 청크 생성 완료!")
    for chunk in chunks:
        print(f"  {chunk}")

    return chunks
