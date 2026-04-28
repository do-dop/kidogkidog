import subprocess
import os
from datetime import datetime


def split_video_into_chunks(video_path, chunk_duration=60, output_dir="simulator/chunks"):
    """
    영상을 5분(300초) 단위로 청크 분할

    Args:
        video_path: 원본 영상 경로
        chunk_duration: 청크 길이 (초, 기본 5분)
        output_dir: 청크 저장 폴더
    """
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pattern = f"{output_dir}/petcam_{date_str}_%03d.mp4"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(chunk_duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern
    ]

    subprocess.run(cmd, check=True)

    chunks = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".mp4")
    ])

    print(f"총 {len(chunks)}개 청크 생성 완료!")
    for chunk in chunks:
        print(f"  {chunk}")

    return chunks


if __name__ == "__main__":
    split_video_into_chunks("test_video_2.mp4")