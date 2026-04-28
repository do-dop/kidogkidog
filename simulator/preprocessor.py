import subprocess
import os
from datetime import datetime


def standardize_video(input_path, output_dir="simulator/processed"):
    """
    영상 전처리: 해상도 720p, FPS 30으로 통일 + 파일명 컨벤션 적용

    Args:
        input_path: 원본 영상 경로
        output_dir: 저장 폴더
    """
    os.makedirs(output_dir, exist_ok=True)

    # 파일명 컨벤션: petcam_{날짜}_{시작시간}.mp4
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/petcam_{date_str}.mp4"

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", "scale=-2:720",  # 해상도 720p
        "-r", "30",  # FPS 30
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",  # 덮어쓰기
        output_path
    ]

    subprocess.run(cmd, check=True)
    print(f"전처리 완료: {output_path}")
    return output_path
