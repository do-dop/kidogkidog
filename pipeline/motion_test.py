from pathlib import Path

from motion_detector import detect_motion


def main() -> None:
    video_path = Path("pipeline/test_video.mp4")

    if not video_path.exists():
        raise FileNotFoundError(
            f"영상 파일이 없습니다: {video_path}\n"
            "pipeline 폴더 안에 test_video.mp4 파일을 넣어주세요."
        )

    segments = detect_motion(str(video_path))

    print(f"motion 감지된 구간 수: {len(segments)}")
    for start, end in segments:
        print(f"  {start}초 ~ {end}초")


if __name__ == "__main__":
    main()
