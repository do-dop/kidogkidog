from pathlib import Path

import cv2


def main() -> None:
    video_path = Path("pipeline/test_video.mp4")
    output_dir = Path("pipeline/frames")

    if not video_path.exists():
        raise FileNotFoundError(
            f"영상 파일이 없습니다: {video_path}\n"
            "pipeline 폴더 안에 test_video.mp4 파일을 넣어주세요."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("영상 파일을 열 수 없습니다.")

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False
    )

    frame_count = 0
    motion_count = 0

    warmup_frames = 60
    motion_threshold = 30000
    min_area = 2500

    last_motion_time = -999
    cooldown_sec = 2.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        fgmask = fgbg.apply(blurred)

        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                motion_score += area

        if frame_count <= warmup_frames:
            continue

        if motion_score > motion_threshold:
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            if timestamp_sec - last_motion_time >= cooldown_sec:
                motion_count += 1
                last_motion_time = timestamp_sec

                safe_time = f"{timestamp_sec:.2f}".replace(".", "_")
                save_path = output_dir / f"frame_{safe_time}s.jpg"

                cv2.imwrite(str(save_path), frame)

                print(
                    f"[Motion Detected] "
                    f"frame={frame_count}, "
                    f"time={timestamp_sec:.2f}s, "
                    f"score={motion_score:.2f}, "
                    f"saved={save_path}"
                )

    cap.release()

    print("\n=== Summary ===")
    print(f"Total frames: {frame_count}")
    print(f"Motion detected events: {motion_count}")
    print(f"Saved frames folder: {output_dir}")


if __name__ == "__main__":
    main()