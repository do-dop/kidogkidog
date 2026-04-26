import cv2


def detect_motion(video_path, threshold=30000, min_area=2500, warmup_frames=60):
    """
    영상에서 motion이 감지된 구간의 타임스탬프를 반환

    Args:
        video_path: 영상 파일 경로
        threshold: motion score 임계값 (contour 면적 합산 기준)
        min_area: 노이즈 제거용 최소 contour 면적
        warmup_frames: 초기 배경 학습 구간 (오탐 방지)

    Returns:
        motion_segments: [(시작초, 종료초), ...] 리스트
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    motion_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        fgmask = fgbg.apply(blurred)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

        if frame_idx >= warmup_frames:
            contours, _ = cv2.findContours(
                fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            motion_score = sum(
                area for c in contours if (area := cv2.contourArea(c)) > min_area
            )

            if motion_score > threshold:
                motion_frames.append(round(frame_idx / fps, 2))

        frame_idx += 1

    cap.release()

    motion_segments = []
    if motion_frames:
        start = motion_frames[0]
        prev = motion_frames[0]
        for t in motion_frames[1:]:
            if t - prev > 2.0:
                motion_segments.append((start, prev))
                start = t
            prev = t
        motion_segments.append((start, prev))

    return motion_segments


if __name__ == "__main__":
    for threshold in [500, 2000, 5000, 10000]:
        segments = detect_motion("test_video_2.mp4", threshold=threshold)
        print(f"threshold={threshold}: {len(segments)}개 구간")
        for start, end in segments:
            print(f"  {start}초 ~ {end}초")
        print()