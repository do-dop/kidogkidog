import cv2


def detect_motion(
    video_path,
    threshold=30000,
    min_area=2500,
    warmup_seconds=2.0,
    sample_fps=2,
    resize_width=640,
):
    """
    영상에서 motion이 감지된 구간의 타임스탬프를 반환

    Args:
        video_path: 영상 파일 경로
        threshold: motion score 임계값 (contour 면적 합산 기준)
        min_area: 노이즈 제거용 최소 contour 면적
        warmup_seconds: 초기 배경 학습 구간 초 단위 (오탐 방지)
        sample_fps: motion 감지에 사용할 초당 프레임 수
        resize_width: motion 감지용 프레임 폭 (원본 저장 프레임에는 영향 없음)

    Returns:
        motion_segments: [(시작초, 종료초), ...] 리스트
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        cap.release()
        raise ValueError(f"영상 FPS를 읽을 수 없습니다: {video_path}")

    frame_interval = max(1, round(fps / sample_fps)) if sample_fps else 1
    source_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or resize_width
    area_scale = 1.0
    if resize_width and source_width > resize_width:
        area_scale = (resize_width / source_width) ** 2
    scaled_threshold = threshold * area_scale
    scaled_min_area = max(1, min_area * area_scale)

    print(
        "motion 감지 설정: "
        f"fps={fps:.2f}, sample_fps={sample_fps}, "
        f"frame_interval={frame_interval}, resize_width={resize_width}",
        flush=True,
    )

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

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps
        if resize_width and frame.shape[1] > resize_width:
            resize_height = round(frame.shape[0] * resize_width / frame.shape[1])
            frame = cv2.resize(frame, (resize_width, resize_height))

        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        fgmask = fgbg.apply(blurred)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

        if timestamp >= warmup_seconds:
            contours, _ = cv2.findContours(
                fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            motion_score = sum(
                area for c in contours if (area := cv2.contourArea(c)) > scaled_min_area
            )

            if motion_score > scaled_threshold:
                motion_frames.append(round(timestamp, 2))

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
