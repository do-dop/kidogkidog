import cv2
import os
from pathlib import Path
from pipeline.motion_detector import detect_motion


def extract_frames(video_path, output_dir="pipeline/frames", fps=0.5, frame_prefix=None):
    """
    motion 감지된 구간에서만 프레임 추출

    Args:
        video_path: 영상 파일 경로
        output_dir: 프레임 저장 폴더
        fps: 추출할 FPS (기본 0.5fps, 2초마다 1장)

    Returns:
        frame_list: [
            {
                "frame_path": ...,
                "timestamp": ...,
                "motion_start": ...,
                "motion_end": ...,
                "segment_id": ...
            },
            ...
        ]
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_prefix = frame_prefix or Path(video_path).stem

    # 1. motion 감지
    print("motion 감지 중...")
    motion_segments = detect_motion(video_path)
    print(f"motion 구간 {len(motion_segments)}개 감지됨")

    # 2. 프레임 추출
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if not video_fps:
        cap.release()
        raise ValueError(f"영상 FPS를 읽을 수 없습니다: {video_path}")

    frame_list = []

    for segment_id, (start, end) in enumerate(motion_segments):
        t = start

        while t <= end:
            current_time = round(t, 2)

            frame_idx = int(current_time * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 저장
            frame_filename = f"{frame_prefix}_frame_{current_time:.2f}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            frame_list.append({
                "frame_path": frame_path,
                "timestamp": current_time,
                "motion_start": round(start, 2),
                "motion_end": round(end, 2),
                "segment_id": segment_id,
            })

            t += (1.0 / fps)

    cap.release()
    print(f"총 {len(frame_list)}개 프레임 추출 완료!")
    return frame_list