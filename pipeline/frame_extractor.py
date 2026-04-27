import cv2
import os
from pipeline.motion_detector import detect_motion


def extract_frames(video_path, output_dir="pipeline/frames", fps=1):
    """
    motion 감지된 구간에서만 프레임 추출

    Args:
        video_path: 영상 파일 경로
        output_dir: 프레임 저장 폴더
        fps: 추출할 FPS (기본 1fps)

    Returns:
        frame_list: [{"frame_path": ..., "timestamp": ...}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. motion 감지
    print("motion 감지 중...")
    motion_segments = detect_motion(video_path)
    print(f"motion 구간 {len(motion_segments)}개 감지됨")

    # 2. 프레임 추출
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_list = []

    for start, end in motion_segments:
        t = start
        while t <= end:
            frame_idx = int(t * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 저장
            frame_filename = f"frame_{t:.2f}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            frame_list.append({
                "frame_path": frame_path,
                "timestamp": t
            })

            t += (1.0 / fps)  # 1fps 간격

    cap.release()
    print(f"총 {len(frame_list)}개 프레임 추출 완료!")
    return frame_list


if __name__ == "__main__":
    frames = extract_frames("test_video_2.mp4")
    for f in frames[:5]:  # 처음 5개만 출력
        print(f"  {f['timestamp']:.2f}초 → {f['frame_path']}")