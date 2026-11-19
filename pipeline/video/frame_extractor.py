import cv2
import os
import subprocess
from pathlib import Path
from pipeline.video.motion_detector import detect_motion


def _extract_frames_with_ffmpeg(video_path, output_dir, frame_prefix, fps):
    """OpenCV가 코덱을 읽지 못하거나 motion 구간이 없을 때의 안전한 fallback."""
    output_pattern = str(Path(output_dir) / f"{frame_prefix}_frame_%06d.jpg")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        output_pattern,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    extracted_paths = sorted(Path(output_dir).glob(f"{frame_prefix}_frame_*.jpg"))
    interval = 1.0 / fps
    return [
        {
            "frame_path": str(frame_path),
            "timestamp": round(index * interval, 2),
            "motion_start": 0.0,
            "motion_end": round(max(0, (len(extracted_paths) - 1) * interval), 2),
            "segment_id": 0,
            # 움직임이 확인된 구간이 아니라 청크 전체를 뽑은 프레임이다.
            "motion_detected": False,
        }
        for index, frame_path in enumerate(extracted_paths)
    ]


def idle_sample_times(duration, motion_segments, interval, pad=1.0):
    """움직임 구간 밖에서 interval초마다 찍을 시각을 계산한다.

    모션 감지가 놓친 짧은 행동을 저빈도로라도 잡기 위한 보조 샘플이다.
    움직임 구간(앞뒤 pad초 포함)과 겹치는 시각은 이미 프레임이 있으므로 제외한다.
    interval이 0 이하이면 사용하지 않는다.
    """
    if not interval or interval <= 0 or not duration or duration <= 0:
        return []
    times = []
    t = 0.0
    while t <= duration:
        if not any(start - pad <= t <= end + pad for start, end in motion_segments):
            times.append(round(t, 2))
        t += interval
    return times


def _default_idle_interval():
    try:
        return float(os.getenv("IDLE_SAMPLE_INTERVAL_SECONDS", "2"))
    except ValueError:
        return 1.0


IDLE_GROUP_SECONDS = 12.0


def _idle_time_groups(times, interval):
    """Do not put idle frames on opposite sides of a motion segment in one VLM window."""
    groups = []
    for timestamp in times:
        if (not groups or timestamp - groups[-1][-1] > interval * 1.5
                or timestamp - groups[-1][0] >= IDLE_GROUP_SECONDS):
            groups.append([timestamp])
        else:
            groups[-1].append(timestamp)
    return groups


def extract_frames(video_path, output_dir="pipeline/frames", fps=1.0, frame_prefix=None, idle_sample_interval=None):
    """
    움직임 구간은 촘촘히, 감지 밖 구간도 보조 프레임으로 추출

    Args:
        video_path: 영상 파일 경로
        output_dir: 프레임 저장 폴더
        fps: 움직임 구간 추출 FPS (기본 1fps, 1초마다 1장)
        idle_sample_interval: motion 구간 밖에서 N초마다 1장씩 추가로 뽑는다.
            None이면 환경변수 IDLE_SAMPLE_INTERVAL_SECONDS(기본 2초)를 따른다.
            추가 프레임은 motion_detected=False이고 연속된 12초 이하 구간으로 묶인다.

    Returns:
        frame_list: [
            {
                "frame_path": ...,
                "timestamp": ...,
                "motion_start": ...,
                "motion_end": ...,
                "segment_id": ...,
                "motion_detected": True,  # 청크 전체 대체 추출이면 False
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

    if idle_sample_interval is None:
        idle_sample_interval = _default_idle_interval()

    if not motion_segments:
        # 정지 청크도 idle 간격을 따른다. 설정이 없으면 기존처럼 fps로 추출한다.
        fallback_fps = fps
        if idle_sample_interval > 0 and 1.0 / idle_sample_interval < fps:
            fallback_fps = 1.0 / idle_sample_interval
        print("motion 구간이 없거나 OpenCV가 영상을 읽지 못해 FFmpeg 전체 프레임 추출로 전환합니다.")
        frame_list = _extract_frames_with_ffmpeg(
            video_path=video_path,
            output_dir=output_dir,
            frame_prefix=frame_prefix,
            fps=fallback_fps,
        )
        print(f"FFmpeg fallback으로 총 {len(frame_list)}개 프레임 추출 완료!")
        return frame_list

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
                "motion_detected": True,
            })

            t += (1.0 / fps)

    if idle_sample_interval > 0:
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        clip_seconds = total_frames / video_fps if total_frames else 0
        idle_times = idle_sample_times(clip_seconds, motion_segments, idle_sample_interval)
        for group_index, times in enumerate(_idle_time_groups(idle_times, idle_sample_interval)):
            for current_time in times:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_time * video_fps))
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_path = os.path.join(output_dir, f"{frame_prefix}_frame_{current_time:.2f}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_list.append({
                    "frame_path": frame_path,
                    "timestamp": current_time,
                    "motion_start": times[0],
                    "motion_end": times[-1],
                    "segment_id": len(motion_segments) + group_index,
                    "motion_detected": False,
                })
        frame_list.sort(key=lambda item: item["timestamp"])

    cap.release()
    print(f"총 {len(frame_list)}개 프레임 추출 완료!")
    return frame_list
