from functools import lru_cache
from pathlib import Path
from typing import List, Set

from ultralytics import YOLO


# 펫캠/동물 영상에서 우선 의미 있는 COCO 클래스만 남김
TARGET_LABELS: Set[str] = {
    "person",
    "dog",
    "cat",
    "bird",
    "horse",
    "sheep",
    "cow",
    "bear",
    "zebra",
    "giraffe",
    "bowl",
    "cup",
    "bottle",
    "chair",
    "couch",
    "bed",
    "dining table",
    "sports ball",
    "tv",
    "remote",
    "backpack",
    "handbag",
    "suitcase",
}


@lru_cache(maxsize=1)
def get_yolo_model():
    """
    YOLO 모델을 한 번만 로드해서 재사용한다.

    yolov8n.pt는 가장 가벼운 모델이라 Celery worker에서 먼저 붙이기 좋다.
    """
    return YOLO("yolov8n.pt")


def detect_objects(
    frame_path: str,
    confidence_threshold: float = 0.35,
    target_labels: Set[str] | None = None,
) -> List[str]:
    """
    단일 프레임 이미지에서 객체를 감지하고 라벨 리스트를 반환한다.

    Args:
        frame_path: 프레임 이미지 경로
        confidence_threshold: 최소 confidence 기준
        target_labels: 감지 대상으로 사용할 라벨 집합

    Returns:
        예: ["dog", "bowl", "couch"]
    """
    frame_path = str(Path(frame_path))
    labels_filter = target_labels or TARGET_LABELS

    try:
        model = get_yolo_model()
        results = model.predict(
            source=frame_path,
            conf=confidence_threshold,
            verbose=False,
        )

        detected_labels = set()

        for result in results:
            names = result.names

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = names[class_id]

                if confidence < confidence_threshold:
                    continue

                if label not in labels_filter:
                    continue

                detected_labels.add(label)

        return sorted(detected_labels)

    except Exception as exc:
        # YOLO 실패가 전체 인덱싱 실패로 이어지면 안 되므로 빈 리스트 반환
        print(f"YOLO 객체 감지 실패: {frame_path} - {exc}", flush=True)
        return []