from functools import lru_cache
import os
from pathlib import Path
from typing import Any, List, Set

from ultralytics import YOLO


DEFAULT_YOLO_MODEL = "yolov8s.pt"
YOLO_MODEL_ENV_VAR = "YOLO_MODEL"
DETECTION_CONFIDENCE_THRESHOLD = 0.4
OBJECT_LABELS_CONFIDENCE_THRESHOLD = 0.5


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

    YOLO_MODEL 환경변수로 모델을 바꿀 수 있으며 기본값은 yolov8s.pt다.
    """
    return YOLO(get_yolo_model_name())


def get_yolo_model_name() -> str:
    """현재 프로세스에서 사용할 모델 이름을 반환한다."""
    return os.getenv(YOLO_MODEL_ENV_VAR, DEFAULT_YOLO_MODEL).strip() or DEFAULT_YOLO_MODEL


def build_detection_result(
    detections: list[dict[str, Any]],
    target_labels: Set[str] | None = None,
    object_labels_threshold: float = OBJECT_LABELS_CONFIDENCE_THRESHOLD,
) -> dict[str, list]:
    """Raw detection 목록에서 legacy 라벨 목록을 구성한다.

    detections는 collection threshold 이상인 항목을 모두 보존하고,
    object_labels는 별도 운영 threshold 이상인 class만 unique하게 포함한다.
    """
    labels_filter = target_labels or TARGET_LABELS
    kept_detections = [
        {
            "label": str(item["label"]),
            "confidence": float(item["confidence"]),
            # xyxy, 원본 이미지 픽셀 좌표계를 사용한다.
            "bbox": [float(value) for value in item["bbox"]],
        }
        for item in detections
        if item.get("label") in labels_filter
    ]
    labels = sorted({
        item["label"]
        for item in kept_detections
        if item["confidence"] >= float(object_labels_threshold)
    })
    return {"object_labels": labels, "object_detections": kept_detections}


def detect_objects_with_details(
    frame_path: str,
    detection_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
    object_labels_threshold: float = OBJECT_LABELS_CONFIDENCE_THRESHOLD,
    target_labels: Set[str] | None = None,
) -> dict[str, list]:
    """프레임 객체 라벨과 confidence/bbox가 보존된 detection을 반환한다."""
    frame_path = str(Path(frame_path))
    labels_filter = target_labels or TARGET_LABELS

    try:
        model = get_yolo_model()
        results = model.predict(
            source=frame_path,
            conf=float(detection_threshold),
            verbose=False,
        )
        detections = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = names[class_id]
                if confidence < float(detection_threshold) or label not in labels_filter:
                    continue
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [float(value) for value in box.xyxy[0].tolist()],
                })
        return build_detection_result(
            detections,
            target_labels=labels_filter,
            object_labels_threshold=object_labels_threshold,
        )
    except Exception as exc:
        # YOLO 실패가 전체 인덱싱 실패로 이어지면 안 되므로 빈 결과 반환
        print(f"YOLO 객체 감지 실패: {frame_path} - {exc}", flush=True)
        return {"object_labels": [], "object_detections": []}


def detect_objects(
    frame_path: str,
    confidence_threshold: float = OBJECT_LABELS_CONFIDENCE_THRESHOLD,
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
    # 기존 호출부와의 호환성을 위해 labels-only 반환은 유지한다.
    result = detect_objects_with_details(
        frame_path,
        detection_threshold=min(
            DETECTION_CONFIDENCE_THRESHOLD,
            float(confidence_threshold),
        ),
        object_labels_threshold=float(confidence_threshold),
        target_labels=target_labels,
    )
    return result["object_labels"]
