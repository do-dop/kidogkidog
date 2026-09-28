"""CLIP crop classification for YOLO cat/dog detections.

YOLO detections remain immutable evidence. This module adds a separate resolved
species view and builds the legacy object_labels field from that view.
"""

from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Callable

from PIL import Image

from pipeline.clip_embedder import embed_images, embed_text
from pipeline.yolo_detector import (
    DETECTION_CONFIDENCE_THRESHOLD,
    OBJECT_LABELS_CONFIDENCE_THRESHOLD,
)


SPECIES_LABELS = frozenset(("cat", "dog"))
SPECIES_RESOLVER_ENV_VAR = "ENABLE_CLIP_SPECIES_RESOLVER"
# LOOV selected 5% and 10% padding in two folds each. Ten percent had the
# strongest fixed-prompt pooled score (cat F1 1.000 vs .977 at 5%; dog F1 tied).
DEFAULT_BBOX_PADDING = 0.10
CAT_PROMPT = "a cat"
DOG_PROMPT = "a dog"


def clip_species_resolver_enabled() -> bool:
    """Return the rollout flag; default OFF preserves the current production path."""
    value = os.getenv(SPECIES_RESOLVER_ENV_VAR, "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


@lru_cache(maxsize=2)
def _text_embedding(prompt: str) -> tuple[float, ...]:
    return tuple(float(value) for value in embed_text(prompt))


def _dot(left, right) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _padded_crop_box(bbox, image_size, padding):
    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x = (x2 - x1) * float(padding)
    pad_y = (y2 - y1) * float(padding)
    box = (
        max(0, math.floor(x1 - pad_x)),
        max(0, math.floor(y1 - pad_y)),
        min(width, math.ceil(x2 + pad_x)),
        min(height, math.ceil(y2 + pad_y)),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


def resolve_species(
    image,
    detections,
    *,
    padding: float = DEFAULT_BBOX_PADDING,
    detection_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
    scorer: Callable | None = None,
) -> list[dict]:
    """Resolve each eligible cat/dog bbox independently using CLIP cosine scores.

    `scorer`, when supplied, is a test seam accepting a crop and returning
    `(cat_similarity, dog_similarity)`. Without it, the existing shared CLIP
    model/preprocess path is used. No full-frame species inference is performed.
    """
    owns_image = isinstance(image, (str, Path))
    source = Image.open(image).convert("RGB") if owns_image else image.convert("RGB")
    try:
        candidates = []
        crops = []
        for index, detection in enumerate(detections or []):
            if detection.get("label") not in SPECIES_LABELS:
                continue
            confidence = float(detection.get("confidence", 0.0))
            if confidence < float(detection_threshold):
                continue
            bbox = detection.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            crop_box = _padded_crop_box(bbox, source.size, padding)
            if crop_box is None:
                continue
            crops.append(source.crop(crop_box))
            candidates.append((index, detection))

        if not candidates:
            return []

        if scorer is not None:
            scores = [scorer(crop) for crop in crops]
        else:
            image_vectors = embed_images(crops)
            cat_vector = _text_embedding(CAT_PROMPT)
            dog_vector = _text_embedding(DOG_PROMPT)
            scores = [
                (_dot(vector, cat_vector), _dot(vector, dog_vector))
                for vector in image_vectors
            ]

        resolutions = []
        for (_, detection), (cat_similarity, dog_similarity) in zip(candidates, scores):
            cat_similarity = float(cat_similarity)
            dog_similarity = float(dog_similarity)
            clip_label = "cat" if cat_similarity >= dog_similarity else "dog"
            resolutions.append({
                "bbox": [float(value) for value in detection["bbox"]],
                "yolo_label": str(detection["label"]),
                "yolo_confidence": float(detection["confidence"]),
                "clip_label": clip_label,
                "clip_cat_similarity": cat_similarity,
                "clip_dog_similarity": dog_similarity,
                "resolved_label": clip_label,
            })
        return resolutions
    finally:
        if owns_image:
            source.close()


def resolve_detection_result(
    image,
    detection_result: dict,
    *,
    enabled: bool | None = None,
    padding: float = DEFAULT_BBOX_PADDING,
    scorer: Callable | None = None,
) -> dict:
    """Return raw YOLO detections plus resolver output and final object labels."""
    raw_detections = list(detection_result.get("object_detections") or [])
    legacy_labels = list(dict.fromkeys(detection_result.get("object_labels") or []))
    if enabled is None:
        enabled = clip_species_resolver_enabled()
    if not enabled:
        return {
            "object_detections": raw_detections,
            "species_resolution": [],
            "object_labels": legacy_labels,
        }

    try:
        resolutions = resolve_species(
            image,
            raw_detections,
            padding=padding,
            detection_threshold=DETECTION_CONFIDENCE_THRESHOLD,
            scorer=scorer,
        )
    except Exception as exc:
        # A resolver failure must not discard otherwise usable YOLO output.
        print(f"CLIP species resolver 실패: {exc}", flush=True)
        return {
            "object_detections": raw_detections,
            "species_resolution": [],
            "object_labels": legacy_labels,
        }

    labels = {
        str(detection["label"])
        for detection in raw_detections
        if detection.get("label") not in SPECIES_LABELS
        and float(detection.get("confidence", 0.0)) >= OBJECT_LABELS_CONFIDENCE_THRESHOLD
    }
    labels.update(item["resolved_label"] for item in resolutions)
    return {
        "object_detections": raw_detections,
        "species_resolution": resolutions,
        "object_labels": sorted(labels),
    }
