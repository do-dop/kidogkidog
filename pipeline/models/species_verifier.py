"""YOLO가 낮은 confidence로 잡은 dog 박스를 CLIP으로 다시 확인한다.

회색 담요 같은 정적 물체가 dog(0.50~0.60)으로 잡히는 오탐을 줄이는 용도다.
confidence 임계값만 올리면 실제 강아지도 같이 빠져서(0.6이면 약 19%),
YOLO가 확신하지 못한 박스에서만 박스 안쪽 이미지가 강아지처럼 보이는지 본다.
"""
import os
from functools import lru_cache

import clip
import torch
from PIL import Image

from pipeline.models.clip_embedder import device, load_clip_model

DOG_PROMPTS = ["a photo of a dog", "a photo of a puppy"]
OTHER_PROMPTS = [
    "a photo of a blanket",
    "a photo of a pillow",
    "a photo of a towel",
    "a photo of a cushion on a chair",
    "a photo of a stuffed toy",
    "a photo of a bed",
    "a photo of a pile of fabric",
]

# 2026-11-19 로컬 영상 2개(고양이 45박스 / 강아지 193박스) 측정으로 정한 값.
# YOLO conf < 0.65 이면서 CLIP이 강아지로 볼 확률이 0.7 미만이면 오탐으로 본다.
DEFAULT_MAX_YOLO_CONF = 0.65
DEFAULT_MIN_DOG_PROB = 0.7


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return default


def enabled() -> bool:
    return os.getenv("YOLO_VERIFY_DOG", "1").strip().lower() not in {"0", "false", "off", ""}


def needs_check(confidence: float) -> bool:
    return confidence < _env_float("YOLO_VERIFY_MAX_CONF", DEFAULT_MAX_YOLO_CONF)


@lru_cache(maxsize=1)
def _text_features():
    model, _ = load_clip_model()
    tokens = clip.tokenize(DOG_PROMPTS + OTHER_PROMPTS).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)


def dog_probability(frame_path: str, xyxy) -> float:
    """박스 안쪽 이미지가 강아지일 확률(강아지 프롬프트 합)을 반환한다."""
    model, preprocess = load_clip_model()
    x0, y0, x1, y1 = (int(v) for v in xyxy)
    crop = Image.open(frame_path).convert("RGB").crop((x0, y0, x1, y1))
    with torch.no_grad():
        image = preprocess(crop).unsqueeze(0).to(device)
        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
        probs = (100.0 * features @ _text_features().T).softmax(dim=-1)[0]
    return float(probs[: len(DOG_PROMPTS)].sum())


def looks_like_dog(frame_path: str, xyxy) -> bool:
    """확인에 실패하면 감지를 지우지 않는다(오탐 1건보다 누락이 더 나쁘다)."""
    try:
        return dog_probability(frame_path, xyxy) >= _env_float(
            "YOLO_VERIFY_MIN_DOG_PROB", DEFAULT_MIN_DOG_PROB
        )
    except Exception as exc:
        print(f"CLIP 강아지 확인 실패, 감지를 유지: {frame_path} - {exc}", flush=True)
        return True
