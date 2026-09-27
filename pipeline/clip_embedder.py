import os
import time
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_preprocess = None


def load_clip_model():
    """CLIP 모델을 최초 임베딩 시점에 한 번만 로드"""
    global _model, _preprocess
    if _model is None or _preprocess is None:
        model_name = os.getenv("CLIP_MODEL", "ViT-L/14")
        cache_dir = Path(os.path.expanduser("~/.cache/clip"))
        start_time = time.time()
        print(
            f"CLIP 모델 로드 시작: model={model_name}, device={device}, cache={cache_dir}",
            flush=True,
        )
        _model, _preprocess = clip.load(model_name, device=device)
        elapsed = time.time() - start_time
        print(f"CLIP 모델 로드 완료: {elapsed:.1f}초", flush=True)
    return _model, _preprocess


def embed_image(image_path):
    """이미지 → CLIP 벡터"""
    model, preprocess = load_clip_model()
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(img)
        embedding = F.normalize(embedding, dim=-1)
    return embedding.cpu().numpy().tolist()[0]  # list로 반환


def embed_images(images):
    """PIL 이미지들을 기존 CLIP 인스턴스와 전처리로 한 번에 임베딩한다."""
    if not images:
        return []
    model, preprocess = load_clip_model()
    batch = torch.stack([
        preprocess(image.convert("RGB")) for image in images
    ]).to(device)
    with torch.no_grad():
        embeddings = F.normalize(model.encode_image(batch), dim=-1)
    return embeddings.cpu().numpy().tolist()


def embed_text(text):
    """텍스트 → CLIP 벡터"""
    model, _ = load_clip_model()
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = F.normalize(embedding, dim=-1)
    return embedding.cpu().numpy().tolist()[0]  # list로 반환
