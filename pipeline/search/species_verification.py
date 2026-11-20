"""Recheck a cat label against the already indexed image embedding."""

from functools import lru_cache

import numpy as np

from pipeline.models.clip_embedder import embed_text
from pipeline.storage.vector_store import get_collection


# On the reviewed long.mov, all 17 YOLO cat labels were dogs (largest margin
# +0.0026). In the reviewed smartbench_0747 cat video, all 13 cat frames had
# margins >= +0.0394. Keep an undecided gap instead of declaring weak hits cats.
CAT_OVER_DOG_MARGIN = 0.02


@lru_cache(maxsize=1)
def _cat_dog_references():
    return (
        np.asarray(embed_text("a photo of a cat"), dtype=float),
        np.asarray(embed_text("a photo of a dog"), dtype=float),
    )


def confirmed_cat_frame_ids(frame_ids) -> set[str] | None:
    """Return confidently cat-like frame IDs, or None when verification fails."""
    requested = list(dict.fromkeys(str(frame_id) for frame_id in frame_ids if frame_id))
    if not requested:
        return set()
    try:
        cat, dog = _cat_dog_references()
        rows = get_collection().get(ids=requested, include=["embeddings"])
        confirmed = set()
        embeddings = rows.get("embeddings")
        for frame_id, embedding in zip(rows.get("ids") or [], embeddings if embeddings is not None else []):
            image = np.asarray(embedding, dtype=float)
            if float(np.dot(image, cat) - np.dot(image, dog)) >= CAT_OVER_DOG_MARGIN:
                confirmed.add(str(frame_id))
        return confirmed
    except Exception as exc:
        print(f"고양이 프레임 재검증 실패: {exc}", flush=True)
        return None


def has_confirmed_cat(video_id: str | None, frames=None) -> bool:
    if frames is None:
        from pipeline.storage.vector_store import get_indexed_frames
        try:
            frames = get_indexed_frames(video_id=video_id)
        except Exception as exc:
            print(f"고양이 추천 질문 근거 조회 실패: {exc}", flush=True)
            return False
    candidates = [
        frame.get("frame_id") for frame in frames
        if "cat" in {part.strip().lower() for part in str(frame.get("object_labels") or "").split(",")}
    ]
    return bool(confirmed_cat_frame_ids(candidates))
