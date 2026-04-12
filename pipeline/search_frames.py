from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image


def load_image_features(
    frame_paths: list[Path], model, preprocess, device: str
) -> torch.Tensor:
    image_tensors = []

    for path in frame_paths:
        image = preprocess(Image.open(path).convert("RGB"))
        image_tensors.append(image)

    image_batch = torch.stack(image_tensors).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        image_features = F.normalize(image_features, dim=-1)

    return image_features


def main() -> None:
    frames_dir = Path("pipeline/frames")

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"프레임 폴더가 없습니다: {frames_dir}\n"
            "먼저 motion_test.py를 실행해서 프레임을 저장하세요."
        )

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(
            f"{frames_dir} 안에 jpg 프레임이 없습니다.\n"
            "motion_test.py 실행 결과를 확인하세요."
        )

    query = input("검색할 문장을 입력하세요: ").strip()
    if not query:
        raise ValueError("검색 문장이 비어 있습니다.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)

    image_features = load_image_features(frame_paths, model, preprocess, device)

    text_tokens = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

        similarities = text_features @ image_features.T
        similarities = similarities.squeeze(0)

    top_k = min(3, len(frame_paths))
    top_scores, top_indices = torch.topk(similarities, k=top_k)

    print("\n=== All Search Results ===")
    for path, score in zip(frame_paths, similarities):
        print(f"{path.name:<25} : {score.item():.4f}")

    print("\n=== Top 3 Matches ===")
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        print(
            f"{rank}. {frame_paths[idx].name} "
            f"(score: {score.item():.4f})"
        )

    best_idx = top_indices[0].item()
    print("\n=== Best Match ===")
    print(f"Query      : {query}")
    print(f"Best frame : {frame_paths[best_idx]}")
    print(f"Score      : {top_scores[0].item():.4f}")


if __name__ == "__main__":
    main()