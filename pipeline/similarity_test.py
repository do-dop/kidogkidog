from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image


def main() -> None:
    image_path = Path("pipeline/test_frame.jpg")

    if not image_path.exists():
        raise FileNotFoundError(
            f"이미지 파일이 없습니다: {image_path}\n"
            "pipeline 폴더 안에 test_frame.jpg 파일을 넣어주세요."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    texts = [
    "a dog eating from a bowl",
    "a dog sniffing near food",
    "a dog lying on the floor",
    "an empty room with no dog",
    "a dog standing in front of a sofa"
    ]

    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarities = image_features @ text_features.T

    print("\nText-Image Similarity Results")
    print("-" * 40)

    for text, score in zip(texts, similarities[0]):
        print(f"{text:<30} : {score.item():.4f}")

    best_idx = similarities[0].argmax().item()
    print("\nBest match:")
    print(f"{texts[best_idx]} ({similarities[0][best_idx].item():.4f})")


if __name__ == "__main__":
    main()