from pathlib import Path

import clip
import torch
from PIL import Image


def main() -> None:
    # 테스트 이미지 경로
    image_path = Path("pipeline/test_frame.jpg")

    if not image_path.exists():
        raise FileNotFoundError(
            f"이미지 파일이 없습니다: {image_path}\n"
            "pipeline 폴더 안에 test_frame.jpg 파일을 넣어주세요."
        )

    # GPU 있으면 cuda, 없으면 cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # CLIP 모델 로드
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 이미지 전처리
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 이미지 임베딩 추출
    with torch.no_grad():
        image_embedding = model.encode_image(image)

    print("Image embedding shape:", image_embedding.shape)
    print("Image embedding sample:", image_embedding[0][:10])


if __name__ == "__main__":
    main()