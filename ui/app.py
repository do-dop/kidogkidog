from pathlib import Path

import clip
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image


@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def load_image_features(frame_paths: list[Path], model, preprocess, device: str):
    image_tensors = []

    for path in frame_paths:
        image = preprocess(Image.open(path).convert("RGB"))
        image_tensors.append(image)

    image_batch = torch.stack(image_tensors).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        image_features = F.normalize(image_features, dim=-1)

    return image_features


def search_top_k(query: str, frame_paths: list[Path], model, preprocess, device: str, top_k: int = 3):
    image_features = load_image_features(frame_paths, model, preprocess, device)

    text_tokens = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

        similarities = text_features @ image_features.T
        similarities = similarities.squeeze(0)

    top_k = min(top_k, len(frame_paths))
    top_scores, top_indices = torch.topk(similarities, k=top_k)

    results = []
    for idx, score in zip(top_indices, top_scores):
        frame_path = frame_paths[idx.item()]
        results.append(
            {
                "path": frame_path,
                "score": score.item(),
                "name": frame_path.name,
            }
        )

    return results


def main():
    st.set_page_config(page_title="Petcam Frame Search", layout="wide")
    st.title("🐶 Petcam Frame Search")
    st.write("자연어 문장을 입력하면 저장된 프레임 중 가장 비슷한 장면 Top-3를 보여줍니다.")

    frames_dir = Path("pipeline/frames")

    if not frames_dir.exists():
        st.error("`pipeline/frames` 폴더가 없습니다. 먼저 frame_extractor.py를 실행하세요.")
        return

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        st.error("저장된 프레임이 없습니다. 먼저 frame_extractor.py를 실행해서 프레임을 저장하세요.")
        return

    st.info(f"현재 저장된 프레임 수: {len(frame_paths)}")

    query = st.text_input(
        "검색 문장을 입력하세요",
        placeholder="예: a dog eating food"
    )

    if st.button("검색"):
        if not query.strip():
            st.warning("검색 문장을 입력해주세요.")
            return

        with st.spinner("CLIP으로 프레임 검색 중입니다..."):
            model, preprocess, device = load_clip_model()
            results = search_top_k(query, frame_paths, model, preprocess, device, top_k=3)

        st.subheader("Top-3 검색 결과")

        cols = st.columns(len(results))
        for col, result in zip(cols, results):
            with col:
                st.image(str(result["path"]), caption=result["name"], use_container_width=True)
                st.write(f"score: {result['score']:.4f}")

        st.subheader("최고 유사도 결과")
        best = results[0]
        st.write(f"**Query**: {query}")
        st.write(f"**Best frames**: {best['name']}")
        st.write(f"**Score**: {best['score']:.4f}")


if __name__ == "__main__":
    main()
