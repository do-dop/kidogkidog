import subprocess
import sys
import time
from pathlib import Path

import clip
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Kidogkidog 🐾",
    page_icon="🐾",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🐾 Kidogkidog</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">당신의 펫, 지금 뭐하고 있을까?</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# CLIP 모델 로드
# ─────────────────────────────────────────
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def load_image_features(frame_paths, model, preprocess, device):
    image_tensors = []
    for path in frame_paths:
        image = preprocess(Image.open(path).convert("RGB"))
        image_tensors.append(image)
    image_batch = torch.stack(image_tensors).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        image_features = F.normalize(image_features, dim=-1)
    return image_features


def search_top_k(query, frame_paths, model, preprocess, device, top_k=3):
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
        results.append({
            "path": frame_path,
            "score": score.item(),
            "name": frame_path.name,
        })
    return results


def get_frame_count():
    frames_dir = Path("pipeline/frames")
    if not frames_dir.exists():
        return 0
    return len(list(frames_dir.glob("*.jpg")))


# ─────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["🔧 시스템 처리 현황", "🐾 검색 서비스"])

# ─────────────────────────────────────────
# 탭 1: 시스템 처리 현황
# ─────────────────────────────────────────
with tab1:
    st.subheader("📹 영상 처리 파이프라인")
    st.info("💡 Celery worker가 실행 중이어야 합니다: `celery -A pipeline.tasks worker --loglevel=info --pool=solo`")

    if st.button("🎬 펫캠 영상 수신 시작", use_container_width=True):

        with st.status("파이프라인 실행 중...", expanded=True) as status:

            # 1. Edge Simulator 실행
            st.write("🚀 Edge Simulator 시작...")
            process = subprocess.Popen(
                [sys.executable, "simulator/edge_simulator.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Edge Simulator 로그 실시간 출력
            chunk_count = 0
            for line in process.stdout:
                line = line.strip()
                if "청크 생성 완료" in line:
                    chunk_count = int(line.split("총 ")[1].split("개")[0])
                    st.write(f"✅ {line}")
                elif "업로드 성공" in line:
                    st.write(f"☁️ {line}")
                elif "서버 응답" in line:
                    st.write(f"📡 {line}")
                elif "대기 중" in line:
                    st.write(f"⏳ {line}")

            process.wait()
            st.write(f"✅ Edge Simulator 완료! 총 {chunk_count}개 청크 전송")

            # 2. Celery 처리 대기
            st.write("⚙️ Celery worker 처리 중... (프레임 추출 대기)")
            prev_count = get_frame_count()
            timeout = 0

            while timeout < 600:  # 최대 10분 대기
                time.sleep(10)
                timeout += 10
                current_count = get_frame_count()

                if current_count > prev_count:
                    st.write(f"🖼️ 프레임 추출 중... ({current_count}개)")
                    prev_count = current_count

                # 처리 완료 판단: 30초 동안 변화 없으면 완료로 간주
                if timeout > 30 and current_count == prev_count and current_count > 0:
                    break

            final_count = get_frame_count()
            st.write(f"✅ 프레임 추출 완료! 총 {final_count}개")
            status.update(label="✅ 파이프라인 완료!", state="complete")

        # 처리 결과 요약
        col1, col2 = st.columns(2)
        with col1:
            st.metric("전송된 청크 수", f"{chunk_count}개")
        with col2:
            st.metric("추출된 프레임", f"{get_frame_count()}개")

        # 추출된 프레임 갤러리
        st.subheader("🖼️ 추출된 프레임")
        frames_dir = Path("pipeline/frames")
        if frames_dir.exists():
            frame_files = sorted(frames_dir.glob("*.jpg"))[:12]
            if frame_files:
                cols = st.columns(4)
                for i, frame_path in enumerate(frame_files):
                    with cols[i % 4]:
                        img = Image.open(frame_path)
                        timestamp = frame_path.stem.replace("frame_", "")
                        st.image(img, caption=f"⏱️ {timestamp}초", use_container_width=True)

# ─────────────────────────────────────────
# 탭 2: 고객용 검색 서비스
# ─────────────────────────────────────────
with tab2:
    st.subheader("🔍 펫 행동 검색")
    st.write("자연어 문장을 입력하면 저장된 프레임 중 가장 비슷한 장면 Top-3를 보여줍니다.")

    frames_dir = Path("pipeline/frames")

    if not frames_dir.exists():
        st.error("`pipeline/frames` 폴더가 없습니다.")
    else:
        frame_paths = sorted(frames_dir.glob("*.jpg"))

        if not frame_paths:
            st.error("저장된 프레임이 없습니다. 탭 1에서 먼저 영상을 처리해주세요!")
        else:
            st.info(f"현재 저장된 프레임 수: {len(frame_paths)}")

            query = st.text_input(
                "",
                placeholder="예: a dog eating food",
                label_visibility="collapsed"
            )

            col1, col2 = st.columns([1, 5])
            with col1:
                search_btn = st.button("🔍 검색", use_container_width=True)

            if search_btn and query:
                with st.spinner("CLIP으로 프레임 검색 중..."):
                    model, preprocess, device = load_clip_model()
                    results = search_top_k(query, frame_paths, model, preprocess, device, top_k=3)

                st.subheader("🏆 Top-3 검색 결과")
                cols = st.columns(len(results))
                for col, result in zip(cols, results):
                    with col:
                        st.image(str(result["path"]), caption=result["name"], use_container_width=True)
                        st.write(f"⭐ score: {result['score']:.4f}")

                st.subheader("✨ 최고 유사도 결과")
                best = results[0]
                st.write(f"**Query**: {query}")
                st.write(f"**Best frame**: {best['name']}")
                st.write(f"**Score**: {best['score']:.4f}")

            elif search_btn and not query:
                st.warning("검색어를 입력해주세요!")

            st.markdown("---")
            st.markdown("**💡 이런 것들을 검색해보세요:**")
            cols = st.columns(3)
            with cols[0]:
                st.info("🐕 a dog eating food")
            with cols[1]:
                st.info("😺 a cat sleeping")
            with cols[2]:
                st.info("🎾 a pet playing with toy")