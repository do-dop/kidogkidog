import subprocess
import sys
import time
import re
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import clip
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from pipeline.s3_uploader import download_bytes, download_video, object_exists
from pipeline.vector_store import (
    get_indexed_frames,
    get_indexed_video_ids,
    index_frames,
    search,
)

from db.metadata import ( 
    init_db,
    insert_search_log,
    upsert_user_frequent_query,
    get_user_top_queries,
    get_user_recent_queries,
)

FRAMES_DIR = PROJECT_ROOT / "pipeline" / "frames"
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
CLIP_DIR = Path("/tmp/kidogkidog_clips")
SOURCE_VIDEO_CACHE_DIR = Path("/tmp/kidogkidog_source_videos")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CHUNK_DURATION_SECONDS = 60
CLIP_LEAD_SECONDS = 3
CLIP_DURATION_SECONDS = 10

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
# DB 초기화 및 사용자 식별
# ─────────────────────────────────────────
init_db()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = f"anonymous-{uuid.uuid4()}"

user_id = st.session_state["user_id"]

if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""

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


def get_video_id(video_path):
    return Path(video_path).stem


def get_target_video_ids(source_path):
    source = Path(source_path)

    if not source.is_absolute():
        source = PROJECT_ROOT / source

    if source.is_file():
        return [source.stem]

    if source.is_dir():
        return sorted(
            path.stem for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )

    return [Path(source_path).stem]


def get_frame_paths(video_id=None):
    if not FRAMES_DIR.exists():
        return []

    if isinstance(video_id, list):
        frame_paths = []

        for item in video_id:
            frame_paths.extend((FRAMES_DIR / item).glob("*.jpg"))

        return sorted(frame_paths)

    if video_id:
        return sorted((FRAMES_DIR / video_id).glob("*.jpg"))

    return sorted(FRAMES_DIR.rglob("*.jpg"))


def get_video_ids():
    if not FRAMES_DIR.exists():
        return []

    return sorted(path.name for path in FRAMES_DIR.iterdir() if path.is_dir())


def get_frame_count(video_id=None):
    return len(get_indexed_frames(video_id))


@st.cache_data(ttl=900)
def get_s3_frame_bytes(s3_key):
    try:
        return download_bytes(s3_key)
    except Exception:
        return None


def get_frame_image_source(result):
    s3_key = result.get("s3_key")

    if s3_key:
        return get_s3_frame_bytes(s3_key)

    frame_path = result.get("frame_path")

    if frame_path and Path(frame_path).exists():
        return frame_path

    return None


def find_source_video(video_id):
    for extension in VIDEO_EXTENSIONS:
        video_path = VIDEO_DIR / f"{video_id}{extension}"

        if video_path.exists():
            return video_path

    SOURCE_VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for extension in VIDEO_EXTENSIONS:
        s3_key = f"videos/{video_id}{extension}"

        if not object_exists(s3_key):
            continue

        cached_video_path = SOURCE_VIDEO_CACHE_DIR / f"{video_id}{extension}"

        if not cached_video_path.exists():
            download_video(s3_key, str(cached_video_path))

        return cached_video_path

    return None


def get_chunk_index(frame_path):
    match = re.search(r"_(\d{3})_frame_", Path(frame_path).name)

    if not match:
        return 0

    return int(match.group(1))


def get_global_timestamp(result):
    chunk_index = get_chunk_index(result["frame_path"])
    return chunk_index * CHUNK_DURATION_SECONDS + float(result["timestamp"])


def extract_clip(result, lead_seconds=CLIP_LEAD_SECONDS, duration=CLIP_DURATION_SECONDS):
    video_id = result["video_id"]
    source_video = find_source_video(video_id)

    if source_video is None:
        return None

    global_timestamp = get_global_timestamp(result)
    start_time = max(global_timestamp - lead_seconds, 0)

    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CLIP_DIR / f"{video_id}_{start_time:.2f}_{duration}.mp4"

    if output_path.exists():
        return output_path

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_time:.2f}",
            "-i",
            str(source_video),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ],
        check=True,
    )

    return output_path


# ─────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔧 시스템 처리 현황", "🐾 검색 서비스", "🧪 영상 검색 테스트"])

# ─────────────────────────────────────────
# 탭 1: 시스템 처리 현황
# ─────────────────────────────────────────
with tab1:
    st.subheader("📹 영상 처리 파이프라인")
    st.info("💡 Celery worker가 실행 중이어야 합니다: `celery -A pipeline.tasks worker --loglevel=info --pool=solo`")

    video_path = st.text_input("처리할 영상 파일 또는 폴더", value="data/videos")

    if st.button("🎬 펫캠 영상 수신 시작", use_container_width=True):
        target_video_ids = get_target_video_ids(video_path)

        if not target_video_ids:
            st.error("처리할 영상 파일이 없습니다. 폴더에 mp4/mov/avi/mkv/m4v 파일을 넣어주세요.")
            st.stop()

        with st.status("파이프라인 실행 중...", expanded=True) as status:
            # 1. Edge Simulator 실행
            st.write("🚀 Edge Simulator 시작...")

            process = subprocess.Popen(
                [sys.executable, "simulator/edge_simulator.py", video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
            )

            # Edge Simulator 로그 실시간 출력
            chunk_count = 0

            for line in process.stdout:
                line = line.strip()

                if "청크 생성 완료" in line:
                    chunk_count += int(line.split("총 ")[1].split("개")[0])
                    st.write(f"✅ {line}")
                elif line.startswith("=== 영상"):
                    st.write(f"🎞️ {line}")
                elif "업로드 성공" in line:
                    st.write(f"☁️ {line}")
                elif "서버 응답" in line:
                    st.write(f"📡 {line}")
                elif "대기 중" in line:
                    st.write(f"⏳ {line}")

            process.wait()

            if process.returncode != 0:
                st.error("Edge Simulator 실행에 실패했습니다. 영상 경로와 FastAPI 서버 상태를 확인해주세요.")
                st.stop()

            st.write(f"✅ Edge Simulator 완료! 총 {chunk_count}개 청크 전송")

            # 2. Celery 처리 대기
            st.write("⚙️ Celery worker 처리 중... (프레임 추출 대기)")

            prev_count = get_frame_count(target_video_ids)
            timeout = 0

            while timeout < 600:
                time.sleep(10)
                timeout += 10

                current_count = get_frame_count(target_video_ids)

                if current_count > prev_count:
                    st.write(f"🖼️ 프레임 추출 중... ({current_count}개)")
                    prev_count = current_count

                # 처리 완료 판단: 30초 동안 변화 없으면 완료로 간주
                if timeout > 30 and current_count == prev_count and current_count > 0:
                    break

            final_count = get_frame_count(target_video_ids)

            st.write(f"✅ 프레임 추출 완료! 총 {final_count}개")
            st.write("✅ ChromaDB 인덱싱 확인 완료")

            status.update(label="✅ 파이프라인 완료!", state="complete")

        # 처리 결과 요약
        col1, col2 = st.columns(2)

        with col1:
            st.metric("전송된 청크 수", f"{chunk_count}개")

        with col2:
            st.metric("추출된 프레임", f"{get_frame_count(target_video_ids)}개")

        # 추출된 프레임 갤러리
        st.subheader("🖼️ 추출된 프레임")

        indexed_frames = get_indexed_frames(target_video_ids)[:12]

        if indexed_frames:
            cols = st.columns(4)

            for i, frame in enumerate(indexed_frames):
                with cols[i % 4]:
                    image_source = get_frame_image_source(frame)

                    if image_source:
                        st.image(
                            image_source,
                            caption=f"⏱️ {float(frame['timestamp']):.2f}초",
                            use_container_width=True,
                        )
                    else:
                        st.caption("만료된 이미지입니다.")


# ─────────────────────────────────────────
# 탭 2: 고객용 검색 서비스
# ─────────────────────────────────────────
with tab2:
    st.subheader("🔍 펫 행동 검색")
    st.write("영상을 선택하고 자연어 문장을 입력하면 저장된 프레임 중 가장 비슷한 장면을 보여줍니다.")

    video_ids = get_indexed_video_ids()

    if not video_ids:
        st.error("인덱싱된 프레임이 없습니다. 탭 1에서 먼저 영상을 처리해주세요!")
    else:
        selected_video = st.selectbox("검색할 영상", ["전체"] + video_ids)
        selected_video_id = None if selected_video == "전체" else selected_video

        indexed_frames = get_indexed_frames(selected_video_id)

        if not indexed_frames:
            st.error("선택한 영상에 저장된 프레임이 없습니다. worker 처리 상태를 확인해주세요!")
        else:
            st.info(f"현재 선택 범위의 인덱싱된 프레임 수: {len(indexed_frames)}")

            if st.button("🔄 로컬 프레임 인덱스 갱신"):
                with st.spinner("ChromaDB 인덱싱 중..."):
                    index_frames(str(FRAMES_DIR))

                st.success("인덱싱 완료")

            # 자주 찾는 검색어 추천
            top_queries = get_user_top_queries(user_id=user_id, limit=5)

            if top_queries:
                st.markdown("#### ⭐ 자주 찾는 검색어")

                cols = st.columns(len(top_queries))

                for i, item in enumerate(top_queries):
                    q = item["query"]
                    count = item["count"]

                    with cols[i]:
                        if st.button(f"{q} ({count})", key=f"top_query_{i}"):
                            st.session_state["search_query"] = q
                            st.rerun()

            with st.form("search_form"):
                query = st.text_input(
                    "검색어",
                    placeholder="예: a dog eating food",
                    key="search_query",
                    label_visibility="collapsed",
                )

                top_k = st.slider("검색 결과 수", min_value=1, max_value=10, value=3)

                search_btn = st.form_submit_button("🔍 검색", use_container_width=True)

            if search_btn and query:
                start_time = time.time()

                with st.spinner("ChromaDB에서 검색 중..."):
                    results = search(query, top_k=top_k, video_id=selected_video_id)

                latency_ms = int((time.time() - start_time) * 1000)
                result_count = len(results) if results else 0

                insert_search_log(
                    user_id=user_id,
                    query_raw=query,
                    video_id=selected_video_id,
                    top_k=top_k,
                    result_count=result_count,
                    latency_ms=latency_ms,
                )

                upsert_user_frequent_query(
                    user_id=user_id,
                    query_raw=query,
                )

                if not results:
                    st.warning("검색 결과가 없습니다. 프레임 인덱싱 상태를 확인해주세요.")
                else:
                    st.subheader(f"🏆 Top-{len(results)} 검색 결과")

                    cols = st.columns(min(len(results), 3))

                    for i, result in enumerate(results):
                        with cols[i % len(cols)]:
                            image_source = get_frame_image_source(result)

                            if image_source:
                                st.image(image_source, use_container_width=True)
                            else:
                                st.caption("만료된 이미지입니다.")

                            st.write(f"🎬 {result['video_id']}")

                            global_timestamp = get_global_timestamp(result)

                            st.write(f"⏱️ {global_timestamp:.2f}초")
                            st.write(f"⭐ score: {result['score']:.4f}")

                            try:
                                clip_path = extract_clip(result)

                                if clip_path:
                                    st.video(str(clip_path), start_time=0)
                                else:
                                    st.caption("원본 영상을 찾을 수 없습니다.")
                            except Exception as exc:
                                st.caption(f"클립 생성 실패: {exc}")

                    st.subheader("✨ 최고 유사도 결과")

                    best = results[0]

                    st.write(f"**Query**: {query}")
                    st.write(f"**Best frame**: {best['frame_id']}")
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


# ─────────────────────────────────────────
# 탭 3: 영상별 검색 테스트
# ─────────────────────────────────────────
with tab3:
    st.subheader("🧪 영상별 검색 테스트")
    st.write("사용자 맞춤 검색 로그와 분리해서, 선택한 영상의 ChromaDB 검색 결과만 확인합니다.")

    test_video_ids = get_indexed_video_ids()

    if not test_video_ids:
        st.error("인덱싱된 프레임이 없습니다. worker 처리 완료 후 다시 확인해주세요.")
    else:
        with st.form("video_search_test_form"):
            test_video = st.selectbox("테스트할 영상", ["전체"] + test_video_ids)
            test_query = st.text_input("검색어", placeholder="예: a dog eating food")
            test_top_k = st.slider("결과 수", min_value=1, max_value=10, value=3)
            test_search_btn = st.form_submit_button("검색 테스트", use_container_width=True)

        test_video_id = None if test_video == "전체" else test_video
        test_frames = get_indexed_frames(test_video_id)

        st.info(f"선택 범위의 인덱싱된 프레임 수: {len(test_frames)}")

        if not test_frames:
            st.warning("선택한 영상에 인덱싱된 프레임이 없습니다.")
        elif test_search_btn and not test_query:
            st.warning("검색어를 입력해주세요.")
        elif test_search_btn:
            with st.spinner("선택한 영상에서 검색 중..."):
                test_results = search(test_query, top_k=test_top_k, video_id=test_video_id)

            if not test_results:
                st.warning("검색 결과가 없습니다.")
            else:
                st.subheader(f"검색 결과 Top-{len(test_results)}")

                cols = st.columns(min(len(test_results), 3))

                for i, result in enumerate(test_results):
                    with cols[i % len(cols)]:
                        image_source = get_frame_image_source(result)

                        if image_source:
                            st.image(image_source, use_container_width=True)
                        else:
                            st.caption("이미지를 표시할 수 없습니다.")

                        timestamp = float(result["timestamp"])

                        st.write(f"영상: {result['video_id']}")
                        st.write(f"시간: {timestamp:.2f}초")
                        st.write(f"score: {result['score']:.4f}")

                        if result.get("s3_key"):
                            st.caption(f"S3: {result['s3_key']}")

                        st.caption(result["frame_id"])

                        try:
                            clip_path = extract_clip(result)

                            if clip_path:
                                st.video(str(clip_path), start_time=0)
                            else:
                                st.caption("원본 영상을 찾을 수 없어 10초 클립을 만들 수 없습니다.")
                        except Exception as exc:
                            st.caption(f"10초 클립 생성 실패: {exc}")