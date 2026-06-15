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
from pipeline.query_suggester import (
    suggest_queries,
    get_suggestion_events,
    get_suggestion_behavior_events,
)
from pipeline.query_analyzer import build_prompt_context
from pipeline.rag_chain import run_rag_query

from db.metadata import (
    init_db,
    insert_search_log,
    upsert_user_frequent_query,
    get_user_top_queries,
    get_user_recent_queries,
    get_scene_records,
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

if "pipeline_processing" not in st.session_state:
    st.session_state["pipeline_processing"] = False

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
    """
    처리 진행 상황 확인용 프레임 개수 조회.

    기존에는 ChromaDB의 get_indexed_frames()를 사용했지만,
    Celery worker가 ChromaDB에 쓰는 중 Streamlit이 동시에 읽으면
    ChromaDB sqlite 파일이 꼬일 수 있어 SQLite scenes 기준으로 조회한다.
    """
    try:
        if isinstance(video_id, list):
            return sum(
                len(get_scene_records(video_id=item))
                for item in video_id
            )

        return len(get_scene_records(video_id=video_id))

    except Exception as exc:
        print(f"프레임 개수 조회 실패: {exc}", flush=True)
        return 0


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
        st.session_state["pipeline_processing"] = True

        # 새 영상 처리 후 추천질문/행동이벤트를 다시 불러오기 위해 캐시 초기화
        for key in list(st.session_state.keys()):
            if key.startswith(("suggested_queries_", "behavior_events_", "scene_events_")):
                st.session_state.pop(key, None)

        target_video_ids = get_target_video_ids(video_path)

        if not target_video_ids:
            st.session_state["pipeline_processing"] = False
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
                st.session_state["pipeline_processing"] = False
                st.error("Edge Simulator 실행에 실패했습니다. 영상 경로와 FastAPI 서버 상태를 확인해주세요.")
                st.stop()

            st.write(f"✅ Edge Simulator 완료! 총 {chunk_count}개 청크 전송")

            # 2. Celery 처리 대기
            #
            # 주의:
            # 이 카운트는 ChromaDB가 아니라 SQLite scenes 기준으로 확인한다.
            # Celery가 ChromaDB에 쓰는 중 Streamlit이 ChromaDB를 읽으면
            # ChromaDB 인덱스가 꼬일 수 있기 때문이다.
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
                # 단, Celery worker가 실제로 모든 청크를 끝냈는지는 터미널 로그에서 확인하는 것이 가장 안전하다.
                if timeout > 30 and current_count == prev_count and current_count > 0:
                    break

            final_count = get_frame_count(target_video_ids)

            st.write(f"✅ 프레임 추출 완료! 총 {final_count}개")
            st.write("⏳ ChromaDB 안정성을 위해 검색 탭은 아직 비활성화 상태입니다.")
            st.write("Celery 터미널에서 모든 청크 처리 완료를 확인한 뒤 아래 활성화 버튼을 눌러주세요.")

            status.update(label="✅ 파이프라인 전송 완료!", state="complete")

        # 처리 결과 요약
        col1, col2 = st.columns(2)

        with col1:
            st.metric("전송된 청크 수", f"{chunk_count}개")

        with col2:
            st.metric("저장된 프레임 메타데이터", f"{get_frame_count(target_video_ids)}개")

        st.info(
            "프레임 갤러리와 검색 기능은 ChromaDB를 읽어야 하므로, "
            "Celery worker가 모든 청크 처리를 끝낸 뒤 활성화 버튼을 눌러주세요."
        )

    if st.session_state.get("pipeline_processing"):
        st.warning(
            "영상 처리 중에는 ChromaDB 안정성을 위해 검색 탭을 비활성화합니다. "
            "Celery 터미널에서 모든 청크 처리가 끝난 것을 확인한 뒤 아래 버튼을 눌러주세요."
        )

        if st.button("✅ Celery 처리 완료 확인 - 검색 기능 활성화", use_container_width=True):
            st.session_state["pipeline_processing"] = False

            # 새로 생성된 행동 이벤트/추천질문을 다시 불러오기 위해 캐시 초기화
            for key in list(st.session_state.keys()):
                if key.startswith(("suggested_queries_", "behavior_events_", "scene_events_")):
                    st.session_state.pop(key, None)

            st.success("검색 기능이 활성화되었습니다.")
            st.rerun()



# ─────────────────────────────────────────
# 탭 2: 고객용 검색 서비스
# ─────────────────────────────────────────
with tab2:
    if st.session_state.get("pipeline_processing"):
        st.info(
            "현재 영상 처리 중입니다. "
            "ChromaDB가 저장되는 동안 검색을 실행하면 인덱스가 꼬일 수 있어 검색 탭을 잠시 비활성화했습니다."
        )
        st.stop()

    st.subheader("🔍 펫 행동 검색")
    st.write("영상을 선택하고 자연어 문장을 입력하면 저장된 프레임 중 가장 비슷한 장면을 보여줍니다.")

    video_ids = get_indexed_video_ids()

    if not video_ids:
        st.error("인덱싱된 프레임이 없습니다. 탭 1에서 먼저 영상을 처리해주세요!")
    else:
        selected_video = st.selectbox("검색할 영상", video_ids)
        selected_video_id = selected_video

        indexed_frames = get_indexed_frames(selected_video_id)

        if not indexed_frames:
            st.error("선택한 영상에 저장된 프레임이 없습니다. worker 처리 상태를 확인해주세요!")
        else:
            st.info(f"현재 선택 범위의 인덱싱된 프레임 수: {len(indexed_frames)}")

            if st.button("🔄 로컬 프레임 인덱스 갱신"):
                with st.spinner("ChromaDB 인덱싱 중..."):
                    index_frames(str(FRAMES_DIR))

                st.success("인덱싱 완료")

            # 행동 이벤트 기반 추천 질문
            #
            # Streamlit은 버튼 클릭 시 전체 스크립트를 다시 실행한다.
            # 그래서 suggest_queries()를 매번 새로 호출하면
            # 추천질문 버튼을 누를 때마다 질문이 바뀐다.
            # 이를 막기 위해 영상별 추천질문/행동이벤트를 session_state에 저장한다.
            suggestion_scope = selected_video_id or "all"

            suggested_queries_key = f"suggested_queries_{suggestion_scope}"
            behavior_events_key = f"behavior_events_{suggestion_scope}"
            scene_events_key = f"scene_events_{suggestion_scope}"

            refresh_suggestions = st.button(
                "🔄 추천 질문 새로고침",
                key=f"refresh_suggestions_{suggestion_scope}",
            )

            if refresh_suggestions:
                st.session_state.pop(suggested_queries_key, None)
                st.session_state.pop(behavior_events_key, None)
                st.session_state.pop(scene_events_key, None)

            if suggested_queries_key not in st.session_state:
                st.session_state[suggested_queries_key] = suggest_queries(
                    user_id=user_id,
                    video_id=selected_video_id,
                    limit=6,
                )

            if behavior_events_key not in st.session_state:
                st.session_state[behavior_events_key] = get_suggestion_behavior_events(
                    video_id=selected_video_id,
                    limit=5,
                )

            suggested_queries = st.session_state[suggested_queries_key]
            behavior_events = st.session_state[behavior_events_key]

            prompt_context = build_prompt_context(
                user_id=user_id,
                video_id=selected_video_id,
            )

            # 행동 이벤트가 아직 없을 때만 기존 장면 후보를 fallback으로 보여준다.
            if not behavior_events:
                if scene_events_key not in st.session_state:
                    st.session_state[scene_events_key] = get_suggestion_events(
                        video_id=selected_video_id,
                        limit=5,
                    )

                scene_events = st.session_state[scene_events_key]
            else:
                scene_events = []

            if behavior_events:
                st.markdown("#### 🐾 오늘 발견한 주요 행동")

                for event in behavior_events[:3]:
                    start_time = event.get("start_time")
                    end_time = event.get("end_time")

                    if start_time is not None and end_time is not None:
                        time_text = f"{float(start_time):.1f}초 ~ {float(end_time):.1f}초"
                    else:
                        time_text = "시간 정보 없음"

                    summary = event.get("summary") or event.get("action") or "행동 설명 없음"
                    action = event.get("action") or "알 수 없음"
                    target_object = event.get("target_object")
                    duration = event.get("duration")
                    repeat_count = event.get("repeat_count")
                    confidence = event.get("confidence")
                    interestingness = event.get("interestingness")

                    st.markdown(f"**{time_text}**")
                    st.write(summary)

                    detail_items = []

                    if action:
                        detail_items.append(f"행동: {action}")

                    if target_object:
                        detail_items.append(f"대상: {target_object}")

                    if duration is not None:
                        detail_items.append(f"지속 시간: {float(duration):.1f}초")

                    if repeat_count:
                        detail_items.append(f"반복 후보: {repeat_count}회")

                    if confidence is not None:
                        detail_items.append(f"신뢰도: {float(confidence):.2f}")

                    if interestingness is not None:
                        detail_items.append(f"흥미도: {float(interestingness):.2f}")

                    if detail_items:
                        st.caption(" · ".join(detail_items))

                    st.markdown("---")

                with st.expander("행동 분석 근거 보기"):
                    for event in behavior_events:
                        start_time = event.get("start_time")
                        end_time = event.get("end_time")

                        if start_time is not None and end_time is not None:
                            time_text = f"{float(start_time):.1f}초 ~ {float(end_time):.1f}초"
                        else:
                            time_text = "시간 정보 없음"

                        st.write(f"- **{time_text}** / {event.get('summary') or event.get('action') or '행동 설명 없음'}")

                        evidence = event.get("evidence") or []

                        for item in evidence:
                            st.caption(f"  - {item}")

                st.markdown("#### 💡 이 행동에서 확인해볼 만한 질문")

            elif suggested_queries:
                st.markdown("#### 💡 이 영상에서 확인해볼 만한 질문")

                if scene_events:
                    with st.expander("추천 질문 생성에 사용된 장면 후보 보기"):
                        for event in scene_events:
                            timestamp = event.get("timestamp")
                            timestamp_text = f"{timestamp:.1f}초" if timestamp is not None else "여러 구간"
                            labels = ", ".join(event.get("labels", [])) or "없음"

                            st.write(f"- **{timestamp_text}** / `{event.get('event_type')}` / {labels}")
                            st.caption(event.get("description", ""))

                object_summary = ", ".join(
                    f"{item['label']}({item['count']})"
                    for item in prompt_context.get("dominant_objects", [])[:5]
                )

                if object_summary:
                    st.caption(f"감지된 주요 객체: {object_summary}")

            if suggested_queries:
                cols = st.columns(min(len(suggested_queries), 3))

                for i, suggested_query in enumerate(suggested_queries):
                    with cols[i % len(cols)]:
                        if st.button(
                            suggested_query,
                            key=f"behavior_suggested_query_{selected_video_id}_{i}",
                            use_container_width=True,
                        ):
                            st.session_state["search_query"] = suggested_query
                            st.rerun()

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
                    placeholder="예: 강아지가 특정 물체 근처에 머무른 장면",
                    key="search_query",
                    label_visibility="collapsed",
                )

                top_k = st.slider("검색 결과 수", min_value=1, max_value=10, value=3)

                search_btn = st.form_submit_button("🔍 검색", use_container_width=True)

            if search_btn and query:
                start_time = time.time()

                with st.spinner("ChromaDB 검색 및 AI 답변 생성 중..."):
                    rag_result = run_rag_query(
                        query=query,
                        video_id=selected_video_id,
                        top_k=top_k,
                        user_id=user_id,
                    )

                results = rag_result.get("results", [])
                answer = rag_result.get("answer", "")
                used_llm = rag_result.get("used_llm", False)

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

                if answer:
                    st.subheader("🤖 AI 답변")
                    st.write(answer)

                    if used_llm:
                        st.caption("LangChain + LLM으로 검색 결과 기반 답변을 생성했습니다.")
                    else:
                        st.caption("OPENAI_API_KEY가 없거나 LLM 호출에 실패하여 검색 metadata 기반 답변을 표시했습니다.")

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

                            if result.get("object_labels"):
                                st.caption(f"감지 객체: {result['object_labels']}")

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

                    if best.get("object_labels"):
                        st.write(f"**Detected objects**: {best['object_labels']}")

            elif search_btn and not query:
                st.warning("검색어를 입력해주세요!")

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

                            if result.get("object_labels"):
                                st.caption(f"감지 객체: {result['object_labels']}")

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

                    if best.get("object_labels"):
                        st.write(f"**Detected objects**: {best['object_labels']}")

            elif search_btn and not query:
                st.warning("검색어를 입력해주세요!")

            st.markdown("---")
            st.caption(
                "💡 추천 질문은 행동을 미리 정의한 것이 아니라, "
                "프레임별 객체 라벨 변화와 눈에 띄는 장면 후보를 바탕으로 자동 생성됩니다."
            )


# ─────────────────────────────────────────
# 탭 3: 영상별 검색 테스트
# ─────────────────────────────────────────
with tab3:
    if st.session_state.get("pipeline_processing"):
        st.info(
            "현재 영상 처리 중입니다. "
            "ChromaDB 안정성을 위해 영상 검색 테스트 탭을 비활성화했습니다."
        )
        st.stop()

    st.subheader("🧪 영상별 검색 테스트")
    st.write("사용자 맞춤 검색 로그와 분리해서, 선택한 영상의 ChromaDB 검색 결과만 확인합니다.")

    test_video_ids = get_indexed_video_ids()

    if not test_video_ids:
        st.error("인덱싱된 프레임이 없습니다. worker 처리 완료 후 다시 확인해주세요.")
    else:
        with st.form("video_search_test_form"):
            test_video = st.selectbox("테스트할 영상", ["전체"] + test_video_ids)
            test_query = st.text_input("검색어", placeholder="예: 강아지가 특정 물체 근처에 머무른 장면")
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

                        if result.get("object_labels"):
                            st.caption(f"감지 객체: {result['object_labels']}")

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