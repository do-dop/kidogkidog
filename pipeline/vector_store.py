import chromadb
from pipeline.clip_embedder import embed_image, embed_text
from pathlib import Path
import os
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = Path(os.getenv("CHROMA_DB_PATH", PROJECT_ROOT / "db" / "chroma"))
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_SSL = os.getenv("CHROMA_SSL", "").lower() in {"1", "true", "yes"}
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "petcam_frames")

_client = None
_collection = None


def get_collection():
    """ChromaDB 컬렉션을 필요 시점에 생성"""
    global _client, _collection
    if _collection is None:
        if CHROMA_HOST:
            print(
                f"ChromaDB 서버 연결 시작: host={CHROMA_HOST}, port={CHROMA_PORT}, ssl={CHROMA_SSL}",
                flush=True,
            )
            _client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                ssl=CHROMA_SSL,
            )
        else:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"ChromaDB 로컬 연결 시작: {CHROMA_DIR}", flush=True)
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        print("ChromaDB 연결 완료", flush=True)
    return _collection


def _timestamp_from_frame_path(frame_path):
    stem = frame_path.stem
    if "_frame_" in stem:
        return float(stem.rsplit("_frame_", 1)[1])
    return float(stem.replace("frame_", ""))


def _video_id_from_frame_path(frame_path, frame_root):
    relative_path = frame_path.relative_to(frame_root)
    if len(relative_path.parts) > 1:
        return relative_path.parts[0]
    return "default"


def _frame_id(frame_path, frame_root):
    relative_path = frame_path.relative_to(frame_root).with_suffix("")
    return "__".join(relative_path.parts)


def _normalize_object_labels(object_labels):
    """
    ChromaDB metadata에 저장하기 좋은 문자열 형태로 변환한다.

    Chroma metadata는 list/dict보다 str, int, float, bool 같은 단순 타입이 안전하다.
    """
    if not object_labels:
        return ""

    if isinstance(object_labels, str):
        return object_labels

    return ",".join(str(label) for label in object_labels)


def index_frame(
    frame_path,
    frame_root="pipeline/frames",
    video_id=None,
    s3_key=None,
    object_labels=None,
):
    """
    단일 프레임을 CLIP 임베딩 후 ChromaDB에 저장
    """
    frame_root = Path(frame_root)
    frame_path = Path(frame_path)
    frame_id = _frame_id(frame_path, frame_root)
    frame_video_id = video_id or _video_id_from_frame_path(frame_path, frame_root)
    collection = get_collection()

    print(f"ChromaDB 기존 프레임 확인 시작: {frame_id}", flush=True)
    existing = collection.get(ids=[frame_id])
    if existing["ids"]:
        print(f"ChromaDB 기존 프레임 스킵: {frame_id}", flush=True)
        return frame_id
    print(f"ChromaDB 기존 프레임 확인 완료: {frame_id}", flush=True)

    print(f"CLIP 이미지 임베딩 시작: {frame_path}", flush=True)
    embedding = embed_image(str(frame_path))
    print(f"CLIP 이미지 임베딩 완료: {frame_path}", flush=True)

    metadata = {
        "frame_path": str(frame_path),
        "timestamp": _timestamp_from_frame_path(frame_path),
        "video_id": frame_video_id,
        "object_labels": _normalize_object_labels(object_labels),
    }

    if s3_key:
        metadata["s3_key"] = s3_key

    print(f"ChromaDB add 시작: {frame_id}", flush=True)
    collection.add(
        embeddings=[embedding],
        ids=[frame_id],
        metadatas=[metadata],
    )
    print(f"ChromaDB add 완료: {frame_id}", flush=True)
    return frame_id


def index_frames(frame_dir="pipeline/frames", video_id=None):
    """
    프레임 폴더의 모든 이미지를 CLIP 임베딩 후 ChromaDB에 저장
    """
    frame_root = Path(frame_dir)
    frame_paths = sorted(frame_root.rglob("*.jpg"))
    print(f"총 {len(frame_paths)}개 프레임 인덱싱 시작...")

    for i, frame_path in enumerate(frame_paths):
        index_frame(frame_path, frame_root=frame_root, video_id=video_id)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(frame_paths)} 완료...")

    print(f"인덱싱 완료! 총 {get_collection().count()}개 저장됨")


def search(query, top_k=3, video_id=None):
    """
    자연어 쿼리로 ChromaDB에서 유사한 프레임 검색
    """
    query_embedding = embed_text(query)
    collection = get_collection()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if video_id:
        query_kwargs["where"] = {"video_id": video_id}

    results = collection.query(**query_kwargs)

    output = []
    if not results["ids"] or not results["ids"][0]:
        return output

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]

        output.append({
            "frame_id": results["ids"][0][i],
            "frame_path": metadata["frame_path"],
            "s3_key": metadata.get("s3_key"),
            "timestamp": metadata["timestamp"],
            "video_id": metadata.get("video_id", "default"),
            "object_labels": metadata.get("object_labels", ""),
            "score": 1 - results["distances"][0][i],
        })

    return output


def expand_search_query(original_query: str) -> list[str]:
    """
    사용자 검색어를 CLIP 텍스트 검색에 더 잘 맞는 영어 검색 문장으로 확장한다.

    외부 LLM/API 없이 동작해야 하므로, 실패하거나 확장 단서를 못 찾으면 원문만 반환한다.
    """
    try:
        query = (original_query or "").strip()
        if not query:
            return [original_query]

        normalized = query.lower()
        compact_query = re.sub(r"\s+", "", normalized)

        subject = _infer_search_subject(compact_query)
        target = _infer_search_target(compact_query)
        action = _infer_search_action(compact_query)

        if not target and not action:
            return [query]

        expanded_queries = _build_expanded_queries(
            subject=subject,
            target=target,
            action=action,
            original_query=query,
        )

        return expanded_queries[:5] or [query]

    except Exception as exc:
        print(f"검색어 확장 실패: {exc}", flush=True)
        return [original_query]


def search_with_query_expansion(query, top_k=3, video_id=None):
    """
    원문 검색 결과에 확장 검색 결과를 보조로 합친 뒤 score 기준으로 정렬한다.
    """
    expanded_queries = _search_queries_with_original(query)
    merged_results = []

    for expanded_query in expanded_queries:
        try:
            query_results = search(
                expanded_query,
                top_k=top_k,
                video_id=video_id,
            )
            merged_results.extend(query_results)
        except Exception as exc:
            print(f"확장 검색 실패: query={expanded_query}, error={exc}", flush=True)

    if not merged_results:
        return []

    deduplicated_results = _deduplicate_search_results(merged_results)
    return sorted(
        deduplicated_results,
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )[:top_k]


def _search_queries_with_original(query: str) -> list[str]:
    expanded_queries = expand_search_query(query)
    search_queries = []
    seen = set()

    for candidate_query in [query, *expanded_queries]:
        normalized_query = (candidate_query or "").strip()
        if not normalized_query or normalized_query in seen:
            continue

        search_queries.append(normalized_query)
        seen.add(normalized_query)

        if len(search_queries) >= 5:
            break

    return search_queries or [query]


def _infer_search_subject(compact_query: str) -> str:
    if any(keyword in compact_query for keyword in ["고양이", "냥이", "cat"]):
        return "cat"
    if any(keyword in compact_query for keyword in ["강아지", "개", "멍멍", "dog"]):
        return "dog"
    return "dog"


def _infer_search_target(compact_query: str) -> str | None:
    target_keywords = [
        (["가방", "bag"], "a bag"),
        (["밥그릇", "사료그릇", "물그릇", "그릇", "bowl"], "a bowl"),
        (["사료", "밥", "food"], "food"),
        (["물병", "물", "water"], "water"),
        (["공", "ball"], "a ball"),
        (["장난감", "toy"], "a toy"),
        (["사람", "보호자", "person", "human"], "a person"),
        (["소파", "쇼파", "sofa", "couch"], "a sofa"),
        (["침대", "bed"], "a bed"),
        (["문", "door"], "a door"),
        (["신발", "shoe"], "a shoe"),
        (["카메라", "camera"], "a camera"),
    ]

    for keywords, target in target_keywords:
        if any(keyword in compact_query for keyword in keywords):
            return target

    return None


def _infer_search_action(compact_query: str) -> str | None:
    action_keywords = [
        (["만지", "만진", "만졌", "건드", "건든", "건드린", "발로", "touch", "paw"], "touching"),
        (["근처", "옆", "주변", "near"], "near"),
        (["먹", "밥먹", "사료먹", "eat"], "eating"),
        (["마시", "drink"], "drinking"),
        (["자", "잠", "누워", "lying", "sleep"], "lying near"),
        (["앉", "sit"], "sitting near"),
        (["뛰", "달리", "run"], "running near"),
        (["걷", "walk"], "walking near"),
        (["짖", "bark"], "barking near"),
        (["냄새", "맡", "sniff"], "sniffing"),
        (["보", "쳐다", "look"], "looking at"),
    ]

    for keywords, action in action_keywords:
        if any(keyword in compact_query for keyword in keywords):
            return action

    return None


def _build_expanded_queries(
    subject: str,
    target: str | None,
    action: str | None,
    original_query: str,
) -> list[str]:
    expanded_queries = []
    subjects = [subject]

    if subject == "dog":
        subjects.append("pet")
    elif subject == "cat":
        subjects.append("pet")
    else:
        subjects.extend(["dog", "cat"])

    if target and action:
        expanded_queries.extend(
            f"{candidate_subject} {action} {target}"
            for candidate_subject in subjects[:2]
        )
        expanded_queries.append(f"{subjects[0]} near {target}")

        if action == "touching":
            expanded_queries.append(f"{subjects[0]} pawing at {target}")
        elif action not in ["near", "lying near", "sitting near"]:
            expanded_queries.append(f"{subjects[0]} interacting with {target}")

    elif target:
        expanded_queries.extend([
            f"{subjects[0]} near {target}",
            f"{subjects[0]} interacting with {target}",
            f"pet near {target}",
        ])

    elif action:
        expanded_queries.extend([
            f"{subjects[0]} {action}",
            f"pet {action}",
        ])

    unique_queries = []
    seen = set()

    for expanded_query in expanded_queries:
        cleaned_query = re.sub(r"\s+", " ", expanded_query).strip()
        if cleaned_query and cleaned_query not in seen:
            unique_queries.append(cleaned_query)
            seen.add(cleaned_query)

    return unique_queries[:5] or [original_query]


def _deduplicate_search_results(results):
    deduplicated = []
    path_to_index = {}

    for result in results:
        duplicate_index = _find_duplicate_result_index(
            result=result,
            deduplicated=deduplicated,
            path_to_index=path_to_index,
        )

        if duplicate_index is None:
            deduplicated.append(result)
            _remember_result_paths(result, len(deduplicated) - 1, path_to_index)
            continue

        existing_result = deduplicated[duplicate_index]
        if float(result.get("score") or 0.0) > float(existing_result.get("score") or 0.0):
            deduplicated[duplicate_index] = result
            _remember_result_paths(result, duplicate_index, path_to_index)

    return deduplicated


def _find_duplicate_result_index(result, deduplicated, path_to_index):
    for path_key in ["clip_path", "frame_path"]:
        path_value = result.get(path_key)
        if path_value and path_value in path_to_index:
            return path_to_index[path_value]

    result_video_id = result.get("video_id")
    result_timestamp = _safe_float(result.get("timestamp"))

    if result_video_id is None or result_timestamp is None:
        return None

    for index, existing_result in enumerate(deduplicated):
        existing_video_id = existing_result.get("video_id")
        existing_timestamp = _safe_float(existing_result.get("timestamp"))

        if existing_video_id != result_video_id or existing_timestamp is None:
            continue

        if abs(existing_timestamp - result_timestamp) <= 1.0:
            return index

    return None


def _remember_result_paths(result, index, path_to_index):
    for path_key in ["clip_path", "frame_path"]:
        path_value = result.get(path_key)
        if path_value:
            path_to_index[path_value] = index


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_indexed_frames(video_id=None):
    """
    ChromaDB에 저장된 프레임 metadata 조회
    """
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    frames = []

    for frame_id, metadata in zip(results["ids"], results["metadatas"]):
        frame_video_id = metadata.get("video_id", "default")
        if isinstance(video_id, list) and frame_video_id not in video_id:
            continue
        if isinstance(video_id, str) and frame_video_id != video_id:
            continue

        frames.append({
            "frame_id": frame_id,
            "frame_path": metadata.get("frame_path"),
            "s3_key": metadata.get("s3_key"),
            "timestamp": metadata.get("timestamp"),
            "video_id": frame_video_id,
            "object_labels": metadata.get("object_labels", ""),
        })

    return frames


def get_indexed_video_ids():
    """
    ChromaDB metadata 기준으로 검색 가능한 video_id 목록 조회
    """
    return sorted({
        frame["video_id"]
        for frame in get_indexed_frames()
    })
