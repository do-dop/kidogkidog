import json
import chromadb
from pipeline.clip_embedder import embed_image, embed_text
from pathlib import Path
from datetime import datetime, timedelta
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


def _frame_recorded_at(metadata):
    recorded_at = metadata.get("recorded_at")
    if recorded_at:
        try:
            return datetime.fromisoformat(str(recorded_at).replace("Z", "+00:00"))
        except ValueError:
            pass

    s3_key = metadata.get("s3_key") or metadata.get("frame_path") or ""
    filename = Path(s3_key).name
    match = re.search(
        r"petcam_(\d{8})_(\d{6})_(\d{3})_frame_([0-9.]+)\.jpg$",
        filename,
    )

    if not match:
        return None

    base = datetime.strptime(
        f"{match.group(1)}{match.group(2)}",
        "%Y%m%d%H%M%S",
    )
    chunk_offset = int(match.group(3)) * 60
    frame_offset = float(match.group(4))
    return base + timedelta(seconds=chunk_offset + frame_offset)


def _matches_datetime_filter(metadata, recording_date=None, time_range=None):
    if not recording_date and not time_range:
        return True

    recorded_at = _frame_recorded_at(metadata)

    if recording_date:
        if not recorded_at or recorded_at.date().isoformat() != recording_date:
            return False

    if not time_range:
        return True

    start_hour = time_range.get("start_hour")
    end_hour = time_range.get("end_hour")

    if start_hour is None or end_hour is None:
        return True

    try:
        start_hour = int(start_hour)
        end_hour = int(end_hour)
    except (TypeError, ValueError):
        return True

    if start_hour <= 0 and end_hour >= 24:
        return True

    if not recorded_at:
        return False

    frame_hour = recorded_at.hour + recorded_at.minute / 60 + recorded_at.second / 3600

    if start_hour <= end_hour:
        return start_hour <= frame_hour < end_hour

    return frame_hour >= start_hour or frame_hour < end_hour


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


def _normalize_object_detections(object_detections):
    """Chroma metadata에 넣을 detection 목록 JSON 문자열로 변환한다."""
    if object_detections is None:
        return "[]"
    if isinstance(object_detections, str):
        # 이미 직렬화된 값을 받는 경우 유효 JSON인지 확인한 뒤 그대로 둔다.
        parsed = json.loads(object_detections)
        if not isinstance(parsed, list):
            raise ValueError("object_detections_json must encode a list")
        return object_detections
    return json.dumps(object_detections, ensure_ascii=False, separators=(",", ":"))


def _parse_object_detections(metadata):
    """Chroma scalar JSON metadata를 검색 결과용 detection list로 복원한다."""
    raw = metadata.get("object_detections_json")
    if not raw:
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _normalize_species_resolution(species_resolution):
    if species_resolution is None:
        return "[]"
    if isinstance(species_resolution, str):
        parsed = json.loads(species_resolution)
        if not isinstance(parsed, list):
            raise ValueError("species_resolution_json must encode a list")
        return species_resolution
    return json.dumps(species_resolution, ensure_ascii=False, separators=(",", ":"))


def _parse_species_resolution(metadata):
    raw = metadata.get("species_resolution_json")
    if not raw:
        return []
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def index_frame(
    frame_path,
    frame_root="pipeline/frames",
    video_id=None,
    s3_key=None,
    object_labels=None,
    recorded_at=None,
    timestamp=None,
    object_detections=None,
    species_resolution=None,
    object_detection_model=None,
    object_detection_confidence_threshold=None,
    object_labels_confidence_threshold=None,
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
        "timestamp": (
            float(timestamp) if timestamp is not None
            else _timestamp_from_frame_path(frame_path)
        ),
        "video_id": frame_video_id,
        "object_labels": _normalize_object_labels(object_labels),
        "object_detections_json": _normalize_object_detections(object_detections),
        "species_resolution_json": _normalize_species_resolution(species_resolution),
    }

    detection_metadata = {
        "object_detection_model": object_detection_model,
        "object_detection_confidence_threshold": object_detection_confidence_threshold,
        "object_labels_confidence_threshold": object_labels_confidence_threshold,
    }
    metadata.update({key: value for key, value in detection_metadata.items() if value is not None})

    if s3_key:
        metadata["s3_key"] = s3_key
    if recorded_at:
        metadata["recorded_at"] = str(recorded_at)

    print(f"ChromaDB add 시작: {frame_id}", flush=True)
    collection.add(
        embeddings=[embedding],
        ids=[frame_id],
        metadatas=[metadata],
    )
    print(f"ChromaDB add 완료: {frame_id}", flush=True)
    return frame_id


def update_frame_detection_metadata(
    frame_id,
    object_labels,
    object_detections=None,
    species_resolution=None,
    detection_model=None,
    detection_confidence_threshold=None,
    object_labels_confidence_threshold=None,
):
    """기존 Chroma frame의 detection metadata만 갱신한다.

    embedding과 frame ID는 변경하지 않는다. 알 수 없는 ID는 실수로 새 frame을
    만들지 않도록 예외 처리한다. 이 함수는 명시적으로 호출될 때만 동작한다.
    """
    collection = get_collection()
    existing = collection.get(ids=[frame_id], include=["metadatas"])
    ids = existing.get("ids") or []
    if frame_id not in ids:
        raise ValueError(f"ChromaDB frame ID not found: {frame_id}")

    index = ids.index(frame_id)
    metadata_rows = existing.get("metadatas") or []
    metadata = dict(metadata_rows[index] or {}) if index < len(metadata_rows) else {}
    metadata.update({
        "object_labels": _normalize_object_labels(object_labels),
        "object_detections_json": _normalize_object_detections(object_detections),
    })
    if species_resolution is not None:
        metadata["species_resolution_json"] = _normalize_species_resolution(species_resolution)
    optional_values = {
        "object_detection_model": detection_model,
        "object_detection_confidence_threshold": detection_confidence_threshold,
        "object_labels_confidence_threshold": object_labels_confidence_threshold,
    }
    metadata.update({key: value for key, value in optional_values.items() if value is not None})

    collection.update(ids=[frame_id], metadatas=[metadata])
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


def explicit_object_labels(query: str) -> set[str]:
    """원문에 명시된 객체만 추출한다. 중립적인 '반려동물'은 종을 지정하지 않는다."""
    terms = {
        "cat": ["고양이", "cat"],
        "dog": ["강아지", "개", "dog"],
        "person": ["사람", "person", "human", "people"],
    }
    return {label for label, names in terms.items() if _matches_search_noun(query, names)}


def filter_frames_by_objects(frames: list[dict], required: set[str]) -> list[dict]:
    """정확한 label 토큰으로 검사한다. 여러 명시 객체는 모두 존재해야 한다."""
    if not required:
        return frames
    result = []
    for frame in frames:
        labels = frame.get("object_labels") or []
        if isinstance(labels, str):
            labels = labels.split(",")
        if not isinstance(labels, (list, tuple, set)):
            continue
        detected = {str(label).strip().lower() for label in labels}
        if required.issubset(detected):
            result.append(frame)
    return result


def _object_candidate_frames(query, video_id=None, recording_date=None, time_range=None):
    # Top K 이후에 필터하면 낮은 순위의 일치 프레임을 놓치므로 범위 metadata부터 조회한다.
    kwargs = {"include": ["metadatas"]}
    if video_id:
        kwargs["where"] = {"video_id": video_id}
    result = get_collection().get(**kwargs)
    frames = [
        {**metadata, "frame_id": frame_id}
        for frame_id, metadata in zip(result["ids"], result["metadatas"])
        if _matches_datetime_filter(metadata, recording_date, time_range)
    ]
    return filter_frames_by_objects(frames, explicit_object_labels(query))


def search(query, top_k=3, video_id=None, recording_date=None, time_range=None):
    """
    자연어 쿼리로 ChromaDB에서 유사한 프레임 검색
    """
    if top_k <= 0:
        return []
    required = explicit_object_labels(query)
    candidates = _object_candidate_frames(query, video_id, recording_date, time_range) if required else None
    if required and not candidates:
        return []
    query_embedding = embed_text(query)
    collection = get_collection()

    requested_results = top_k
    has_filter = bool(recording_date or time_range)
    query_results = min(max(top_k * 30, top_k), 200) if has_filter else top_k

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": query_results,
    }
    if video_id:
        query_kwargs["where"] = {"video_id": video_id}
    if candidates is not None:
        query_kwargs["ids"] = [frame["frame_id"] for frame in candidates]
        query_kwargs["n_results"] = min(query_results, len(candidates))

    results = collection.query(**query_kwargs)

    output = []
    if not results["ids"] or not results["ids"][0]:
        return output

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        if not filter_frames_by_objects([metadata], required):
            continue

        if not _matches_datetime_filter(metadata, recording_date=recording_date, time_range=time_range):
            continue

        output.append({
            "frame_id": results["ids"][0][i],
            "frame_path": metadata["frame_path"],
            "s3_key": metadata.get("s3_key"),
            "timestamp": metadata["timestamp"],
            "video_id": metadata.get("video_id", "default"),
            "object_labels": metadata.get("object_labels", ""),
            "object_detections": _parse_object_detections(metadata),
            "species_resolution": _parse_species_resolution(metadata),
            "score": 1 - results["distances"][0][i],
        })

        if len(output) >= requested_results:
            break

    return output


def get_indexed_frames_in_time_window(video_id, start_time=None, end_time=None):
    """유사도 순위 제한 없이 영상/로컬 시간 범위의 metadata를 조회한다.

    청크는 이 metadata를 호출자의 기존 청크 판별 함수에 전달해 제한한다.
    """
    if not video_id:
        return []

    conditions = [{"video_id": video_id}]
    if start_time is not None and end_time is not None:
        conditions.extend([
            {"timestamp": {"$gte": float(start_time)}},
            {"timestamp": {"$lte": float(end_time)}},
        ])
    where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
    results = get_collection().get(where=where, include=["metadatas"])
    return [
        {**metadata, "frame_id": frame_id}
        for frame_id, metadata in zip(results["ids"], results["metadatas"])
    ]


def search_within_frames(query, frames, top_k=3):
    """이미 확정된 이벤트 후보 ID 안에서만 기존 CLIP/확장 검색을 수행한다."""
    required = explicit_object_labels(query)
    frames = filter_frames_by_objects(frames, required)
    candidate_ids = list(dict.fromkeys(frame["frame_id"] for frame in frames))
    if not candidate_ids or top_k <= 0:
        return []

    collection = get_collection()
    allowed_ids = set(candidate_ids)
    merged_results = []
    for expanded_query in _search_queries_with_original(query):
        try:
            results = collection.query(
                ids=candidate_ids,
                query_embeddings=[embed_text(expanded_query)],
                # 이벤트 후보 전부를 평가한 뒤 중복 제거 및 최종 top_k를 적용한다.
                n_results=len(candidate_ids),
                include=["metadatas", "distances"],
            )
            if not results["ids"] or not results["ids"][0]:
                continue
            for frame_id, metadata, distance in zip(
                results["ids"][0], results["metadatas"][0], results["distances"][0]
            ):
                if frame_id not in allowed_ids or not filter_frames_by_objects([metadata], required):
                    continue
                merged_results.append({
                    "frame_id": frame_id,
                    "frame_path": metadata.get("frame_path"),
                    "s3_key": metadata.get("s3_key"),
                    "timestamp": metadata["timestamp"],
                    "video_id": metadata.get("video_id", "default"),
                    "object_labels": metadata.get("object_labels", ""),
                    "object_detections": _parse_object_detections(metadata),
                    "species_resolution": _parse_species_resolution(metadata),
                    "score": 1 - distance,
                })
        except Exception as exc:
            print(f"이벤트 후보 검색 실패: query={expanded_query}, error={exc}", flush=True)

    return sorted(
        _deduplicate_search_results(merged_results),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )[:top_k]


def expand_search_query(original_query: str) -> list[str]:
    """
    사용자 검색어를 CLIP 텍스트 검색에 더 잘 맞는 영어 검색 문장으로 확장한다.

    외부 LLM/API 없이 동작해야 하므로, 실패하거나 확장 단서를 못 찾으면 원문만 반환한다.
    """
    try:
        query = (original_query or "").strip()
        if not query:
            return [original_query]

        subject = _infer_search_subject(query)
        target = _infer_search_target(query)
        action = _infer_search_action(query)

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


def search_with_query_expansion(query, top_k=3, video_id=None, recording_date=None, time_range=None):
    """
    원문 검색 결과에 확장 검색 결과를 보조로 합친 뒤 score 기준으로 정렬한다.
    """
    if explicit_object_labels(query):
        frames = _object_candidate_frames(query, video_id, recording_date, time_range)
        # 원문에서 추출한 조건을 유지한 채 기존 확장 문장/CLIP 점수로만 정렬한다.
        return search_within_frames(query, frames, top_k=top_k)

    expanded_queries = _search_queries_with_original(query)
    merged_results = []

    for expanded_query in expanded_queries:
        try:
            query_results = search(
                expanded_query,
                top_k=top_k,
                video_id=video_id,
                recording_date=recording_date,
                time_range=time_range,
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


# 명사 뒤의 조사만 허용한다. 단어 내부의 '문/물/공/개'는 의미 단서가 아니다.
_SEARCH_NOUN_SUFFIX = r"(?:들)?(?:은|는|이|가|을|를|의|에|에서|에게|한테|와|과|랑|이랑|하고|도|만|로|으로|부터|까지|처럼|보다)*"
_SEARCH_VERB_ENDING = r"(?:다|고|는|던|며|면서|면|다가|기|지|게|거나|는지|고서)"


def _matches_search_terms(query: str, korean_patterns: list[str], english_words: list[str]) -> bool:
    """공백/구두점 경계를 보존하고 한국어 활용형·영어 단어 전체를 비교한다."""
    tokens = re.findall(r"[^\W_]+", query.lower())
    return any(
        token in english_words
        or any(re.fullmatch(pattern, token) for pattern in korean_patterns)
        for token in tokens
    )


def _matches_search_noun(query: str, keywords: list[str]) -> bool:
    korean_patterns = [
        re.escape(word) + _SEARCH_NOUN_SUFFIX
        for word in keywords if not word.isascii()
    ]
    english_words = [
        form for word in keywords if word.isascii()
        for form in (word, word + "s")
    ]
    return _matches_search_terms(query, korean_patterns, english_words)


def _infer_search_subject(query: str) -> str:
    if _matches_search_noun(query, ["고양이", "냥이", "cat"]):
        return "cat"
    if _matches_search_noun(query, ["강아지", "개", "멍멍", "dog"]):
        return "dog"
    return "pet"


def _infer_search_target(query: str) -> str | None:
    target_keywords = [
        (["가방", "bag"], "a bag"),
        (["밥그릇", "사료그릇", "물그릇", "그릇", "bowl"], "a bowl"),
        (["사료", "밥", "food"], "food"),
        (["물병", "물", "water"], "water"),
        (["공", "ball"], "a ball"),
        (["장난감", "toy"], "a toy"),
        (["사람", "보호자", "person", "people", "human"], "a person"),
        (["소파", "쇼파", "sofa", "couch"], "a sofa"),
        (["침대", "bed"], "a bed"),
        (["문", "door"], "a door"),
        (["신발", "shoe"], "a shoe"),
        (["카메라", "camera"], "a camera"),
    ]

    for keywords, target in target_keywords:
        if _matches_search_noun(query, keywords):
            return target

    return None


def _infer_search_action(query: str) -> str | None:
    ending = _SEARCH_VERB_ENDING
    action_keywords = [
        ([rf"(?:만지|건드리){ending}", r"만진|만졌(?:다|던|어|어요)?|건든|건드린|건드렸(?:다|던|어|어요)?|발로"],
         ["touch", "touches", "touching", "touched", "paw", "paws", "pawing", "pawed"], "touching"),
        ([rf"(?:근처|옆|주변){_SEARCH_NOUN_SUFFIX}"], ["near", "nearby"], "near"),
        ([rf"(?:먹|밥먹|사료먹){ending}", r"먹은|먹었(?:다|던|어|어요)?|먹어(?:요)?"],
         ["eat", "eats", "eating", "ate", "eaten"], "eating"),
        ([rf"마시{ending}", r"마신|마셨(?:다|던|어|어요)?|마셔(?:요)?"],
         ["drink", "drinks", "drinking", "drank", "drunk"], "drinking"),
        ([rf"(?:자|잠자){ending}", rf"잠{_SEARCH_NOUN_SUFFIX}", r"잠든|잠들(?:다|고|어|어서|었(?:다|던)?)|잠드는|잤(?:다|던|어|어요)?"],
         ["sleep", "sleeps", "sleeping", "slept", "asleep"], "sleeping near"),
        ([r"누워(?:서|요)?|누운|누웠(?:다|던)?"], ["lying"], "lying near"),
        ([rf"앉{ending}", r"앉은|앉아(?:서|요)?|앉았(?:다|던)?"],
         ["sit", "sits", "sitting", "sat"], "sitting near"),
        ([rf"(?:뛰|달리){ending}", r"뛴|뛰어|뛰었(?:다|던)?|달린|달려|달렸(?:다|던)?"],
         ["run", "runs", "running", "ran"], "running near"),
        ([rf"걷{ending}", r"걸은|걸어(?:서|요)?|걸었(?:다|던)?"],
         ["walk", "walks", "walking", "walked"], "walking near"),
        ([rf"짖{ending}", r"짖은|짖어(?:요)?|짖었(?:다|던)?"],
         ["bark", "barks", "barking", "barked"], "barking near"),
        ([rf"냄새{_SEARCH_NOUN_SUFFIX}", rf"맡{ending}", r"맡은|맡았(?:다|던)?"],
         ["sniff", "sniffs", "sniffing", "sniffed"], "sniffing"),
        ([rf"(?:보|쳐다보){ending}", r"본|봤(?:다|던)?|쳐다본|쳐다봤(?:다|던)?"],
         ["look", "looks", "looking", "looked"], "looking at"),
    ]

    for korean_patterns, english_words, action in action_keywords:
        if _matches_search_terms(query, korean_patterns, english_words):
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

    if subject in {"dog", "cat"}:
        subjects.append("pet")

    if target and action:
        expanded_queries.extend(
            f"{candidate_subject} {action} {target}"
            for candidate_subject in subjects[:2]
        )
        expanded_queries.append(f"{subjects[0]} near {target}")

        if action == "touching":
            expanded_queries.append(f"{subjects[0]} pawing at {target}")
        elif action not in ["near", "sleeping near", "lying near", "sitting near"]:
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
            "object_detections": _parse_object_detections(metadata),
            "species_resolution": _parse_species_resolution(metadata),
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
