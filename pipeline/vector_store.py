import chromadb
from pipeline.clip_embedder import embed_image, embed_text
from pathlib import Path
import os

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
