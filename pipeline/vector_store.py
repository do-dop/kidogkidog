import chromadb
from pipeline.clip_embedder import embed_image, embed_text
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "db" / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB 로컬 클라이언트
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name="petcam_frames",
    metadata={"hnsw:space": "cosine"}
)

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


def index_frames(frame_dir="pipeline/frames", video_id=None):
    """
    프레임 폴더의 모든 이미지를 CLIP 임베딩 후 ChromaDB에 저장
    """
    frame_root = Path(frame_dir)
    frame_paths = sorted(frame_root.rglob("*.jpg"))
    print(f"총 {len(frame_paths)}개 프레임 인덱싱 시작...")

    for i, frame_path in enumerate(frame_paths):
        frame_id = _frame_id(frame_path, frame_root)
        frame_video_id = video_id or _video_id_from_frame_path(frame_path, frame_root)

        # 이미 저장된 프레임은 스킵
        existing = collection.get(ids=[frame_id])
        if existing["ids"]:
            continue

        # CLIP 임베딩
        embedding = embed_image(str(frame_path))

        # ChromaDB에 저장
        collection.add(
            embeddings=[embedding],
            ids=[frame_id],
            metadatas=[{
                "frame_path": str(frame_path),
                "timestamp": _timestamp_from_frame_path(frame_path),
                "video_id": frame_video_id,
            }]
        )

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(frame_paths)} 완료...")

    print(f"인덱싱 완료! 총 {collection.count()}개 저장됨")


def search(query, top_k=3, video_id=None):
    """
    자연어 쿼리로 ChromaDB에서 유사한 프레임 검색
    """
    query_embedding = embed_text(query)

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
        output.append({
            "frame_id": results["ids"][0][i],
            "frame_path": results["metadatas"][0][i]["frame_path"],
            "timestamp": results["metadatas"][0][i]["timestamp"],
            "video_id": results["metadatas"][0][i].get("video_id", "default"),
            "score": 1 - results["distances"][0][i]
        })

    return output
