from pathlib import Path

from fastapi import APIRouter, HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAMES_DIR = PROJECT_ROOT / "pipeline" / "frames"

router = APIRouter(prefix="/frames", tags=["frames"])


@router.post("/reindex")
def reindex_local_frames(video_id: str | None = None):
    if not FRAMES_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=f"프레임 폴더가 없습니다: {FRAMES_DIR}",
        )

    try:
        from pipeline.vector_store import index_frames

        if video_id:
            target_dir = FRAMES_DIR / video_id
            if not target_dir.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"영상 프레임 폴더가 없습니다: {target_dir}",
                )
            index_frames(str(target_dir), video_id=video_id)
        else:
            index_frames(str(FRAMES_DIR))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"로컬 프레임 인덱스 갱신 실패: {exc}",
        ) from exc

    return {
        "status": "ok",
        "video_id": video_id,
        "frame_root": str(FRAMES_DIR if not video_id else FRAMES_DIR / video_id),
    }
