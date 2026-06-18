import hashlib
import mimetypes
from pathlib import Path
import tempfile
from urllib.parse import quote

from fastapi import HTTPException, Request, Response

from pipeline.s3_uploader import (
    download_bytes,
    download_range,
    download_video,
    get_object_metadata,
)


def media_path_for_key(s3_key):
    return f"/media/s3?key={quote(s3_key)}"


def thumbnail_path_for_key(s3_key):
    return f"/media/thumbnail?key={quote(s3_key)}"


def build_s3_media_response(key: str, request: Request):
    if not key:
        raise HTTPException(status_code=400, detail="key가 필요합니다.")

    media_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    try:
        if range_header:
            metadata = get_object_metadata(key)
            total_size = int(metadata["ContentLength"])
            range_value = range_header.replace("bytes=", "", 1)
            start_text, _, end_text = range_value.partition("-")
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else min(start + 1024 * 1024 - 1, total_size - 1)
            end = min(end, total_size - 1)

            if start > end or start >= total_size:
                return Response(
                    status_code=416,
                    headers={
                        "Content-Range": f"bytes */{total_size}",
                        "Accept-Ranges": "bytes",
                    },
                )

            data = download_range(key, start, end)

            return Response(
                content=data,
                status_code=206,
                media_type=media_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{total_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(data)),
                    "Cache-Control": "private, max-age=300",
                },
            )

        data = download_bytes(key)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"S3 객체를 불러오지 못했습니다: {exc}",
        ) from exc

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{Path(key).name}"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, max-age=300",
        },
    )


def build_s3_thumbnail_response(key: str):
    if not key:
        raise HTTPException(status_code=400, detail="key가 필요합니다.")

    cache_dir = Path(tempfile.gettempdir()) / "kidogkidog_thumbnails"
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    thumbnail_path = cache_dir / f"{digest}.jpg"

    if not thumbnail_path.exists():
        video_path = cache_dir / f"{digest}.mp4"

        try:
            download_video(key, str(video_path))

            import cv2

            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            cap.release()

            if not ok:
                raise ValueError("영상 첫 프레임을 읽을 수 없습니다.")

            cv2.imwrite(str(thumbnail_path), frame)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"썸네일을 생성하지 못했습니다: {exc}",
            ) from exc
        finally:
            try:
                video_path.unlink(missing_ok=True)
            except Exception:
                pass

    return Response(
        content=thumbnail_path.read_bytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )
