import hashlib
import io
import mimetypes
from collections import OrderedDict
from pathlib import Path
import subprocess
import tempfile
from threading import Lock
from urllib.parse import quote

from fastapi import HTTPException, Request, Response

from pipeline.storage.gcs_uploader import (
    download_bytes,
    download_range,
    download_video,
    get_object_metadata,
)


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MIN_IMAGE_WIDTH = 64
MAX_IMAGE_WIDTH = 1280
_RESIZED_IMAGE_CACHE: "OrderedDict[tuple[str, int], bytes]" = OrderedDict()
_RESIZED_IMAGE_CACHE_LIMIT = 256
_RESIZED_IMAGE_CACHE_LOCK = Lock()


def media_path_for_key(s3_key):
    return f"/media/s3?key={quote(s3_key)}"


def thumbnail_path_for_key(s3_key):
    return f"/media/thumbnail?key={quote(s3_key)}"


def build_resized_image_response(key: str, width: int):
    """프레임 이미지를 목록·카드용 크기로 줄여서 돌려준다. 원본은 1920px(약 650KB)라 한 화면에 수십 장이면 느리다."""
    width = max(MIN_IMAGE_WIDTH, min(width, MAX_IMAGE_WIDTH))
    cache_key = (key, width)

    with _RESIZED_IMAGE_CACHE_LOCK:
        cached = _RESIZED_IMAGE_CACHE.get(cache_key)
        if cached is not None:
            _RESIZED_IMAGE_CACHE.move_to_end(cache_key)

    if cached is None:
        try:
            from PIL import Image

            with Image.open(io.BytesIO(download_bytes(key))) as image:
                image = image.convert("RGB")
                if image.width > width:
                    image = image.resize((width, max(1, round(image.height * width / image.width))))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80, optimize=True)
                cached = buffer.getvalue()
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"이미지를 불러오지 못했습니다: {exc}",
            ) from exc

        with _RESIZED_IMAGE_CACHE_LOCK:
            _RESIZED_IMAGE_CACHE[cache_key] = cached
            while len(_RESIZED_IMAGE_CACHE) > _RESIZED_IMAGE_CACHE_LIMIT:
                _RESIZED_IMAGE_CACHE.popitem(last=False)

    return Response(
        content=cached,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


def build_s3_media_response(key: str, request: Request, width: int | None = None):
    if not key:
        raise HTTPException(status_code=400, detail="key가 필요합니다.")

    if width and key.lower().endswith(IMAGE_SUFFIXES):
        return build_resized_image_response(key, width)

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
            detail=f"GCS 객체를 불러오지 못했습니다: {exc}",
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

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    "0",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(thumbnail_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            if not thumbnail_path.exists():
                raise ValueError("영상 첫 프레임을 생성하지 못했습니다.")
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
