from fastapi import APIRouter, Request

from api.services.media_service import build_s3_media_response, build_s3_thumbnail_response


router = APIRouter(prefix="/media", tags=["media"])


@router.get("/thumbnail")
def get_s3_video_thumbnail(key: str):
    return build_s3_thumbnail_response(key)


@router.get("/s3")
def get_s3_media(key: str, request: Request):
    return build_s3_media_response(key, request)
