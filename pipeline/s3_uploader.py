import boto3
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET = os.getenv('AWS_BUCKET_NAME')


def upload_file(local_path, s3_key, extra_args=None):
    if not BUCKET:
        raise ValueError("AWS_BUCKET_NAME 환경변수가 설정되어 있지 않습니다.")

    upload_kwargs = {}
    if extra_args:
        upload_kwargs["ExtraArgs"] = extra_args

    s3.upload_file(local_path, BUCKET, s3_key, **upload_kwargs)
    print(f"업로드 성공: {s3_key}")
    return s3_key


def upload_video(local_path, s3_key):
    return upload_file(local_path, s3_key)


def upload_frame(local_path, video_id):
    frame_filename = os.path.basename(local_path)
    s3_key = f"frames/{video_id}/{frame_filename}"
    return upload_file(
        local_path,
        s3_key,
        extra_args={"ContentType": "image/jpeg"},
    )

def download_video(s3_key, local_path):
    if not BUCKET:
        raise ValueError("AWS_BUCKET_NAME 환경변수가 설정되어 있지 않습니다.")

    s3.download_file(BUCKET, s3_key, local_path)
    print(f"다운로드 성공: {local_path}")


def create_presigned_url(s3_key, expires_in=3600):
    if not BUCKET:
        raise ValueError("AWS_BUCKET_NAME 환경변수가 설정되어 있지 않습니다.")

    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": s3_key},
        ExpiresIn=expires_in,
    )


def object_exists(s3_key):
    if not BUCKET:
        raise ValueError("AWS_BUCKET_NAME 환경변수가 설정되어 있지 않습니다.")

    try:
        s3.head_object(Bucket=BUCKET, Key=s3_key)
        return True
    except ClientError as exc:
        status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        error_code = exc.response.get("Error", {}).get("Code")
        if status_code == 404 or error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
