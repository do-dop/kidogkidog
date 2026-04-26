import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET = os.getenv('AWS_BUCKET_NAME')

def upload_video(local_path, s3_key):
    s3.upload_file(local_path, BUCKET, s3_key)
    print(f"업로드 성공: {s3_key}")

def download_video(s3_key, local_path):
    s3.download_file(BUCKET, s3_key, local_path)
    print(f"다운로드 성공: {local_path}")


if __name__ == "__main__":
    upload_video("test_video_long.mp4", "videos/test_video_long.mp4")