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

# 테스트 파일 생성
with open("test.txt", "w") as f:
    f.write("kidogkidog test!")

# 업로드
s3.upload_file("test.txt", BUCKET, "test/test.txt")
print("업로드 성공!")

# 다운로드
s3.download_file(BUCKET, "test/test.txt", "downloaded.txt")
print("다운로드 성공!")

# 확인
with open("downloaded.txt", "r") as f:
    print(f.read())