FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cloud Run과 worker VM에는 GPU가 없으므로 CPU 전용 Torch를 먼저 설치한다.
# PyPI 기본 Torch는 CUDA 라이브러리(약 2GB)를 함께 받아 이미지가 3GB까지 커진다.
# 버전은 requirements.txt의 torch/torchvision 고정값과 맞춘다.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 torchvision==0.26.0

# requirements.txt가 바뀌지 않는 한 이 레이어(특히 CLIP/OpenCV)는 재사용한다.
# 위에서 설치한 2.11.0+cpu가 torch==2.11.0을 만족하므로 CUDA 빌드를 다시 받지 않는다.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 소스 변경은 의존성 설치 레이어를 무효화하지 않는다.
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
