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

# CLIP 가중치(ViT-L/14, 약 890MB)를 이미지에 넣어 둔다. 콜드 스타트마다 인터넷에서 내려받으면
# 13초 이상 걸리고 편차도 크다. 소스보다 앞에 두어 코드 변경이 이 레이어를 무효화하지 않게 한다.
# 런타임이 읽는 경로(~/.cache/clip)와 같은 곳에 저장된다. 모델을 바꾸면 CLIP_MODEL도 같이 바꾼다.
ARG CLIP_MODEL=ViT-L/14
ENV CLIP_MODEL=${CLIP_MODEL}
RUN python -c "import os, clip; clip.load(os.environ['CLIP_MODEL'], device='cpu')"

# 소스 변경은 의존성 설치 레이어를 무효화하지 않는다.
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
