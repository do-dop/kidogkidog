FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt가 바뀌지 않는 한 이 레이어(특히 Torch/CLIP/OpenCV)는 재사용한다.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 소스 변경은 의존성 설치 레이어를 무효화하지 않는다.
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
