# Kidogkidog

> 당신의 펫, 지금 뭐하고 있을까?

Kidogkidog은 펫캠 영상에서 반려동물의 행동을 자연어로 검색하고, 관련 장면의 타임스탬프와 영상을 확인할 수 있는 Multimodal Video RAG 서비스입니다.

## 주요 기능

- 자연어 행동 검색: "강아지가 물 마신 장면 보여줘" 같은 문장으로 장면 검색
- 타임스탬프 반환: 검색 결과의 영상 ID, 시점, 유사도 제공
- 영상 재생: React UI에서 검색 결과 또는 녹화 청크 재생
- 이벤트 기반 처리: 영상 청크를 받아 motion 감지 후 필요한 프레임 추출
- 벡터 검색: CLIP 임베딩을 ChromaDB에 저장하고 유사 장면 검색
- 추천 질문: 저장된 행동 이벤트와 사용자 검색 로그 기반 질문 추천
- GCS 연동: 영상 청크 업로드·다운로드, API를 통한 비공개 미디어 스트리밍

## 기술 스택

| 분류 | 기술 |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI |
| Async Worker | Celery, RabbitMQ |
| AI/ML | CLIP, YOLO, LangChain |
| Video Processing | OpenCV, FFmpeg |
| Database | ChromaDB, MySQL (Cloud SQL) |
| Storage | Google Cloud Storage |
| Infra | Docker Compose |

## 시스템 구조

```text
원본 영상
  ↓
Edge Simulator
  ↓ 영상 청크 분할 + GCS 업로드
FastAPI
  ↓ Celery 작업 등록
RabbitMQ
  ↓
Celery Worker
  ↓ GCS 청크 다운로드
Motion 감지 → 프레임 추출 → YOLO 객체 감지
  ↓
CLIP 임베딩 → ChromaDB 저장
  ↓
React UI 자연어 검색
  ↓
FastAPI 검색 API → RAG 응답 + 검색 결과 반환
  ↓
React UI에서 결과 및 영상 재생
```

## 로컬 실행

### 1. 환경 변수 준비

루트 경로에 `.env`를 만들고 GCS, DB, LLM 등에 필요한 값을 설정합니다.

```bash
cp ui-react/.env.sample ui-react/.env
```

React 개발 서버는 기본적으로 `/api` 요청을 `http://127.0.0.1:8000`으로 프록시합니다. 다른 API 서버를 쓰려면 `ui-react/.env`의 `VITE_API_PROXY_TARGET` 값을 변경합니다.

### 2. GCP ChromaDB 연결 및 최소 인프라 실행

GCP에 배포된 API와 같은 `petcam_frames` 컬렉션을 사용합니다.
루트 `.env`에 다음 값을 설정합니다. Docker Compose는 호스트 주소만
`host.docker.internal`로 바꿔 같은 터널에 연결합니다.

```dotenv
CHROMA_HOST=127.0.0.1
CHROMA_PORT=18001
CHROMA_SSL=false
CHROMA_COLLECTION=petcam_frames
```

GCP 로그인 및 VM SSH 접근 권한이 있는 컴퓨터에서 별도 터미널로 실행하고,
로컬 API 또는 worker를 사용하는 동안 열어 둡니다. 터널이 이미 실행 중이면
다시 실행하지 않습니다. 종료는 `Ctrl+C`입니다.

```bash
bash scripts/chroma-tunnel.sh
```

이 터널은 `127.0.0.1:18001`을 GCP `kidog-chroma` VM의 8000번 포트로 연결합니다.
VM의 ChromaDB 포트를 인터넷에 공개하지 않습니다. Cloud Run은 기존 내부 IP로
직접 접근하므로 이 터널이 필요하지 않습니다.

영상 처리 작업을 등록하려면 RabbitMQ도 실행합니다.

```bash
docker compose up -d rabbitmq
```

기본 Compose 실행에서는 로컬 ChromaDB를 시작하지 않습니다. 기존 로컬 데이터는
삭제하지 않으며, 로컬 ChromaDB 서비스는 `local-chroma` 프로필에 보관되어 있습니다.
기존 로컬 벡터가 GCP로 자동 복사되지는 않습니다.

### 3. 백엔드 실행

```bash
python -m dotenv -f .env run -- uvicorn api.main:app --reload
```

영상 청크 처리까지 테스트하려면 worker도 별도 터미널에서 실행합니다.

```bash
python -m dotenv -f .env run -- celery -A pipeline.tasks worker --loglevel=info --pool=solo
```

### 4. React 프론트엔드 실행

```bash
cd ui-react
npm install
npm run dev
```

접속 주소:

```text
http://localhost:5173
```

### 5. API와 worker를 Docker로 실행

로컬 Python 실행 대신 API와 worker까지 Docker로 실행하려면 Docker Desktop과
위 SSH 터널을 켠 다음 실행합니다. 변경된 환경 변수를 적용하려면 컨테이너를
다시 생성해야 합니다.

```bash
docker compose up -d --build --force-recreate api worker
```

상태 확인:

```bash
docker compose ps
```

종료:

```bash
docker compose down
```

### 영상 처리 시간 관측 (Prometheus + Grafana)

Docker Desktop을 켠 뒤 모니터링 프로필을 함께 실행합니다. 처음에는 백엔드 이미지를
빌드하므로 시간이 걸릴 수 있습니다.

```bash
docker compose --profile monitoring up -d --build api worker prometheus grafana
```

Grafana는 `http://localhost:3000` (초기 계정 `admin` / `admin`), Prometheus는
`http://localhost:9090`에서 엽니다. Grafana의 **Kidogkidog → 영상 처리** 대시보드에
YOLO 감지, CLIP 이미지·텍스트 임베딩, ChromaDB 저장·검색, 프레임 추출 등 단계별
평균·95백분위 처리 시간과 실패 횟수가 표시됩니다. 초기 비밀번호는 로컬 `.env`의
`GRAFANA_ADMIN_PASSWORD`로 변경할 수 있습니다. Prometheus의 Targets에서
`kidog-api`, `kidog-worker`가 모두 UP인지 확인합니다.

측정값은 모니터링을 켠 뒤 **새로 처리한 영상**부터 쌓입니다. 기존 검색 DB의
프레임을 조회하는 것만으로는 영상 처리 단계 지표가 생기지 않습니다. 로컬에서
Python worker를 직접 실행할 때는 `METRICS_ENABLED=true METRICS_PORT=8002`를
설정하고 위의 `--pool=solo` 방식을 사용합니다. worker 지표는
`http://localhost:8002/metrics`, API 지표는 `http://localhost:8000/metrics`에서
직접 확인할 수 있습니다. 영상 ID·질문 원문은 Prometheus 라벨에 넣지 않습니다.

```bash
docker compose --profile monitoring down
```

영상 업로드는 같은 GCS 버킷을 사용하고, 로컬 API에 작업을 등록합니다.

```bash
python -m dotenv -f .env run -- python -m simulator.edge_simulator data/videos --server-url http://localhost:8000
```

worker가 배포 API와 같은 GCS 버킷과 Cloud SQL 데이터베이스를 사용해야 영상과
분석 결과도 배포 페이지에서 조회됩니다. 로컬 GCP 인증에는 GCS 읽기·쓰기 권한이
필요합니다. ChromaDB 연결 변경만으로 MySQL/GCS 인증이 설정되지는 않습니다.
처리가 끝난 뒤에는 로컬 worker·RabbitMQ·터널을 꺼도 배포 페이지에서 검색할 수
있습니다. 검색어 벡터 생성과 검색 응답 처리는 Cloud Run API가 수행합니다.

## 프로젝트 구조

```text
kidogkidog/
├── api/
│   ├── Dockerfile
│   └── main.py                    # FastAPI 엔드포인트, 검색 API, 미디어 프록시
├── data/
│   └── videos/                    # 로컬 테스트 영상 입력 경로
├── db/
│   ├── behavior_events.py         # 행동 이벤트 저장 및 조회
│   ├── connection.py              # MySQL 연결
│   ├── json_utils.py              # DB JSON 직렬화 유틸리티
│   ├── scenes.py                  # 장면 메타데이터 저장 및 조회
│   ├── schema.py                  # 테이블 생성 및 초기화
│   └── search_history.py          # 검색 로그와 빈출 검색어 관리
├── pipeline/
│   ├── tasks.py                   # Celery 영상 처리 작업 및 작업 이름 유지
│   ├── video/                    # 움직임 감지와 프레임 추출
│   ├── models/                   # CLIP 임베딩과 YOLO 객체 감지
│   ├── storage/                  # GCS 파일과 ChromaDB 벡터 저장소
│   ├── events/                   # 행동·장면 이벤트 추출
│   ├── search/                   # 질의 분석, 추천 질문, RAG 답변
│   ├── observability/            # 처리 시간 및 실패 지표
│   └── *.py                      # 기존 import 경로를 위한 호환 모듈
├── simulator/
│   ├── chunk_splitter.py          # FFmpeg 청크 분할
│   └── edge_simulator.py          # 청크 업로드 및 API 전송 시뮬레이터
├── ui-react/
│   ├── src/
│   │   ├── App.css
│   │   ├── App.tsx                # 주요 화면 및 API 연동
│   │   └── main.tsx
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts             # Vite dev server 및 API 프록시
├── docker-compose.yml
├── requirements.txt
└── yolov8n.pt
```

## 로컬 산출물

다음 파일과 디렉토리는 실행 중 생성되는 산출물이므로 Git에 올리지 않습니다.

```text
pipeline/frames/
simulator/chunks/
db/chroma/
ui-react/dist/
ui-react/node_modules/
```

## 팀원

| 이름 | 역할 |
|---|---|
| 박도연 | Backend & Data Pipeline Lead |
| 김태희 | AI Application & Web Frontend Lead |

## 커밋 규칙

| 타입 | 설명 |
|---|---|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 |
| `style` | 코드 포맷, 기능 변경 없음 |
| `refactor` | 리팩토링 |
| `test` | 테스트 코드 |
| `chore` | 환경설정, 패키지 등 |

예시:

```bash
feat: React 검색 결과 재생 화면 추가
fix: S3 청크 presigned URL 만료 처리 수정
chore: Docker Compose ChromaDB 볼륨 설정 추가
```

## 브랜치 전략

| 브랜치 | 용도 |
|---|---|
| `main` | 최종 배포용 |
| `dev` | 통합 개발 |
| `feature/기능명` | 기능별 개발 |

```text
feature/기능명 → dev → main
```
