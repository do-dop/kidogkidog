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
- S3 연동: 영상 청크 업로드, 다운로드, presigned URL 생성

## 기술 스택

| 분류 | 기술 |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI |
| Async Worker | Celery, RabbitMQ |
| AI/ML | CLIP, YOLO, LangChain |
| Video Processing | OpenCV, FFmpeg |
| Database | ChromaDB, SQLite, MySQL 호환 메타데이터 계층 |
| Storage | AWS S3 |
| Infra | Docker Compose |

## 시스템 구조

```text
원본 영상
  ↓
Edge Simulator
  ↓ 영상 청크 분할 + S3 업로드
FastAPI
  ↓ Celery 작업 등록
RabbitMQ
  ↓
Celery Worker
  ↓ S3 청크 다운로드
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

루트 경로에 `.env`를 만들고 S3, DB, LLM 등에 필요한 값을 설정합니다.

```bash
cp ui-react/.env.sample ui-react/.env
```

React 개발 서버는 기본적으로 `/api` 요청을 `http://127.0.0.1:8000`으로 프록시합니다. 다른 API 서버를 쓰려면 `ui-react/.env`의 `VITE_API_PROXY_TARGET` 값을 변경합니다.

### 2. 최소 인프라 실행

React UI와 로컬 FastAPI 연결만 확인하려면 RabbitMQ만 먼저 띄웁니다.

```bash
docker compose up -d rabbitmq
```

ChromaDB까지 Docker로 함께 띄우려면 다음 명령을 사용합니다.

```bash
docker compose up -d rabbitmq chromadb
```

### 3. 백엔드 실행

```bash
uvicorn api.main:app --reload
```

영상 청크 처리까지 테스트하려면 worker도 별도 터미널에서 실행합니다.

```bash
celery -A pipeline.tasks worker --loglevel=info --pool=solo
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

로컬 Python 실행 대신 API와 worker까지 Docker로 실행하려면 다음 명령을 사용합니다.

```bash
docker compose up -d api worker
```

상태 확인:

```bash
docker compose ps
```

종료:

```bash
docker compose down
```

## 프로젝트 구조

```text
kidogkidog/
├── api/
│   ├── Dockerfile
│   ├── main.py                    # FastAPI 엔드포인트, 검색 API, 미디어 프록시
│   └── requirements.txt
├── data/
│   └── videos/                    # 로컬 테스트 영상 입력 경로
├── db/
│   ├── behavior_events.py         # 행동 이벤트 저장 및 조회
│   ├── connection.py              # SQLite/MySQL 연결 어댑터
│   ├── json_utils.py              # DB JSON 직렬화 유틸리티
│   ├── scenes.py                  # 장면 메타데이터 저장 및 조회
│   ├── schema.py                  # 테이블 생성 및 초기화
│   └── search_history.py          # 검색 로그와 빈출 검색어 관리
├── pipeline/
│   ├── behavior_event_extractor.py
│   ├── clip_embedder.py
│   ├── frame_extractor.py
│   ├── motion_detector.py
│   ├── query_analyzer.py
│   ├── query_suggester.py
│   ├── rag_chain.py
│   ├── s3_uploader.py
│   ├── tasks.py                   # Celery 영상 처리 작업
│   ├── vector_store.py            # ChromaDB 저장 및 검색
│   └── yolo_detector.py
├── scripts/
│   └── migrate_sqlite_to_mysql.py
├── simulator/
│   ├── chunk_splitter.py          # FFmpeg 청크 분할
│   ├── edge_simulator.py          # 청크 업로드 및 API 전송 시뮬레이터
│   └── preprocessor.py
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
db/*.db
db/*.sqlite
db/*.sqlite3
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
