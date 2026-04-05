# 🐾 Kidogkidog

> "당신의 펫, 지금 뭐하고 있을까?"

Kidogkidog은 하루종일 녹화된 펫캠 영상에서 원하는 반려동물의 행동을 자연어로 검색할 수 있는 **Multimodal Video RAG 시스템**입니다.

---

## 💡 프로젝트 배경

1인 가구와 맞벌이 가정이 늘면서 펫캠 사용이 보편화되었지만,  
"강아지가 간식 봉지 뜯은 게 언제야?" 같은 질문에 답하려면  
몇 시간짜리 영상을 처음부터 끝까지 직접 돌려봐야 했습니다.

**Kidogkidog** 은 이 불편함을 AI로 해결합니다.

---

## ✨ 주요 기능

- 🔍 **자연어 행동 검색** — "강아지가 물 마신 장면 보여줘"
- ⏱️ **타임스탬프 자동 반환** — 해당 장면의 시작/종료 시간 제공
- 🎬 **즉시 영상 재생** — 검색 결과 클릭 시 해당 구간 바로 재생
- 🤖 **이벤트 기반 처리** — 움직임 감지 시에만 프레임 추출, 효율적인 연산
- ☁️ **클라우드 자동 관리** — S3 업로드 및 3일 TTL 자동 삭제

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| Backend | FastAPI, Celery, RabbitMQ |
| AI/ML | CLIP, YOLO, LangChain, VLM |
| Video Processing | OpenCV, FFmpeg |
| Database | ChromaDB, SQLite, AWS S3 |
| Frontend | Streamlit |
| Infra | Docker |

---

## 🏗️ 시스템 구조

```
펫캠 영상
    ↓ 5분 단위 자동 청크 전송
FastAPI 서버
    ↓ 이벤트 감지 (OpenCV)
프레임 추출 (1fps)
    ↓
CLIP 임베딩 → ChromaDB 저장
    ↓ 자연어 쿼리 입력
코사인 유사도 검색 → Top-K 프레임 추출
    ↓
VLM 응답 생성 + 타임스탬프 반환
    ↓
Streamlit UI에서 영상 재생
```

---

## 👥 팀원

| 이름 | 역할 |
|---|---|
| 박도연 | Backend & Data Pipeline Lead |
| 김태희 | AI Application & Web Frontend Lead |

---

## 📁 프로젝트 구조

```
kidogkidog/
├── api/          # FastAPI 서버
├── pipeline/     # 영상 처리, 임베딩
├── db/           # ChromaDB, SQLite 연결
├── ui/           # Streamlit 프론트엔드
├── simulator/    # Edge 영상 전송 시뮬레이터
├── requirements.txt
└── README.md
```

---

## 📌 커밋 규칙

| 타입 | 설명 |
|---|---|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 |
| `style` | 코드 포맷 (기능 변경 없음) |
| `refactor` | 리팩토링 |
| `test` | 테스트 코드 |
| `chore` | 환경설정, 패키지 등 |

**예시**
```
feat: CLIP 이미지 임베딩 파이프라인 추가
fix: ChromaDB 연결 오류 수정
chore: requirements.txt 업데이트
docs: README 프로젝트 구조 설명 추가
```

---
 
## 🌿 브랜치 전략
 
| 브랜치 | 용도 |
|---|---|
| `main` | 최종 배포용 |
| `dev` | 통합 개발 |
| `feature/기능명` | 기능별 개발 |
 
**흐름**
```
feature/기능명 → dev → main
```
 
**예시**
```bash
git checkout -b feature/clip-embedding
git checkout -b feature/fastapi-server
```
 
- `feature/` 브랜치는 항상 `dev` 에서 분기
- 작업 완료 후 `dev` 로 PR
- 최종 완성본만 `main` 으로 머지
 