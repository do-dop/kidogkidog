# 시퀀스 검증 실행 기록 — 2026-11-04

## 실행한 검사

| 검사 | 결과 | 범위 |
|---|---|---|
| `python -m unittest tests.test_sequence_query.SequenceQueryTests -v` | 4개 통과 | 질문 분류, 청크 간 순서, 근거 이벤트 연결, 날짜 제외, 3개/10개 표시 차이 |
| `notebooks/sequence-question-results.ipynb` 코드 셀 순차 실행 | 통과 | 가상 이벤트 4개가 `바닥 → 소파 → 내려옴 → 재등장`으로 정렬되고 빈 날짜에 근거 없이 답하지 않음 |
| 잠정 SmartHome-Bench 평가 CSV 읽기 | 통과 | 반려동물 핵심 4개 영상의 잠정 순서가 노트북에 표시됨 |

### 추가 실행 (main 기준 `feat/sequence-question` 브랜치, Python 3.12 임시 환경)

| 검사 | 결과 | 범위 |
|---|---|---|
| `tests.test_sequence_query` 전체 | 7개 통과 | 위 4개 + 라우팅 통합 3개: 시퀀스 질문은 프레임 검색을 건너뜀, `몇 번` 질문은 집계 경로 우선, `source_event_id`가 있으면 이벤트 근거 경로 우선 |
| `ui-react` `npm run build` | 통과 | `tsc -b` 타입 검사 포함 |

이 환경에는 `clip`(torch)을 설치하지 않아 `test_search_date_filter`, `test_question_generator`, `test_question_event_linking`은 실행하지 못했다. 같은 3개 모듈은 수정하지 않은 `origin/main`에서도 같은 이유로 실패한다.

## 실제 영상 API 검증 상태

**미실행.** `http://127.0.0.1:8000/health` 연결 실패. Docker 엔진 미실행. 프로젝트의 `data/videos`와 확인한 다운로드 위치에 시험 영상이 없었다. 검사에 사용한 Python 3.11 환경에는 전체 API 실행에 필요한 `pymysql`, `prometheus_client`, `chromadb`가 없어 라우팅 통합 테스트도 실행할 수 없었다. 이 상태에서 가상 이벤트 결과를 실제 영상 정확도로 해석하지 않는다.

## 실제 영상 판정 절차

1. 사용할 수 있는 로컬 영상 하나를 준비하고, API·worker·DB·GCS·Chroma 연결을 확인한다.
2. `simulator.edge_simulator`로 영상 작업을 등록한 뒤 worker에서 처리 완료와 DB 이벤트 생성을 확인한다. 업로드 응답만으로는 완료가 아니다.
3. 노트북 마지막 셀에 실제 `video_id`를 입력하고 `top_k=10`으로 `/query`를 호출한다.
4. `query_type=sequence`, `total_event_count`, 행동 순서, 각 근거의 재생 시각을 원본 영상과 비교한다.
5. 누락·순서 역전·설명 오류·재생 오류를 각각 기록한다. `0065`의 네 번째 행동은 화면 기본 `top_k=3`에서는 잘리므로 API와 화면을 따로 판정한다.
