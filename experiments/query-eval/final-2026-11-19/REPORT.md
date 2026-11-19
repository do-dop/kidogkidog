# 질의 성능 평가 보고서 (full)

> 이 문서의 모든 수치는 아래 **테스트 조건**에서 실제로 측정·계산된 값이다. `n/a`는 평가할 수 있는 라벨된 케이스가 없다는 뜻이며 0%가 아니다.

## 1. 테스트 조건

- **측정 시각(UTC)**: 2026-11-19 01:52:54
- **git commit**: 91fb15f
- **python**: 3.9.6
- **실행 backend**: http
- **실행 base_url**: https://kidog-api-staging-ptiwpgodca-du.a.run.app
- **실행 pipeline**: full
- **Top-K(요청)**: 5
- **지연시간 반복 횟수**: 1
- **Hit/Recall 평가 K**: 1, 3, 5
- **구간 일치 기준**: IoU ≥ 0.5
- **Semantic 구간 필드**: playback
- **Sequence 구간 필드**: event
- **근거 평가 상위 N**: 3
- **데이터셋**: evaluation/data/labeling.final.csv (sha256 f3d46e60b4f0aa9a)
- **전체 케이스**: 25
- **정확도 평가 대상(verified)**: 23
- **라벨 상태별**: {"semantic": {"verified": 18}, "sequence": {"verified": 5}, "aggregate": {"excluded": 2}}
- **검색 범위**: {"전체 영상": 18, "영상 한정": 5, "정답에 등장한 영상 수": 6}
- **요청 오류**: 0

> 라벨이 `verified`가 아닌 케이스 2건은 정확도 지표에서 제외했다(질의 실행·지연시간·근거 `Unverifiable`에만 반영).

> **미평가 유형: aggregate** — verified 라벨이 없어 정확도 지표를 계산하지 않았다(해당 유형의 `n/a`는 0%가 아님).

## 2. 지표 요약


### Semantic/영상 단위(전체 영상 검색)

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| Hit@1 | 0.611 | 18 |  ⚠표본 작음 |
| Recall@1 | 0.611 | 18 |  ⚠표본 작음 |
| Hit@3 | 0.667 | 18 |  ⚠표본 작음 |
| Recall@3 | 0.667 | 18 |  ⚠표본 작음 |
| Hit@5 | 0.667 | 18 |  ⚠표본 작음 |
| Recall@5 | 0.667 | 18 |  ⚠표본 작음 |
| MRR | 0.639 | 18 |  ⚠표본 작음 |

### Semantic/구간 단위(전체 영상 검색)

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| Hit@1 | 0.056 | 18 | IoU≥0.5 ⚠표본 작음 |
| Recall@1 | 0.056 | 18 |  ⚠표본 작음 |
| Hit@3 | 0.278 | 18 | IoU≥0.5 ⚠표본 작음 |
| Recall@3 | 0.278 | 18 |  ⚠표본 작음 |
| Hit@5 | 0.278 | 18 | IoU≥0.5 ⚠표본 작음 |
| Recall@5 | 0.278 | 18 |  ⚠표본 작음 |
| MRR | 0.167 | 18 |  ⚠표본 작음 |
| Overlap Hit@1 | 0.167 | 18 | 정답 구간과 1프레임이라도 겹치면 적중(IoU보다 느슨함) ⚠표본 작음 |
| Overlap Hit@3 | 0.611 | 18 | 정답 구간과 1프레임이라도 겹치면 적중(IoU보다 느슨함) ⚠표본 작음 |
| Overlap Hit@5 | 0.667 | 18 | 정답 구간과 1프레임이라도 겹치면 적중(IoU보다 느슨함) ⚠표본 작음 |
| 1위 재생 구간 평균 길이(초) | 6.393 | 14 | 정답 구간 평균 8.4초. 재생 구간이 길면 IoU가 낮아짐 ⚠표본 작음 |

### Semantic/구간 단위(영상 한정 검색)

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| Hit@1 | n/a | 0 | IoU≥0.5 |
| Recall@1 | n/a | 0 |  |
| Hit@3 | n/a | 0 | IoU≥0.5 |
| Recall@3 | n/a | 0 |  |
| Hit@5 | n/a | 0 | IoU≥0.5 |
| Recall@5 | n/a | 0 |  |
| MRR | n/a | 0 |  |
| Overlap Hit@1 | n/a | 0 | 정답 구간과 1프레임이라도 겹치면 적중(IoU보다 느슨함) |
| Overlap Hit@3 | n/a | 0 | 정답 구간과 1프레임이라도 겹치면 적중(IoU보다 느슨함) |
| Overlap Hit@5 | n/a | 0 | 정답 구간과 1프레임이라도 겹치면 적중(IoU보다 느슨함) |
| 1위 재생 구간 평균 길이(초) | n/a | 0 |  |

### Sequence

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| Exact Match | 0.000 | 5 |  ⚠표본 작음 |
| Order Accuracy(쌍 단위) | 0.000 | 5 | 정답 단계가 2개 이상인 케이스의 단계 쌍. 못 찾은 단계가 낀 쌍은 오답 ⚠표본 작음 |
| Temporal Localization(IoU≥0.5인 단계 비율) | 0.059 | 17 | 분모=정답 단계 수 ⚠표본 작음 |
| Temporal Localization(평균 최고 IoU) | 0.157 | 5 |  ⚠표본 작음 |

### Aggregation

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| Answer Coverage(계산 가능 응답 비율) | n/a | 0 | 계산 불가 응답은 0회로 세지 않음 |
| Exact Count Accuracy(응답한 질문 기준) | n/a | 0 |  |
| Count MAE(응답한 질문 기준) | n/a | 0 | 단위: 회 |
| End-to-end Exact(계산 불가를 오답으로 봄) | n/a | 0 | Coverage×Accuracy와 같음 |

### Grounding/semantic

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| Correct | 0.056 | 18 | 1건 ⚠표본 작음 |
| Partial | 0.167 | 18 | 3건 ⚠표본 작음 |
| Incorrect | 0.611 | 18 | 11건 ⚠표본 작음 |
| Unverifiable | 0.167 | 18 | 3건 ⚠표본 작음 |

### Grounding/sequence

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| Correct | 0.600 | 5 | 3건 ⚠표본 작음 |
| Partial | 0.200 | 5 | 1건 ⚠표본 작음 |
| Incorrect | 0.200 | 5 | 1건 ⚠표본 작음 |
| Unverifiable | 0.000 | 5 | 0건 ⚠표본 작음 |

### Grounding/aggregate

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| Correct | 0.000 | 2 | 0건 ⚠표본 작음 |
| Partial | 0.000 | 2 | 0건 ⚠표본 작음 |
| Incorrect | 0.000 | 2 | 0건 ⚠표본 작음 |
| Unverifiable | 1.000 | 2 | 2건 ⚠표본 작음 |

### Latency/semantic

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| P50(ms) | 3,860.5 | 18 | 표본 20 미만: P95 신뢰 낮음 |
| P95(ms) | 23,585.0 | 18 | 표본 20 미만: P95 신뢰 낮음 |

### Latency/sequence

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| P50(ms) | 171.9 | 5 | 표본 20 미만: P95 신뢰 낮음 |
| P95(ms) | 927.8 | 5 | 표본 20 미만: P95 신뢰 낮음 |

### Latency/aggregate

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| P50(ms) | 166.0 | 2 | 표본 20 미만: P95 신뢰 낮음 |
| P95(ms) | 476.8 | 2 | 표본 20 미만: P95 신뢰 낮음 |

### Latency/all

| 지표 | 값 | n | 비고 |
|---|---:|---:|---|
| P50(ms) | 3,576.7 | 25 | nearest-rank |
| P95(ms) | 4,929.8 | 25 | nearest-rank |

근거 판정은 **자동 사전 판정**(정답 구간과 시간 겹침)이다. `grounding_review.csv`에 수동 판정을 적고 `--grounding-review`로 다시 집계하면 확정 값이 된다.

## 3. 질문 유형별 범위

| 유형 | 검색 범위 | verified | 그 외 |
|---|---|---:|---:|
| aggregate | single_video | 0 | 2 |
| semantic | all_videos | 18 | 0 |
| sequence | single_video | 5 | 0 |

## 해석 시 주의

- Semantic 영상 단위는 결과 카드(영상별 1장으로 묶인 목록) 기준이고, 구간 단위는 묶기 전 개별 장면(`related_scenes`)을 점수순으로 본 값이다.
- Hit@K 계열은 요청 Top-K 이하의 K에서만 계산한다. 요청 Top-K보다 큰 K를 만들어 내지 않는다.
- Sequence 카드의 시간은 이벤트 시작·종료(60초 청크 안 상대 시간)이며, 연속된 같은 행동은 한 카드로 묶여 정답 단계 구간과 달라질 수 있다.
- Aggregation은 시스템이 0회를 답하지 않고 '계산 불가'로 응답하므로, 정답이 0회인 질문은 Coverage 분모에 들어가 미응답으로 집계된다.
- 지연시간은 질의를 순차 실행해 측정했으며 첫 호출의 모델 로딩(cold start)이 포함될 수 있다.

## 실패 사례

| case_id | 유형 | 질문 | 원인 |
|---|---|---|---|
| sem-smartbench_0642-4 | semantic | 개가 계속 연못에서 헤엄침 장면 보여줘 | 정답 구간과 IoU≥0.5인 장면이 상위 5개에 없음 |
| sem-smartbench_0608-1 | semantic | 강아지가 수영장 가장자리에서 물에 들어가 헤엄침 장면 보여줘 | 정답 구간과 IoU≥0.5인 장면이 상위 5개에 없음 |
| sem-smartbench_0608-2 | semantic | 강아지가 얕은 곳에서 계단으로 나와 데크로 올라옴 장면 보여줘 | 정답 구간과 IoU≥0.5인 장면이 상위 5개에 없음 |
| sem-smartbench_0608-4 | semantic | 강아지가 물에 뛰어들어 헤엄침 장면 보여줘 | 정답 구간과 IoU≥0.5인 장면이 상위 5개에 없음 |
| seq-smartbench_0642 | sequence | 강아지 행동을 시간순으로 알려줘 | 정답 4단계 중 1단계 일치, 예측 3단계 |
| seq-smartbench_0608 | sequence | 강아지 행동을 시간순으로 알려줘 | 정답 4단계 중 0단계 일치, 예측 4단계 |
| sem-smartbench_0065-1 | semantic | 검은 개가 소파에서 내려감 장면 보여줘 | 정답 구간과 IoU≥0.5인 장면이 상위 5개에 없음 |
| sem-smartbench_0065-2 | semantic | 검은 개가 소파에 한 번 오르고 거실 바닥을 달려 지나감 장면 보여줘 | 정답 구간과 IoU≥0.5인 장면이 상위 5개에 없음 |
| sem-smartbench_0188-1 | semantic | 강아지가 울타리 앞에 서서 앞발을 울타리에 걸침 장면 보여줘 | 정답 영상 ['smartbench_0188']이 상위 5개 카드에 없음 (1위=dog_escape) |
| sem-smartbench_0188-2 | semantic | 강아지가 울타리 위로 올라타 넘어가려 함 장면 보여줘 | 정답 영상 ['smartbench_0188']이 상위 5개 카드에 없음 (1위=None) |
| sem-smartbench_0188-3 | semantic | 강아지가 울타리에 다시 올라타 넘어가려 함 장면 보여줘 | 정답 영상 ['smartbench_0188']이 상위 5개 카드에 없음 (1위=None) |
| sem-IMG_8450_2-1 | semantic | 고양이가 오른쪽에서 걸어와 급식기 쪽으로 이동함 장면 보여줘 | 정답 영상 ['IMG_8450_2']이 상위 5개 카드에 없음 (1위=smartbench_0142) |
| sem-IMG_8450_2-2 | semantic | 고양이가 방 가운데 러그에 서서 주변을 둘러봄 장면 보여줘 | 정답 영상 ['IMG_8450_2']이 상위 5개 카드에 없음 (1위=smartbench_0065) |
| sem-IMG_8450_2-3 | semantic | 흰 고양이가 러그 위에서 카메라를 보며 울다가 고개를 숙임 장면 보여줘 | 정답 영상 ['IMG_8450_2']이 상위 5개 카드에 없음 (1위=None) |
| sem-dog_drinking_water-1 | semantic | 개가 초록 그릇에 고개를 숙이고 물을 마심 장면 보여줘 | 정답 구간과 IoU≥0.5인 장면이 상위 5개에 없음 |
| seq-smartbench_0065 | sequence | 강아지 행동을 시간순으로 알려줘 | 정답 3단계 중 0단계 일치, 예측 5단계 |
| seq-smartbench_0188 | sequence | 강아지 행동을 시간순으로 알려줘 | 정답 3단계 중 0단계 일치, 예측 3단계 |
| seq-IMG_8450_2 | sequence | 강아지 행동을 시간순으로 알려줘 | 정답 3단계 중 0단계 일치, 예측 3단계 |

## 서비스 적재 확인 (응답에서 관측)

정답 영상이 서비스(스테이징)에 분석·적재되지 않았다면 그 영상의 검색 실패는 검색 품질이 아니라 데이터 부재다.

| 영상 | 검색 응답에 이 영상 장면이 나온 semantic 질문 수 | 순서 질문의 분석 이벤트 수 |
|---|---:|---:|
| IMG_8450_2 | 0/3 | 3 |
| dog_drinking_water | 1/1 | n/a |
| smartbench_0065 | 3/3 | 5 |
| smartbench_0188 | 0/3 | 3 |
| smartbench_0608 | 4/4 | 4 |
| smartbench_0642 | 4/4 | 3 |
