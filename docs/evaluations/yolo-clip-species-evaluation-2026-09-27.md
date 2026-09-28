# YOLO bbox + CLIP cat/dog 2차 분류 평가

평가는 읽기 전용으로 수행했다. YOLO/CLIP raw score를 평가용으로 계산했으며 Chroma, MySQL, behavior_events, production 코드는 변경하지 않았다. eligible 240개만 사용했고 stale 377개는 제외했다. GCS 원본은 읽기만 했다.

## 표본과 방법

- YOLOv8s에서 confidence ≥ 0.4인 cat/dog bbox 311개를 각기 독립 crop했다. 원 bbox, 5%, 10% padding을 비교하고 crop 좌표는 원본 이미지 픽셀 기준 xyxy로 clamp했다. 한 프레임의 여러 detection은 각각 분류해 종을 영상 전체에 고정하지 않았다.
- 기존 ViT-L/14와 프로젝트의 CLIP 전처리를 사용했다. 이미지와 텍스트 embedding을 L2 normalize한 뒤 cosine similarity를 비교했다. Prompt A는 `a cat` / `a dog`, B는 `a photo of a cat` / `a photo of a dog`, C는 photo/pet/domestic 3개 prompt embedding 평균이다.
- GT는 기존 107개 검수와 canary 15개 검수를 frame_id로 합쳐 중복 제거한 110개다(고양이 양성 21, 강아지 양성 77). canary 판단은 기존 판단을 대체했다. frame-level multilabel 점수라 한 프레임에 cat과 dog가 함께 있을 수 있다.
- YOLO baseline은 threshold 0.5 object_labels. Temporal은 기존 same-chunk ±4초 majority 결과다. CLIP은 0.4 이상 YOLO cat/dog bbox마다 판정하고, frame의 species label은 detection 판정들의 합집합으로 만들었다.
- TP/FP/FN은 각 class의 프레임 존재 여부 기준이다. detection bbox 정답 주석은 없으므로 localization 품질/IoU 자체는 평가하지 못했다.

## YOLO baseline 및 temporal 결과

| 방식 | cat TP/FP/FN; P/R/F1 | dog TP/FP/FN; P/R/F1 | macro F1 |
|---|---|---|---:|
| YOLOv8s threshold 0.5 | cat: 20/12/1 (P 0.625, R 0.952, F1 0.755) | dog: 45/6/32 (P 0.882, R 0.584, F1 0.703) | 0.729 |
| Temporal ±4초 majority | cat: 21/5/0; 0.808/1.000/0.894 | dog: 47/6/30; 0.887/0.610/0.723 | 0.808 |

Temporal은 macro F1을 0.729에서 0.808로 높였지만, 오류 비교에서는 기존 오답 12프레임을 개선하는 대신 5프레임을 새로 틀렸다. 본 사용자의 이전 판단대로 production 적용 대상으로 보지 않았고, CLIP은 대체 후보가 될 수 있는지 비교했다.

## CLIP 정책 성능

아래 LOOV(leave-one-video-out)는 매번 3개 영상에서 padding/prompt/margin을 고르고 나머지 1개 영상만 평가했다. 네 영상뿐이므로 안정적인 일반화 증명은 아니다.

| 정책 | cat TP/FP/FN; P/R/F1 | dog TP/FP/FN; P/R/F1 | macro F1 | 기존 오류 개선 / 새 오답 |
|---|---|---|---:|---:|
| A: CLIP으로 항상 재분류 | cat 21/1/0; 0.955/1.000/0.977 | dog 61/2/16; 0.968/0.792/0.871 | 0.924 | 22 / 1 |
| B: YOLO/CLIP 일치만 확정 | cat 21/1/0; 0.955/1.000/0.977 | dog 51/2/26; 0.962/0.662/0.785 | 0.881 | 20 / 0 |
| C: CLIP margin 기준 | cat 21/1/0; 0.955/1.000/0.977 | dog 61/2/16; 0.968/0.792/0.871 | 0.924 | 22 / 1 |

LOOV에서 Policy A/C는 macro F1 0.924로 YOLO baseline 0.729 및 temporal 0.808보다 높았다. Cat/dog 합산 기존 오류 22 frame을 개선했고 1 frame을 새로 틀렸다(`dog_escape` 4초). Policy B는 개선 20, 새 오답 0이지만 dog recall이 .792에서 .662로 낮아졌다. LOOV에서 선택된 설정들을 합치면 22개 bbox detection이 unresolved였고, 9개 프레임에서 후보 bbox가 있지만 확정 species가 없었다. 동일 표본에서 고른 Policy B 최고 설정에서는 unresolved가 39 bbox / 19 frame으로 더 많았으며, 이는 full-sample 최적 설정의 보조 수치다.

네 fold 모두 A/C에서 prompt A(`a cat`/`a dog`)를 선택했고 padding은 5% 또는 10%였다. C의 margin은 네 fold 모두 0을 선택했다. 즉 작은 정답 표본에서 margin으로 불확실 사례를 거르는 것이 held-out 성능을 올리지는 못했다.

## 설정 민감도 및 margin

전체 110개를 보고 설정을 선택하고 같은 110개로 평가한 최고값은 낙관적 참고치다: A는 padding 10%, prompt A에서 macro F1 0.936(cat 21/0/0, dog 61/2/16), 기존 오류 22개 개선·새 오답 0개였다. B 최고는 padding 5%, prompt A로 macro F1 0.892(cat 21/1/0, dog 51/2/26), unresolved detection 39개/프레임 19개였다. 이 full-sample 최적값을 성능 보장으로 해석하면 안 된다.

Prompt A가 가장 자주 상위에 있었지만 세 prompt 구성이 크게 다른 결론을 만들지는 않았다. padding은 일부 프레임에서 영향을 줬고 10%가 전체 표본 최적이었으나 LOOV에서는 5%/10%가 영상별로 갈렸다. 따라서 이 4개 영상만으로 고정 설정을 확정하기 어렵다.

Policy C의 margin sweep 예시(no padding, prompt A):

| 요구 margin | macro F1 | cat F1 | dog F1 | unresolved detection / frame | 개선 / 새 오답 |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.909 | 0.955 | 0.863 | 0 / 0 | 21 / 1 |
| 0.005 | 0.909 | 0.955 | 0.863 | 5 / 2 | 21 / 1 |
| 0.010 | 0.909 | 0.955 | 0.863 | 9 / 4 | 21 / 1 |
| 0.020 | 0.923 | 1.000 | 0.847 | 15 / 8 | 20 / 0 |
| 0.030 | 0.923 | 1.000 | 0.847 | 26 / 12 | 20 / 0 |
| 0.050 | 0.689 | 0.950 | 0.429 | 175 / 103 | 16 / 24 |
| 0.100 | 0.000 | 0.000 | 0.000 | 311 / 200 | 13 / 60 |

이 표에서 높은 margin은 오히려 올바른 강아지 판정을 많이 unresolved로 보냈다. Full-sample에서 0.02~0.03은 새 오답 없이 F1 .923을 내는 설정도 있었지만, 동일 표본에서 고른 결과이고 LOOV가 margin 0을 선택했으므로 운영 threshold 근거로 삼지 않는다.

## 대표 오류와 detection 누락

| 사례 | YOLO baseline / raw bbox | CLIP 결과 (prompt A; bbox별) |
|---|---|---|
| IMG_8450_2 / 66.0초 (GT cat) | dog; bbox 1: YOLO dog 0.882 → CLIP cat (margin 0.053)<br>bbox 2: YOLO dog 0.529 → CLIP cat (margin 0.042)<br>bbox 3: YOLO cat 0.470 → CLIP cat (margin 0.044) |
| dog_drinking_water / 12.5초 (GT dog) | cat; bbox 1: YOLO cat 0.730 → CLIP dog (margin 0.041) |
| dog_drinking_water / 14.5초 (GT dog) | cat; bbox 1: YOLO cat 0.627 → CLIP dog (margin 0.043) |
| dog_escape / 12.0초 (GT dog) | empty; bbox 1: YOLO dog 0.426 → CLIP dog (margin 0.033) |

- IMG_8450_2 66초의 실제 고양이는 YOLO threshold-0.5 label에서 dog였지만, 세 prompt와 세 padding 모두 해당 bbox들을 cat으로 분류했다. CLIP이 대표 cat↔dog 오류를 바로잡았다.
- dog_drinking_water 12.5초와 14.5초는 YOLO cat bbox를 CLIP이 dog로 분류했다. 세 prompt 및 padding에서 일관되게 복구했다.
- dog_escape 12초의 dog .426은 baseline threshold 0.5 label에서 빠졌지만 raw 후보 bbox는 0.4 이상이었다. CLIP은 dog로 판단했고, Policy A 및 YOLO/CLIP 일치형 B에서 다시 포함할 수 있었다. 이는 species 재분류이면서 label threshold 아래 detection의 활용 사례다.
- 이 네 대표 사례는 모두 CLIP 판정으로 개선됐다. 전체 LOOV에서는 이득과 함께 `dog_escape` 4초라는 새 오답 1건도 있었다.

검수된 실제 종 양성 프레임의 후보 bbox 기준 분해:

| 실제 종 | 양성 프레임 | cat/dog 후보 bbox 없음(≥0.4) | 반대 종만 검출 | 같은 종 .4–.5 | 같은 종 object_label ≥.5 |
|---|---:|---:|---:|---:|---:|
| cat | 21 | 0 | 0 | 1 | 20 |
| dog | 77 | 16 | 10 | 6 | 45 |

강아지 양성 77프레임 가운데 16개(20.8%)에는 cat/dog 후보 bbox가 0.4 이상 하나도 없어 crop CLIP으로 복구할 수 없다. 이는 검출/후보 위치 실패의 proxy이며, GT bbox가 없어서 실제 localization 실패와 종 분류 실패를 완전히 분리하지는 못한다. 10개는 반대 species만 검출됐고 CLIP이 대표 오류처럼 분류를 고칠 수 있는 영역이다. 6개는 맞는 dog 후보가 0.4–0.5에 있어 현재 object_labels threshold가 놓치지만 raw detections에서 CLIP을 돌리면 일부 회복된다. 고양이는 21개 양성 모두 후보가 있었고 그중 1개는 올바른 cat 후보가 0.4–0.5였다.

따라서 unresolved 중에서도 후보 bbox 자체가 없는 경우와, bbox는 있으나 Policy B/C가 종을 확정하지 않은 경우를 구분해야 한다. 후자는 검색에서 누락을 만들 수 있다.

## 검색/데이터 구조 시사점

현재 실험은 bbox별 YOLO raw class/confidence/bbox를 보존한 채 별도의 CLIP 결과를 계산했다. 향후 설계로는 질문에 제시한 `species_resolution` 배열을 raw `object_detections`와 분리해 둘 수 있다. `object_labels`는 합의된 해석값만 반영하되, raw detection을 덮어쓰면 안 된다. 이 평가는 해당 구조를 구현하거나 저장소에 반영하지 않았다.

고정된 최종 species만 hard filter하면 잘못된 positive 검색 결과가 줄 수 있지만, unresolved를 제외하면 진짜 cat/dog frame이 빠져 recall이 낮아진다. 후보 정책으로는 (1) strict exclusion: 정밀도 우선, 미해결 recall 저하, (2) unresolved를 CLIP similarity로 query-time 재검증하거나 보조 후보군에 포함: recall을 지키되 별도 검증/순위 정책 필요가 있다. 현재 결과만으로 production hard filter 동작을 바꾸는 것은 권하지 않는다.

## 판단

CLIP bbox 2차 분류는 이 110개 검수 subset에서 YOLO 단일-frame species 오류에 효과가 있었고, temporal 결과보다 LOOV macro F1이 높았다. 특히 “반대 species로 검출된 bbox”를 고치는 데 유망하다. 다만 영상이 4개뿐이고, 110개 표본은 기존 평가/ canary에서 모아진 편의 표본이며, margin·padding 선택도 제한된 그룹에서 이뤄졌다. 또한 16개 dog 양성 프레임처럼 후보 bbox가 없는 누락은 해결하지 못했다.

현재 근거로는 추가 독립 영상/사전 고정 설정 검증 없이 object_labels나 검색 hard filter에 바로 적용하지 않는 것이 안전하다. 다음 검증 단계에서 더 다양한 영상으로 prompt/padding을 사전 고정하고, 객체별 bbox GT를 일부 만들어 localization과 species classification을 분리 평가하는 것이 필요하다.

## 산출물

- 상세 JSON: `docs/evaluations/yolo-clip-species-evaluation-2026-09-27.json` (240 eligible frame, 311 detection, 933 crop variant의 similarity 및 정책별 frame 결과 포함)
- 보고서: `docs/evaluations/yolo-clip-species-evaluation-2026-09-27.md`
- 평가 장치: CPU, 기존 cached ViT-L/14. GCS 원본 240개만 읽었고 실패 0. GCS 인증 관련 quota/PQC 경고는 있었으나 다운로드와 평가 결과에 오류는 없었다.
