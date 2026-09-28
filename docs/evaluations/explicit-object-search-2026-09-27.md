# 명시 객체 조건 검색 평가 (2026-09-27)

## 결과와 한계

명시적인 고양이/cat, 강아지/개/dog, 사람/person/human/people 조건을 후보 단계에서 검사하도록 했다. 일반 검색과 event-grounded 검색 모두, 반환된 프레임의 저장 object_labels가 조건을 만족한다. 무조건 3개를 채우지 않는다.

**실제 객체 정확도가 모두 해결된 것은 아니다.** dog_drinking_water와 two_dogs에서 강아지를 cat으로, dog_drinking_water에서 강아지를 person으로 저장한 오검출이 확인됐다. 수정 후에도 이 프레임은 라벨 조건을 통과한다. IMG_8450_2에서는 실제 고양이가 dog으로 저장된 반대 오류도 있다. 이번에는 YOLO 모델/검출/저장 데이터를 변경하지 않았으며, 객체 라벨 일치와 실제 시각적 사실을 아래 표에서 분리했다.

## 현재 흐름과 원인

- `index_frame()`의 `_normalize_object_labels()`는 YOLO label list를 쉼표 문자열로 저장한다. 예: `cat,dog`, `dog,person`, 빈 문자열. 기존 YOLO는 yolov8n, confidence 0.5이며 Chroma metadata에는 bbox/개별 confidence가 없다.
- `get_indexed_frames_in_time_window()`는 원래 metadata 전체를 유지한다. 이벤트 chunk/time 후보에도 object_labels가 있다.
- `search()`와 `search_within_frames()`도 object_labels를 반환한다. 문제는 저장/전달 누락이 아니라, 후보를 제한하지 않고 CLIP 유사도만으로 선정한 것이다.
- 일반 검색은 확장 문장별 Top K를 합치고 최고 score로 정렬한다. 명시한 객체가 없어도 유사도가 높으면 반환됐으며 Top K 이후 필터만 추가하면 낮은 순위의 일치 프레임을 놓친다.
- event 검색은 기존 chunk/time 후보 → CLIP → Top K로 실행하고, 실패하면 source_frames를 사용한다. 기존 fallback에도 객체 검사가 없었다.
- `/query`는 `results`, `evidence_items`를 그대로 반환한다. React `buildSearchResults()`는 `object_labels`를 objects 태그로 사용한다. 이번에 API/React는 변경하지 않았다.

## 실제 라벨 분포

Chroma 전체 617개 metadata를 읽었으며, 평가 대상 네 영상은 합계 325개다. 다음 값은 영상 전체 검색 범위이며, 존재/부재는 실제 영상 판독이 아니라 **저장 label** 기준이다.

| 영상 | 프레임 | cat 검출 / 미검출 | dog 검출 / 미검출 | person 검출 / 미검출 |
|---|---:|---:|---:|---:|
| IMG_8450_2 | 33 | 29 / 4 | 6 / 27 | 0 / 33 |
| dog_drinking_water | 89 | 12 / 77 | 46 / 43 | 2 / 87 |
| dog_escape | 28 | 0 / 28 | 0 / 28 | 5 / 23 |
| two_dogs | 175 | 4 / 171 | 91 / 84 | 16 / 159 |

실제 시각 확인은 전체 프레임 정답 라벨링이 아니라 전후 Top 3와 각 label 표본을 확인한 것이다. 따라서 이 표본만으로 precision/recall을 추정하지 않는다. 비교용 contact sheet는 아래에 보관했다.

- [IMG 수정 전](object-label-inspection/before_IMG_8450_2_0.jpg), [수정 후](object-label-inspection/after_IMG_8450_2_0.jpg)
- [물 마시기 수정 전 1](object-label-inspection/before_dog_drinking_water_0.jpg), [수정 전 2](object-label-inspection/before_dog_drinking_water_1.jpg), [수정 후](object-label-inspection/after_dog_drinking_water_0.jpg)
- [탈출 수정 전](object-label-inspection/before_dog_escape_0.jpg), [수정 후](object-label-inspection/after_dog_escape_0.jpg)
- [두 강아지 수정 전](object-label-inspection/before_two_dogs_0.jpg), [수정 후](object-label-inspection/after_two_dogs_0.jpg)

## 선택한 정책: hard filter + 범위 안 source_frames 재확인

1. 질문 원문에 명시된 객체만 조건으로 사용한다. 물 마시기/반려동물/급식기에는 종 조건을 추가하지 않는다. 확장 문장이 pet으로 완화돼도 원문 조건을 유지한다.
2. event 검색은 기존 chunk/time/±3초 후보를 확보한 뒤 label 조건으로 제한하고 기존 CLIP 정렬을 실행한다.
3. 일반 검색은 video/date/time 범위 metadata에서 일치 ID를 확보하고 기존 확장 문장과 CLIP 점수로 검색한다. 일반 검색을 다른 모델이나 점수식으로 재설계하지 않았다.
4. 일치 결과가 없으면 event 검색에서 같은 범위의 source_frames 전체를 조건 검사한 뒤 Top K를 적용한다. 그래도 없으면 빈 results/evidence_items와 '검출 누락 가능성'을 설명하는 응답을 반환한다. 일반 검색도 조건을 풀거나 무관한 이벤트/인덱스 대표 프레임을 붙이지 않는다.
5. 여러 객체가 명시되면 모두 검출된 프레임만 사용한다. 쉼표/리스트 라벨을 정확한 토큰으로 비교한다. 일반 자연어의 부정/OR 논리를 해석하는 기능은 아니다.

왜 rerank나 조건 해제를 선택하지 않았는가: 기존 CLIP의 높은 점수 자체가 객체 존재를 보장하지 않는 사례가 재현됐고, person 검출이 0개인 범위에서 조건을 풀면 기존의 무관한 프레임이 그대로 돌아온다. 임의 점수 보너스도 사용하지 않았다. 동일 범위 source_frames 재확인으로 일부 metadata 누락은 보완하지만, 양쪽 모두 label이 빠진 실제 객체는 반환하지 못한다. 이는 정확도 우선 정책의 재현율 손실이며 해결됐다고 주장하지 않는다. label false positive도 별도 검출 품질 개선이 필요하다.

## 측정 방법과 event 후보 수

실제 DB 이벤트, 실제 ChromaDB 거리, 기존 ViT-L/14 CLIP과 기존 query expansion을 사용해 전후 동일 질문으로 측정했다. 질문 답변 문장 생성을 위한 LLM만 비활성화했고 검색 함수는 실제 실행했다. `top_k=3`이다. event 경로는 각각 #3/#6/#5/#1을 명시적으로 넣어 비교했으며, 최신 추천 연결을 재생한 결과라고 주장하지 않는다. timestamp는 기존 청크 로컬 값이므로 같은 시간이 다른 프레임을 뜻할 수 있다. JSON에 frame_id를 보존했다.

| 영상 | event | 이벤트 범위 후보 | cat 검출 / 미검출 | person 검출 / 미검출 |
|---|---:|---:|---:|---:|
| IMG_8450_2 | #3 | 5 | 2 / 3 | 0 / 5 |
| dog_drinking_water | #6 | 18 | 3 / 15 | 0 / 18 |
| dog_escape | #5 | 28 | 0 / 28 | 5 / 23 |
| two_dogs | #1 | 90 | 3 / 87 | 7 / 83 |

## 전후 Top 3: 고양이가 보이는 장면 보여줘

| 영상 | 경로 | 전/후 | 순위 | timestamp | CLIP similarity | object_labels | 대상 검출 | 실제 프레임 확인 |
|---|---|---|---:|---:|---:|---|---|---|
| IMG_8450_2 | general | 전 | 1 | 34.0 | 0.200965 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | general | 전 | 2 | 2.0 | 0.200571 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | general | 전 | 3 | 54.0 | 0.198650 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | general | 후 | 1 | 34.0 | 0.200965 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | general | 후 | 2 | 2.0 | 0.200571 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | general | 후 | 3 | 54.0 | 0.198650 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | event | 전 | 1 | 8.5 | 0.181970 | dog | 아니오 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | event | 전 | 2 | 2.0 | 0.181912 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | event | 전 | 3 | 4.5 | 0.178854 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | event | 후 | 1 | 2.0 | 0.181912 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | event | 후 | 2 | 4.5 | 0.178854 | cat | 예 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| dog_drinking_water | general | 전 | 1 | 18.5 | 0.146692 | dog | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | general | 전 | 2 | 50.0 | 0.146258 | (빈 라벨) | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | general | 전 | 3 | 46.0 | 0.145258 | (빈 라벨) | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | general | 후 | 1 | 42.0 | 0.139267 | cat | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | general | 후 | 2 | 52.0 | 0.139016 | cat | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | general | 후 | 3 | 14.0 | 0.136483 | bowl,cat | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 전 | 1 | 50.0 | 0.146258 | (빈 라벨) | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 전 | 2 | 40.0 | 0.142998 | chair | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 전 | 3 | 30.0 | 0.140249 | (빈 라벨) | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 후 | 1 | 42.0 | 0.139267 | cat | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 후 | 2 | 52.0 | 0.139016 | cat | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 후 | 3 | 56.0 | 0.132371 | bowl,cat | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_escape | general | 전 | 1 | 36.0 | 0.168457 | bed | 아니오 | 펜스·소파 / 고양이·사람 확인 안 됨 |
| dog_escape | general | 전 | 2 | 6.0 | 0.166418 | (빈 라벨) | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | general | 전 | 3 | 24.0 | 0.165968 | (빈 라벨) | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | general | 후 | — | — | — | — | 일치 결과 없음 | 빈 결과 |
| dog_escape | event | 전 | 1 | 36.0 | 0.168457 | bed | 아니오 | 펜스·소파 / 고양이·사람 확인 안 됨 |
| dog_escape | event | 전 | 2 | 6.0 | 0.166418 | (빈 라벨) | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | event | 전 | 3 | 24.0 | 0.165968 | (빈 라벨) | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | event | 후 | — | — | — | — | 일치 결과 없음 | 빈 결과 |
| two_dogs | general | 전 | 1 | 22.0 | 0.174564 | cow,tv | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | general | 전 | 2 | 26.5 | 0.174295 | cow,tv | 아니오 | 이미지 없음: 확인 불가 |
| two_dogs | general | 전 | 3 | 66.0 | 0.173986 | tv | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | general | 후 | 1 | 90.0 | 0.163878 | cat,tv | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | general | 후 | 2 | 24.5 | 0.151022 | cat,tv | 예 | 이미지 없음: 확인 불가 |
| two_dogs | general | 후 | 3 | 72.0 | 0.150394 | cat,dog,tv | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | event | 전 | 1 | 22.0 | 0.174564 | cow,tv | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | event | 전 | 2 | 66.0 | 0.173986 | tv | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | event | 전 | 3 | 160.0 | 0.173413 | chair,person,tv | 아니오 | 사람 있음 / 고양이 확인 안 됨 |
| two_dogs | event | 후 | 1 | 90.0 | 0.163878 | cat,tv | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | event | 후 | 2 | 72.0 | 0.150394 | cat,dog,tv | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | event | 후 | 3 | 106.0 | 0.138630 | cat,dog,tv | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |

## 전후 Top 3: 사람이 함께 보이는 장면 있어?

| 영상 | 경로 | 전/후 | 순위 | timestamp | CLIP similarity | object_labels | 대상 검출 | 실제 프레임 확인 |
|---|---|---|---:|---:|---:|---|---|---|
| IMG_8450_2 | general | 전 | 1 | 2.0 | 0.221440 | cat | 아니오 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | general | 전 | 2 | 34.0 | 0.220457 | cat | 아니오 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | general | 전 | 3 | 54.0 | 0.219070 | cat | 아니오 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | general | 후 | — | — | — | — | 일치 결과 없음 | 빈 결과 |
| IMG_8450_2 | event | 전 | 1 | 8.5 | 0.197896 | dog | 아니오 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | event | 전 | 2 | 12.0 | 0.195194 | dog | 아니오 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | event | 전 | 3 | 4.5 | 0.193030 | cat | 아니오 | 고양이 있음 / 강아지·사람 확인 안 됨 |
| IMG_8450_2 | event | 후 | — | — | — | — | 일치 결과 없음 | 빈 결과 |
| dog_drinking_water | general | 전 | 1 | 42.0 | 0.197053 | cat | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | general | 전 | 2 | 40.0 | 0.195371 | chair | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | general | 전 | 3 | 2.0 | 0.190529 | dog | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | general | 후 | 1 | 24.0 | 0.171781 | dog,person | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | general | 후 | 2 | 28.0 | 0.159705 | person | 예 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 전 | 1 | 42.0 | 0.197053 | cat | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 전 | 2 | 40.0 | 0.195371 | chair | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 전 | 3 | 48.0 | 0.188474 | (빈 라벨) | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| dog_drinking_water | event | 후 | — | — | — | — | 일치 결과 없음 | 빈 결과 |
| dog_escape | general | 전 | 1 | 24.0 | 0.206391 | (빈 라벨) | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | general | 전 | 2 | 22.0 | 0.206064 | (빈 라벨) | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | general | 전 | 3 | 16.0 | 0.205492 | bed | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | general | 후 | 1 | 38.0 | 0.195733 | person | 예 | 강아지 근접 / 가림으로 사람 여부 불명확 |
| dog_escape | general | 후 | 2 | 46.0 | 0.174617 | person | 예 | 사람 있음 / 고양이 확인 안 됨 |
| dog_escape | general | 후 | 3 | 40.0 | 0.164305 | person | 예 | 사람 있음 / 고양이 확인 안 됨 |
| dog_escape | event | 전 | 1 | 24.0 | 0.206391 | (빈 라벨) | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | event | 전 | 2 | 22.0 | 0.206064 | (빈 라벨) | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | event | 전 | 3 | 16.0 | 0.205492 | bed | 아니오 | 강아지·펜스 / 고양이·사람 확인 안 됨 |
| dog_escape | event | 후 | 1 | 38.0 | 0.195733 | person | 예 | 강아지 근접 / 가림으로 사람 여부 불명확 |
| dog_escape | event | 후 | 2 | 46.0 | 0.174617 | person | 예 | 사람 있음 / 고양이 확인 안 됨 |
| dog_escape | event | 후 | 3 | 40.0 | 0.164305 | person | 예 | 사람 있음 / 고양이 확인 안 됨 |
| two_dogs | general | 전 | 1 | 6.0 | 0.238383 | dog | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | general | 전 | 2 | 96.0 | 0.238007 | (빈 라벨) | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | general | 전 | 3 | 36.5 | 0.237818 | (빈 라벨) | 아니오 | 이미지 없음: 확인 불가 |
| two_dogs | general | 후 | 1 | 160.0 | 0.227341 | chair,person,tv | 예 | 사람 있음 / 고양이 확인 안 됨 |
| two_dogs | general | 후 | 2 | 36.5 | 0.225464 | chair,person,tv | 예 | 이미지 없음: 확인 불가 |
| two_dogs | general | 후 | 3 | 38.5 | 0.225059 | chair,dog,person,tv | 예 | 이미지 없음: 확인 불가 |
| two_dogs | event | 전 | 1 | 6.0 | 0.238383 | dog | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | event | 전 | 2 | 96.0 | 0.238007 | (빈 라벨) | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | event | 전 | 3 | 134.0 | 0.228861 | (빈 라벨) | 아니오 | 강아지 있음 / 고양이·사람 확인 안 됨 |
| two_dogs | event | 후 | 1 | 160.0 | 0.227341 | chair,person,tv | 예 | 사람 있음 / 고양이 확인 안 됨 |
| two_dogs | event | 후 | 2 | 168.0 | 0.224993 | chair,dog,person,tv | 예 | 사람 있음 / 고양이 확인 안 됨 |
| two_dogs | event | 후 | 3 | 162.0 | 0.223759 | chair,person,tv | 예 | 사람 있음 / 고양이 확인 안 됨 |

## 확인된 변화

- dog_escape 사람 검색: 기존 24/22/16초 펜스·강아지 프레임 → 38/46/40초 person 라벨 프레임. 46/40초에는 실제 사람이 보인다. 38초는 강아지 근접·가림으로 사람 여부가 불명확하여 성공으로 단정하지 않는다.
- two_dogs 사람 event 검색: 기존 6/96/134초 강아지 장면 → 160/168/162초 실제 사람 장면.
- IMG_8450_2 사람 검색은 person 검출이 없어 양쪽 경로 모두 빈 결과.
- dog_drinking_water 고양이 검색의 dog-only/빈 라벨 프레임은 제거되었지만, cat 오검출 프레임도 실제 강아지여서 고양이 실제 정확도 문제는 남는다. person 일반 검색도 잘못 저장된 라벨 때문에 실제 사람으로 확인되지 않는다.
- two_dogs 고양이 검색도 cat 오검출 프레임에 실제 강아지가 보인다. label 검사만으로 해결하지 못한다.
- IMG event 고양이 검색의 기존 dog 라벨 8.5초 프레임은 실제 고양이였지만 필터에서 제외된다. 검출 누락/오분류에 따른 재현율 손실을 실제로 확인했다. 대신 cat 라벨의 실제 고양이 프레임 2개가 남는다.

## 파일 및 테스트

- `pipeline/vector_store.py`: `explicit_object_labels`, `filter_frames_by_objects`, `_object_candidate_frames` 추가. `search`, `search_within_frames`, `search_with_query_expansion`에 후보 조건 적용.
- `pipeline/rag_chain.py`: 일반/event 결과 및 event source fallback의 조건 검사. `_object_condition_unconfirmed`로 빈 결과를 안전하게 반환.
- `tests/test_explicit_object_search.py`: 신규 12개 테스트. cat/dog/person, 중립 질문, 정확한 label 토큰, 후보 단계 필터, 확장 중 원문 조건 유지, 낮은 순위의 일치 프레임 발견, 날짜 범위, 빈 결과, event source 재확인 및 범위 유지, 직접 search 검증.
- `.venv/bin/python -m unittest discover -s tests`: **총 58개 통과**(기존 46 + 신규 12). 질문 생성/연결, event/chunk/확장 검색 회귀 테스트도 통과.
- `git diff --check` 통과.

## 범위 밖 / 운영상 한계

YOLO 종/사람 오검출과 누락, GCS에 없는 일부 과거 프레임, 같은 영상의 여러 청크에서 같은 timestamp를 사용하는 기존 중복 판정, 라벨 문장 조사 오류는 수정하지 않았다. 객체 존재 필터는 소파 위/아래 등 공간 관계를 판단하지 않는다. 추천 프롬프트/validation/원본 event 보존/0점 연결 정책, 이벤트 생성, query expansion 문장 생성, CLIP 모델, chunk/±3초/React를 변경하지 않았다.

명시 객체 일반 검색은 현재 범위 metadata를 읽고 일치 후보를 평가하므로 기존 Top K 전용 조회보다 비용이 증가한다. 현재 네 영상 325프레임에서는 동작을 검증했지만 대규모 인덱스 성능 검증은 하지 않았다.
