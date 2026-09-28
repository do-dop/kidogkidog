# YOLOv8 cat/dog/person threshold evaluation — 2026-09-27

## 1. 표본과 판정 방법

총 107개 실제 프레임을 사용했다. 이전 육안 확인 표본 32개를 포함하고, 분포를 늘린 75개를 추가했다. 네 영상 모두에서 프레임을 열어 확인했으며, object_labels만을 정답으로 사용하지 않았다. 고양이는 IMG_8450_2에 21개로만 나타나고, 사람 정답은 6개뿐이어서 클래스·영상 다양성이 제한적이다. dog_escape의 삽입 이미지 사람 프레임 1개와 two_dogs의 일러스트 프레임 1개는 해당 class의 정답을 모호 사례로 제외했다.

| 영상 | 표본 | GT cat | GT dog | GT person |
|---|---:|---:|---:|---:|
| IMG_8450_2 | 21 | 21 | 0 | 0 |
| dog_drinking_water | 32 | 0 | 32 | 0 |
| dog_escape | 28 | 0 | 19 | 4 |
| two_dogs | 26 | 0 | 22 | 2 |

측정 단위는 **프레임별 class 존재 여부**다. 한 프레임에서 해당 class detection이 하나라도 threshold 이상이면 양성으로 계산했다. bbox IoU 기준 mAP가 아니라 현재 `object_labels` 소비 방식에 맞춘 지표다. 모델은 `conf=0.001`로 한 번 추론하고 동일 NMS 결과를 다섯 threshold에서 재분류했다. 모든 threshold 비교에서 프레임과 raw detection은 같다.

## 2–3. threshold별 class 성능

각 셀은 `TP/FP/FN`와 precision/recall/F1이다.

### yolov8n

평균 추론 21.41 ms/frame, 중앙값 21.74 ms/frame (CPU, batch 1, 앞선 5회 warm-up 제외).

| threshold | cat | dog | person |
|---:|---|---|---|
| 0.4 | 18/11/3<br>P=0.621 R=0.857 F1=0.720 | 33/7/40<br>P=0.825 R=0.452 F1=0.584 | 6/4/0<br>P=0.600 R=1.000 F1=0.750 |
| 0.5 | 17/9/4<br>P=0.654 R=0.810 F1=0.723 | 29/6/44<br>P=0.829 R=0.397 F1=0.537 | 6/3/0<br>P=0.667 R=1.000 F1=0.800 |
| 0.6 | 16/3/5<br>P=0.842 R=0.762 F1=0.800 | 22/4/51<br>P=0.846 R=0.301 F1=0.444 | 6/2/0<br>P=0.750 R=1.000 F1=0.857 |
| 0.7 | 15/2/6<br>P=0.882 R=0.714 F1=0.789 | 16/2/57<br>P=0.889 R=0.219 F1=0.352 | 6/0/0<br>P=1.000 R=1.000 F1=1.000 |
| 0.8 | 7/0/14<br>P=1.000 R=0.333 F1=0.500 | 10/1/63<br>P=0.909 R=0.137 F1=0.238 | 6/0/0<br>P=1.000 R=1.000 F1=1.000 |

### yolov8s

평균 추론 41.44 ms/frame, 중앙값 41.46 ms/frame (CPU, batch 1, 앞선 5회 warm-up 제외).

| threshold | cat | dog | person |
|---:|---|---|---|
| 0.4 | 21/12/0<br>P=0.636 R=1.000 F1=0.778 | 49/7/24<br>P=0.875 R=0.671 F1=0.760 | 6/4/0<br>P=0.600 R=1.000 F1=0.750 |
| 0.5 | 20/10/1<br>P=0.667 R=0.952 F1=0.784 | 43/7/30<br>P=0.860 R=0.589 F1=0.699 | 6/3/0<br>P=0.667 R=1.000 F1=0.800 |
| 0.6 | 19/5/2<br>P=0.792 R=0.905 F1=0.844 | 38/7/35<br>P=0.844 R=0.521 F1=0.644 | 6/3/0<br>P=0.667 R=1.000 F1=0.800 |
| 0.7 | 18/4/3<br>P=0.818 R=0.857 F1=0.837 | 35/7/38<br>P=0.833 R=0.479 F1=0.609 | 6/1/0<br>P=0.857 R=1.000 F1=0.923 |
| 0.8 | 18/0/3<br>P=1.000 R=0.857 F1=0.923 | 28/4/45<br>P=0.875 R=0.384 F1=0.533 | 6/0/0<br>P=1.000 R=1.000 F1=1.000 |

## 4. 대표 오분류 사례: 동일 프레임 n vs s

bbox는 `[x1,y1,x2,y2]` 픽셀 좌표다.

| 실제 장면 / frame | yolov8n (class conf bbox) | yolov8s (class conf bbox) | 결과 |
|---|---|---|---|
| 실제 cat, n에서 dog .819<br>`IMG_8450_2__petcam_20260926_173123_000_frame_8.50` | dog 0.819 [547.8, 494.5, 1009.8, 929.0]<br>dog 0.112 [681.3, 504.0, 1005.8, 746.4] | cat 0.887 [547.5, 494.5, 898.0, 925.3]<br>cat 0.786 [675.3, 502.9, 1068.2, 755.4] | s는 cat으로 교정 |
| 실제 dog, n에서 cat .765<br>`two_dogs__two_dogs_000_frame_000046` | cat 0.765 [0.4, 526.9, 432.6, 1046.1]<br>cat 0.008 [0.0, 531.3, 211.0, 1041.5] | dog 0.865 [0.7, 528.1, 426.5, 1067.8]<br>person 0.002 [445.5, 291.2, 600.1, 381.8] | s는 dog로 교정 |
| 실제 dog, n에서 person .700<br>`dog_drinking_water__petcam_20260926_173602_001_frame_28.00` | person 0.700 [0.0, 130.3, 112.8, 596.8]<br>dog 0.162 [133.3, 3.3, 850.3, 630.2] | dog 0.644 [17.6, 1.6, 892.7, 626.6]<br>dog 0.197 [0.4, 125.4, 115.7, 589.7] | s에서 person score는 .5 미만, dog가 상위 |
| 실제 dog, n에서 dog .420<br>`dog_drinking_water__petcam_20260926_173602_001_frame_36.00` | dog 0.420 [141.4, 1.3, 607.7, 521.7]<br>dog 0.045 [250.2, 238.9, 589.2, 518.4] | dog 0.826 [125.6, 2.9, 601.2, 538.7]<br>cat 0.231 [245.8, 139.5, 610.7, 562.8] | s는 dog confidence를 높임 |

### 교차 오분류 수 (프레임 단위)

| 모델 / threshold | 실제 cat → dog 검출 | 실제 dog → cat 검출 | 동물만 있는 프레임에서 person 오검출 |
|---|---:|---:|---:|
| yolov8n / 0.4 | 6 | 9 | 4 |
| yolov8n / 0.5 | 6 | 8 | 3 |
| yolov8n / 0.6 | 4 | 2 | 2 |
| yolov8n / 0.7 | 2 | 1 | 0 |
| yolov8n / 0.8 | 1 | 0 | 0 |
| yolov8s / 0.4 | 4 | 12 | 3 |
| yolov8s / 0.5 | 4 | 10 | 2 |
| yolov8s / 0.6 | 4 | 5 | 2 |
| yolov8s / 0.7 | 4 | 4 | 1 |
| yolov8s / 0.8 | 2 | 0 | 0 |

표본에서 s는 네 대표 오류 모두 class를 바로잡았다. 그러나 전체 집계에서는 오류가 남았다. 예를 들어 s의 0.5에서 dog-only 프레임 10개에서 cat이 검출되고, cat-only 프레임 4개에서 dog가 검출됐다. s가 모든 장면에서 더 잘 분류한다고 결론 내릴 수는 없다.

## 5. threshold trade-off

n에서 threshold를 0.5→0.8로 올리면 FP는 cat 9→0, dog 6→1, person 3→0으로 줄었지만 FN은 cat 4→14, dog 44→63, person 0→0으로 늘었다. s는 FP cat 10→0, dog 7→4, person 3→0; FN cat 1→3, dog 30→45, person 0→0이다. 특히 dog는 높은 threshold에서 누락이 크게 증가하므로 precision만 보고 0.8을 선택하면 부적절하다. 낮은 threshold는 dog recall을 보존하지만 cat/person false positive가 추천 생성과 검색 hard filter에 전파된다.

## 6. 추론 비용

| 모델 | 평균 ms/frame | 중앙값 ms/frame | 617 frames 환산(직렬 평균 추론) |
|---|---:|---:|---:|
| yolov8n | 21.41 | 21.74 | 13.2초 |
| yolov8s | 41.44 | 41.46 | 25.6초 |

s는 n보다 평균 1.94배 느리고 프레임당 20.03 ms 더 걸렸다. 추출 간격이 약 2초/frame이면 단일 요청 추론은 그보다 훨씬 짧다. 이 측정은 Apple arm64 CPU, batch 1이며 운영 worker 하드웨어와 동시성에서는 달라질 수 있다.

## 7. 평가 기반 후보 조합

현재 표본에서 class별 최고 F1을 고르면 **yolov8s + cat 0.8 / dog 0.4 / person 0.8**이다. 이 때 F1은 cat 0.923 (P=1.000, R=0.857), dog 0.760 (P=0.875, R=0.671), person 1.000 (P=1.000, R=1.000; 정답 6개뿐)이다. macro F1은 약 0.894다. n 최선 class별 조합(cat 0.6 F1 .800, dog 0.4 F1 .584, person 0.7 F1 1.000)의 macro F1은 약 0.795다.

다음 검증 후보로는 **yolov8s + class별 threshold**가 가장 근거가 있다. 공통 threshold는 class별 confidence와 오류 비용 차이를 반영하지 못한다. 다만 이 threshold는 같은 소표본으로 선택한 in-sample 값이라 낙관 편향이 있다. 운영 적용 전 영상 단위로 분리한 별도 holdout에서 다시 검증해야 한다. cat은 한 영상에만 있고 person은 정답 6개뿐이므로 현재 평가만으로 production threshold를 확정할 수 없다. Hard filter에서 누락 비용이 더 크면 cat threshold를 낮추고, 잘못된 cat 추천 방지가 더 중요하면 0.8이 유리하다.

## 8. confidence/bbox 저장 제안 (설계만)

- **ChromaDB frame metadata:** 기존 `object_labels` 문자열은 검색 조건/호환성을 위해 유지한다. 추가 scalar JSON 문자열 `object_detections_json`에 `[{"label":"dog","confidence":0.84,"bbox":[x1,y1,x2,y2]}]`를 저장한다. Chroma metadata는 단순 scalar가 안전하고 list/dict를 metadata 값으로 직접 저장하기 어렵다. confidence 검색이 빈번하면 `cat_confidence`, `dog_confidence`, `person_confidence` 같은 per-class 최대 confidence scalar를 별도로 두고 bbox는 JSON 문자열로 둔다. 좌표 정규화 규약과 이미지 크기도 기록한다.
- **MySQL `scenes`:** 현재 `object_labels TEXT`를 쓴다. 간단히는 `object_detections JSON` 컬럼을 추가할 수 있다. 향후 검색·집계가 빈번하면 `scene_object_detections(scene_id,label,confidence,x1,y1,x2,y2,model_version)` 자식 테이블이 질의하기 쉽다.
- **behavior event `source_frames`:** 기존 프레임별 `object_labels`를 유지하고 각 frame에 `object_detections` 배열을 포함한다. MySQL의 `source_frames_json JSON`은 배열/객체를 보존한다. event aggregate label에 confidence를 임의로 붙이지 말고, source frame까지 증거를 추적한다. 모델 이름, weight hash, threshold를 함께 보존한다.

## 9. 기존 617개 frame 갱신 절차 (실행하지 않음)

1. Weight SHA, class threshold, Ultralytics 버전, bbox 좌표 규약을 확정하고 Chroma/DB 및 frame ID 대응표를 백업한다.
2. 617개 원본 프레임을 재추론하고 frame ID별 labels/detections를 staging 결과로 만든다. 누락·중복 ID와 confidence 경계 표본을 검토한다.
3. 현재 `index_frame()`은 기존 ID를 조회하고 skip하므로 worker 재실행만으론 metadata가 갱신되지 않는다. 동일 ID에 `collection.update(ids, metadatas)`를 적용하면 embedding/ID를 유지하며 metadata를 바꿀 수 있다. vector count와 전체 ID 불변성을 검증한다.
4. MySQL `scenes`는 frame별 row지만 독립 frame_id 컬럼이 없다. timestamp 중복 가능성이 있어 s3_key 또는 별도 frame ID 대응으로 정확히 update/upsert해야 한다.
5. behavior `source_frames_json`의 frame-level label/detection만 갱신하고 event action/summary가 이에 의존하지 않으면 event 재생성은 불필요하다. event 문장 자체가 이전 label 기반으로 생성됐다면 event 추출을 재실행하고 결과를 비교해야 한다.
6. 완료 후 Chroma IDs/count, scenes row, event source frame labels를 대조하고 cat/dog/person 검색 및 추천 질문을 다시 검증한다. 이 절차는 이번에 실행하지 않았다.

## 평가 재현 정보

Ultralytics 8.4.33, Python 3.12.14, arm64 CPU. yolov8n SHA256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`; yolov8s SHA256 `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`. 전체 프레임별 정답, raw detections, confidence, bbox와 inference time은 동봉 JSON에 포함했다. yolov8s weight는 `/tmp/yolo_eval_weights/yolov8s.pt`에 임시 보관했고 프로젝트 폴더에는 두지 않았다.
