# 저장 YOLO object_labels 정확도 점검 (2026-09-27)

코드·MySQL·ChromaDB는 변경하지 않았다. 617개 전체 metadata를 읽었고, 네 영상에서 라벨별 timestamp 분포를 확인했다. 영상 구간에 걸쳐 label별 최대 3개를 골랐으며, 겹치는 표본은 중복 제거했다. 표본 36개 중 32개 이미지는 GCS에서 읽을 수 있었다. `two_dogs`의 표본 4개는 Chroma에 frame metadata는 있으나 해당 GCS object가 없어 시각 확인과 재추론이 불가능했다. 직접 확인 샘플은 confidence 범주별로 의도적으로 선택했으므로 전체 모집단 precision/recall 표본은 아니다.

## 현재 YOLO→저장 흐름

1. `pipeline/yolo_detector.py:get_yolo_model()`은 `YOLO("yolov8n.pt")`를 사용한다. 현재 repository에 weight가 있고 크기는 약 6.2 MB다. 이 실행 환경의 `ultralytics`는 8.4.33, `torch`는 2.11.0이다. 로컬 weight SHA-256은 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
2. `detect_objects()` 기본 confidence threshold는 0.5다. `TARGET_LABELS`는 `person,dog,cat,bird,horse,sheep,cow,bear,zebra,giraffe,bowl,cup,bottle,chair,couch,bed,dining table,sports ball,tv,remote,backpack,handbag,suitcase`.
3. YOLO `result.boxes`를 순회하며 class ID→이름, confidence, bbox를 읽는다. threshold 미만과 target 외 class를 제거하고 class 이름을 set에 모은 뒤 sorted list를 반환한다. 한 프레임에서 고양이 2마리 및 그릇이 보여도 class별 label 하나만 남는다.
4. confidence/bbox는 detect 함수 내부에서만 읽고, 반환 직전에 버린다. `tasks.process_chunk()`는 label list를 `index_frame()`에 전달하고 behavior event extractor에도 전달한다.
5. `vector_store.index_frame()`는 labels를 Chroma의 문자열 `cat,dog`로 저장한다. 저장 metadata에는 frame path, timestamp, video_id, object_labels와 선택적인 storage/date key뿐이다. bbox/confidence는 Chroma metadata에 저장되지 않는다.
6. `scenes.object_labels`는 JSON label 이름만 저장한다. behavior event의 `source_frames`에도 frame ID/path/key/timestamp와 `object_labels: list[str]`만 남는다. label confidence/bbox는 없으며 event `confidence`는 이벤트 분석 confidence로 detection box confidence와 별도다.

따라서 저장 과정에서 **YOLO label 이름은 남고 detection score 및 box 좌표는 사라진다.** 원 데이터로 threshold 사후 변경할 수 없고, confidence를 쓰려면 이미지에 모델을 다시 돌려야 한다.

## Indexed 데이터 분포

영상은 총 617 frame이며 빈 label 61개다. 아래 cat/dog/person 및 기타 개수는 다중 label frame에서 각 label을 각각 세었다. `cat+dog`는 같은 frame의 동시 label이다.

| 영상 | frame | 빈 label | cat | dog | person | cat+dog | 기타 주요 label |
|---|---:|---:|---:|---:|---:|---:|---|
| IMG_8450_2 | 33 | 1 | 29 | 6 | 0 | 3 | — |
| dog_drinking_water | 89 | 21 | 12 | 46 | 2 | 1 | chair 11, bowl 5 |
| dog_escape | 28 | 11 | 0 | 0 | 5 | 0 | bed 12, chair 1 |
| two_dogs | 175 | 22 | 4 | 91 | 16 | 2 | tv 137, chair 75, couch 52, horse 6, cow 5 |
| cats_5min | 292 | 6 | 286 | 4 | 0 | 4 | bowl 12 |
| **전체** | **617** | **61** | **331** | **147** | **23** | **10** | **tv 137, chair 87, couch 52, bed 12, bowl 17, horse 6, cow 5** |

명백한 내용 모순: IMG_8450_2는 고양이 영상인데 6 frame에 dog label, 그중 3 frame은 cat,dog 동시. 직접 본 dog-only 8.5초 프레임은 고양이이며 current 재추론 dog 0.819. `two_dogs`는 개들 영상인데 cat label 4개. 실제 확인 가능했던 cat label frame 90,106초는 모두 강아지. 반대로 IMG의 실제 고양이 8.5초 dog label 및 빈 label 6.5초 일부 고양이는 종 false positive와 false negative를 모두 보여준다.

`dog_escape`에는 cat/dog label이 아예 0개이며 dog가 등장한다. person 5개는 실제 현관 영상 속 사람 2명과 Ring 예시 inset 화면의 사람 이미지 1개가 섞여 있다. #38은 개가 렌즈 가까이를 덮은 프레임인데 `person` label이 있다.

`dog_drinking_water`의 실제 영상 주체는 강아지다. 저장된 cat 12개 중에서 stratified 표본 3개(6,26,56초)를 열어보니 모두 강아지였고, person 2개(24,28초)도 실제 장면에서는 강아지였다. 단, 표본 3개로 cat label 12개 전부가 오검출이라고 일반화하지 않는다.

## 접근 불가 프레임

아래 네 Chroma metadata에는 label이 있지만 GCS object가 없어 원 이미지 판정 및 YOLO 재실행을 할 수 없었다.

| video_id | chunk/frame | timestamp | 저장 labels | 표본 그룹 |
|---|---|---:|---|---|
| two_dogs | two_dogs__petcam_20260528_164504_001_frame_24.50 | 24.5 | cat,tv | cat |
| two_dogs | two_dogs__petcam_20260528_164504_002_frame_16.50 | 16.5 | chair,couch,person,tv | person |
| two_dogs | two_dogs__petcam_20260528_164504_002_frame_44.50 | 44.5 | chair,dog,person,tv | person |
| two_dogs | two_dogs__petcam_20260528_164504_002_frame_52.50 | 52.5 | (빈 값) | empty |

## 실제 이미지 표본과 현재 모델 재실행

접근 가능한 이미지 32개에 현재 로컬 `yolov8n.pt`를 다시 실행했다. raw prediction은 `conf=0.001`로 받아 class별 confidence와 xyxy bbox를 기록하고, 같은 결과에서 0.5/0.6/0.7/0.8를 적용한 label도 계산했다. 이는 네 threshold로 각각 별도 `predict()`한 것이 아니라 동일 raw pass의 score cutoff 비교다. `detect_objects()`와 같은 weight/Ultralytics 버전을 사용했고 기본 inference 크기/NMS 설정도 그대로 둔 실행이지만, 인덱싱 때 사용된 당시 런타임 버전까지 증명하지는 않는다.

아래 실제 객체 판정은 contact sheet를 사람이 확인한 결과다. bbox는 raw rerun의 최고 confidence 해당 class 좌표이며 원래 저장 당시 box는 아니다. frame ID에 청크 stem을 포함했다. 표본 contact sheet: [IMG_8450_2](yolo-label-audit/yolo_IMG_8450_2_0.jpg), [dog_drinking_water](yolo-label-audit/yolo_dog_drinking_water_0.jpg), [dog_escape](yolo-label-audit/yolo_dog_escape_0.jpg), [two_dogs](yolo-label-audit/yolo_two_dogs_0.jpg).

| video_id | chunk/frame | t(s) | 저장 labels | 화면에서 보이는 객체 | 라벨 판정 | 현재 YOLO 최고 class confidence [bbox x1,y1,x2,y2] |
|---|---|---:|---|---|---|---|
| IMG_8450_2 | petcam_20260926_173123_000 | 2.0 | cat | cat | 맞음:cat | cat 0.782 [571,546,1006,766]; dog 0.008 [1596,359,1899,540]; person 0.003 [0,69,118,862] |
| IMG_8450_2 | petcam_20260926_173123_000 | 38.0 | cat | cat | 맞음:cat | cat 0.884 [864,599,1111,953]; dog 0.023 [719,947,1037,1077] |
| IMG_8450_2 | petcam_20260926_173123_000 | 66.0 | cat | cat | 맞음:cat | cat 0.804 [727,291,1007,662]; dog 0.002 [834,656,1165,998] |
| IMG_8450_2 | petcam_20260926_173123_000 | 8.5 | dog | cat | 오검출:dog; 미검출:cat | cat 0.023 [0,385,114,863]; dog 0.819 [548,494,1010,929]; person 0.001 [1683,336,1920,512] |
| IMG_8450_2 | petcam_20260926_173123_000 | 16.0 | cat,dog | cat | 맞음:cat; 오검출:dog | cat 0.587 [634,487,1044,709]; dog 0.577 [548,650,818,1074]; person 0.008 [1500,273,1919,518] |
| IMG_8450_2 | petcam_20260926_173123_000 | 46.0 | cat,dog | cat | 맞음:cat; 오검출:dog | cat 0.632 [826,382,1236,589]; dog 0.644 [716,750,1137,1076] |
| IMG_8450_2 | petcam_20260926_173123_000 | 6.5 | (빈 값) | cat | 미검출:cat | cat 0.219 [603,550,1022,988]; dog 0.385 [599,490,1022,990]; person 0.011 [0,475,64,786] |
| dog_drinking_water | petcam_20260926_173602_003 | 6.0 | cat,dog | dog | 맞음:dog; 오검출:cat | cat 0.507 [0,3,808,590]; dog 0.545 [1,3,809,588]; person 0.009 [3,274,56,532] |
| dog_drinking_water | petcam_20260926_173602_001 | 26.0 | cat | dog | 오검출:cat; 미검출:dog | cat 0.511 [136,1,787,634]; dog 0.309 [143,2,786,639]; person 0.324 [0,107,110,707] |
| dog_drinking_water | petcam_20260926_173602_001 | 56.0 | cat | dog | 오검출:cat; 미검출:dog | cat 0.694 [532,138,828,441]; dog 0.259 [111,2,692,528]; person 0.062 [2,96,1133,712] |
| dog_drinking_water | petcam_20260926_173602_001 | 2.0 | dog | dog | 맞음:dog | cat 0.002 [114,202,510,530]; dog 0.822 [21,4,870,540]; person 0.012 [29,500,1075,715] |
| dog_drinking_water | petcam_20260926_173602_002 | 28.0 | dog | dog | 맞음:dog | cat 0.003 [637,264,797,424]; dog 0.582 [197,3,810,530]; person 0.019 [0,3,237,514] |
| dog_drinking_water | petcam_20260926_173602_000 | 62.0 | dog | dog | 맞음:dog | cat 0.002 [2,7,506,623]; dog 0.532 [0,4,803,611]; person 0.016 [15,565,998,720] |
| dog_drinking_water | petcam_20260926_173602_001 | 24.0 | dog,person | dog | 맞음:dog; 오검출:person | dog 0.567 [295,14,864,625]; person 0.530 [0,56,176,594] |
| dog_drinking_water | petcam_20260926_173602_001 | 28.0 | person | dog | 오검출:person; 미검출:dog | cat 0.019 [133,4,894,646]; dog 0.162 [133,3,850,630]; person 0.700 [0,130,113,597] |
| dog_drinking_water | petcam_20260926_173602_000 | 3.5 | (빈 값) | dog | 미검출:dog | cat 0.214 [26,3,838,635]; dog 0.173 [203,3,828,625]; person 0.184 [12,4,844,635] |
| dog_drinking_water | petcam_20260926_173602_001 | 36.0 | (빈 값) | dog | 미검출:dog | cat 0.018 [248,242,594,508]; dog 0.420 [141,1,608,522]; person 0.024 [1,2,162,119] |
| dog_drinking_water | petcam_20260926_173602_000 | 64.0 | (빈 값) | dog | 미검출:dog | cat 0.238 [0,2,517,646]; dog 0.023 [213,178,539,624]; person 0.418 [1,2,512,673] |
| dog_escape | petcam_20260926_173415_000 | 38.0 | person | dog | 오검출:person; 미검출:dog; 주의:가까이 카메라를 덮은 강아지; 사람 아님 | dog 0.068 [636,409,1192,1037]; person 0.687 [1,3,787,1066] |
| dog_escape | petcam_20260926_173415_000 | 42.0 | person | person | 맞음:person | person 0.876 [687,562,958,1074] |
| dog_escape | petcam_20260926_173415_000 | 56.0 | person | person | 맞음:person; 주의:person은 Ring 예시 영상의 inset 화면 안에 있음; 실제 현관 장면의 사람인지는 다름 | person 0.881 [257,532,762,903] |
| dog_escape | petcam_20260926_173415_000 | 6.0 | (빈 값) | dog | 미검출:dog | cat 0.152 [0,440,473,966]; dog 0.002 [841,291,1157,1078]; person 0.012 [0,364,401,1068] |
| dog_escape | petcam_20260926_173415_000 | 28.0 | (빈 값) | dog | 미검출:dog | cat 0.043 [463,202,709,535]; dog 0.024 [0,582,409,1062]; person 0.027 [1787,706,1920,1078] |
| dog_escape | petcam_20260926_173415_000 | 52.0 | (빈 값) | 없음/그래픽 | 종/사람 label 일치 | person 0.021 [950,751,1021,857] |
| two_dogs | two_dogs_000 | 90.0 | cat,tv | dog | 오검출:cat; 미검출:dog | cat 0.765 [0,527,433,1046]; person 0.006 [554,391,574,420] |
| two_dogs | two_dogs_000 | 106.0 | cat,dog,tv | dog | 맞음:dog; 오검출:cat | cat 0.531 [39,573,1098,1074]; dog 0.781 [4,571,1093,1073]; person 0.046 [271,517,316,584] |
| two_dogs | two_dogs_000 | 2.0 | chair,dog,tv | dog | 맞음:dog | dog 0.587 [680,307,1458,1072]; person 0.071 [1087,435,1223,576] |
| two_dogs | two_dogs_000 | 44.0 | chair,couch,dog,tv | dog | 맞음:dog | dog 0.839 [797,365,1167,705]; person 0.187 [0,430,48,608] |
| two_dogs | two_dogs_000 | 170.0 | chair,dog,tv | dog | 맞음:dog | dog 0.788 [1476,491,1789,952]; person 0.162 [1743,333,1920,1069] |
| two_dogs | two_dogs_000 | 168.0 | chair,dog,person,tv | dog,person | 맞음:dog,person | cat 0.003 [497,648,588,748]; dog 0.714 [338,473,604,978]; person 0.913 [452,197,910,1012] |
| two_dogs | two_dogs_000 | 4.0 | (빈 값) | dog | 미검출:dog | person 0.192 [864,574,1348,1072] |
| two_dogs | two_dogs_000 | 178.0 | (빈 값) | 없음/그래픽 | 종/사람 label 일치; 주의:라이브 동물 대신 종료 그래픽 속 개 그림 | dog 0.116 [375,471,570,1010]; person 0.126 [563,416,862,922] |

실제 확인 표본에서 눈에 띄는 혼동은 다음과 같다.

- IMG_8450_2 #frame_8.50: 고양이에 저장 label `dog`; 현재 dog 0.819로 threshold 0.8까지 통과한다.
- IMG_8450_2 #frame_16.00: 고양이 장면에 `cat,dog`; 현재 cat 0.587, dog 0.577.
- IMG_8450_2 #frame_46.00: 고양이 장면에 `cat,dog`; 현재 cat 0.632, dog 0.644.
- dog_drinking_water #frame_26.00, #56.00: 강아지인데 저장 `cat`; 재실행 cat은 각각 0.511과 0.694.
- dog_drinking_water #frame_28.00: 강아지인데 저장 `person`; 재실행 person 약 0.700.
- two_dogs #frame_000046 (t=90s): 강아지인데 저장 `cat,tv`; 재실행 cat 0.765로 강하게 통과.
- dog_escape #frame_38.00: 개가 렌즈를 가린 장면에서 저장/current `person`, current person 0.687. 반면 실제 사람이 보이는 #42초는 person 0.876.
- 빈 label인 dog_drinking_water #36초에는 분명 강아지가 보이지만 dog 최고 score는 0.420으로 저장 threshold 0.5보다 낮다. #64초도 강아지가 보이지만 dog score는 낮다. 빈 값은 단순히 confidence 저장 누락이 아니라 threshold 아래 검출/검출 누락일 수 있다.

## Threshold 비교: 표본에서 본 사례

다음은 해당 클래스가 threshold 이상인지 여부다. 실제 모델 recall/precision 점수로 일반화하지 않는다.

| 장면/대상 | 현재 class score | 0.5 | 0.6 | 0.7 | 0.8 | 의미 |
|---|---:|---|---|---|---|---|
| 실제 cat, IMG 8.5초 / 잘못된 dog | dog 0.819 | 남음 | 남음 | 남음 | 남음 | 임계값만 올려도 오검출 제거 안 됨 |
| 실제 cat, IMG 46초 / cat+dog | cat 0.632, dog 0.644 | 둘 다 | 둘 다 | 둘 다 제거 | 둘 다 제거 | 0.7은 오류와 올바른 cat을 같이 잃음 |
| 실제 dog, water 26초 / 잘못된 cat | cat 0.511, dog 0.309 | 잘못된 cat 남음 | 제거, dog 없음 | 없음 | 없음 | 오검출 감소가 맞는 dog 복구는 아님 |
| 실제 dog, water 56초 / 잘못된 cat | cat 0.694 | 잘못된 cat 남음 | 남음 | 제거 | 없음 | 0.6으로 충분치 않음 |
| 실제 dog, water 28초 / 잘못된 person | person 약 0.700 | 남음 | 남음 | cutoff 경계에서 제거 | 제거 | 점수를 올리면 이 false positive는 감소 |
| 실제 dog, two_dogs 90초 / 잘못된 cat | cat 0.765 | 남음 | 남음 | 남음 | 제거 | 0.8에서야 제거 |
| 실제 cat, IMG 38초 | cat 0.884 | 남음 | 남음 | 남음 | 남음 | 높은 score의 올바른 탐지 유지 사례 |
| 실제 dog, water 2초 | dog 0.822 | 남음 | 남음 | 남음 | 남음 | 높은 score의 올바른 탐지 유지 사례 |
| 실제 dog, water 36초 / 저장 label 없음 | dog 0.420 | 없음 | 없음 | 없음 | 없음 | 0.5 이상 threshold 비교만으로 누락 해결 안 됨 |
| 실제 person, dog_escape 42초 | person 0.876 | 남음 | 남음 | 남음 | 남음 | 높은 score의 올바른 사람 탐지 유지 사례 |

표본 raw pass에서 class가 하나 이상 threshold를 넘은 frame 수(cat/dog/person)는 0.5에서 10/13/6, 0.6에서 6/7/5, 0.7에서 4/6/3, 0.8에서 2/3/3으로 줄었다. 이 수는 정답 수가 아니라 **탐지 발생 수**다. 표본은 저장 label strata에 맞춰 선택했고, 네 영상은 객체 구성도 서로 다르므로 표본 precision/recall로 해석할 수 없다. 개별 사례만 봐도 0.6~0.7은 일부 오검출을 없애지만 IMG 46초의 참 cat과 일부 참 dog도 함께 놓친다. 0.8에도 dog→cat 90초, dog→dog 8.5초 오분류가 남는다. 이는 threshold만의 문제가 아니다.

## 모델 크기 비교 가능성

프로젝트는 `yolov8n.pt`만 참조하고 저장소 weight도 이 파일뿐이다. 이 환경에서 찾은 로컬/cache weight 중 yolov8s/m/l/x는 없었다. 모델 비교를 위해 새 weight 다운로드는 하지 않았다. 큰 모델이 작은 객체/occlusion을 더 잘 구분할 수도 있지만, 고양이/강아지 고 confidence 혼동을 반드시 해결한다는 근거는 없다. 동일 표본의 paired inference와 수작업 정답 평가 없이 모델 교체 결론을 내릴 수 없다.

## 검색과 추천 질문으로 전파되는 경로

- 검색: 명시 cat/dog/person query → 저장 `object_labels` exact filter → 조건을 만족한 프레임에 CLIP ranking. 필터는 문자열 label 정확성만 보장한다. 실제 강아지에 `cat`이 있으면 고양이 query를 통과한다. dog_drinking_water #26/#56, two_dogs #90이 그 예다. IMG #8.5의 실제 cat에 `dog`가 있어 dog query도 통과한다. confidence는 저장되지 않아 검색이 점수로 제거할 수 없다.
- 추천: `query_suggester._labels_from_indexed_frames()`는 `get_indexed_frames()` labels를 세고 `_sort_labels()`에서 cat을 dog보다 높은 우선순위로 둔다. `_labels_from_behavior_events()`도 event source_frames의 labels를 세며 target labels는 가중한다. `_generate_video_specific_questions()`는 이 결과로 cat/dog/person 질문을 만들고, indexed frames fallback도 labels를 사용한다.
- 현재 실제 frame metadata로 추천 label 경로를 재실행해 `dog_drinking_water`와 `two_dogs` 모두 `cat`이 첫 label로 나왔고, 두 영상 모두 “고양이가 보이는 장면 보여줘 / 고양이가 움직인 장면 찾아줘”가 앞에 생성됐다. dog_drinking_water의 stratified cat 라벨 표본은 실제 강아지였다. 두 영상의 cat 추천은 저장된 false-positive label이 실제 추천에 영향을 줄 수 있음을 보여준다. 실제 최종 추천 우선순위에는 behavior LLM 질문도 섞이므로, label 질문이 늘 항상 최종 3개에 들어간다고 단정하진 않는다.
- behavior-event 기반 질문은 event summary/action을 사용하고 label 질문은 frame/event labels를 사용한다. 잘못된 라벨이 모든 자연어 행동 질문을 바꾸는 것은 아니지만, label fallback 후보와 source frame evidence에 영향을 준다.

## 원인과 개선 순서 제안

1. **YOLO 자체 오분류:** 현재 yolov8n 재추론에서도 높은 confidence cat↔dog 및 dog→person 혼동이 확인됐다. 강아지 #8.5 dog 0.819, two_dogs #90 cat 0.765는 threshold를 조금 올려도 남는다.
2. **Threshold:** 0.5 아래 score로 label 빈 프레임이 있고, 0.6~0.8은 일부 오검출을 제거하지만 정답 탐지도 함께 줄인다. frame별 score를 저장하지 않아 기존 오검출이 threshold 경계였는지도 사후 구별할 수 없다.
3. **저장 정보 손실:** 파이프라인이 confidence와 bbox를 읽지만 frame index 직전 label set만 반환해 이 정보를 버린다. Chroma와 event source frame 모두 label names only다.
4. **검색 hard filter:** 이번 명시 객체 필터는 잘못 저장한 label이 있으면 틀린 실제 객체를 hard-pass시킨다. 필터 코드보다 upstream label precision에 민감하다.
5. **label 추천:** 저장 label 우선순위로 cat 후보가 실제 강아지 영상에서도 앞에 나왔다. 검색과 추천 모두 오염되지만 서로 다른 경로로 노출된다.

직접적인 순서 제안: (1) 표본에 사람 정답을 붙인 고정 평가셋을 늘려 종별 confusion matrix를 만들고, GCS 없는 frame을 복구할 수 있는지 확인, (2) 원본 detector class/confidence/bbox를 보존하도록 재처리/backfill 경로를 설계, (3) 같은 표본에서 0.5~0.8과 yolov8s paired 비교 후 per-class 기준 선택, (4) 선택한 모델/기준으로 기존 frame 617개를 다시 분석해 Chroma object_labels와 MySQL scenes 및 behavior source_frames를 일관되게 backfill, (5) 검증된 label을 검색과 질문 생성에 적용한다. metadata만 바로 필터링하면 이미 오염된 617개의 오류는 남는다.

라벨 수정은 반드시 모든 소비 저장소에 반영해야 한다. Chroma frame metadata만 재분석하면 label fallback은 바뀔 수 있지만 MySQL behavior `source_frames`는 이전 label로 남는다. 동일 CLIP image embeddings는 object-label 변경만으로 재계산할 필요는 없어 보이며, Chroma metadata update와 DB label update/backfill로 가능할 수 있다. 현재 저장 스키마는 per-box 값을 보관하지 않으므로 재분석은 기존 617 이미지에 YOLO를 다시 실행해야 한다. 이 단계에서는 어떤 backfill/reindex도 수행하지 않았다.

## 산출물과 한계

- 표본 raw detection JSON에는 접근 가능 이미지 32개에 대해 전체 cat/dog/person class, confidence, bbox, 0.5/0.6/0.7/0.8 cutoff label을 기록했다.
- contact sheets는 위 표에 연결했다. 접근 불가 GCS object 4개는 판정하지 않았다.
- 표본 결과는 선택 편향이 있으므로 전체 데이터 정확도 추정치가 아니다. 샘플의 눈에 띄는 고 confidence 혼동은 확인했지만 617개 전체 수작업 검증은 하지 않았다.
- 모델 threshold/크기/Chroma/DB/index 설정은 변경하지 않았다.
