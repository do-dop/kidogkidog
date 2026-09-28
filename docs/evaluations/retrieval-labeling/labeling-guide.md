# Retrieval Ground Truth labeling guide

이 자료는 retrieval 결과와 독립적으로 실제 프레임을 보고 정답 구간을 표시하기 위한 pack입니다. contact sheet에는 원본 이미지와 중립적인 frame 번호·timestamp·chunk stem·짧은 frame ID만 표시합니다.

## 판단 기준

- 객체 질문(강아지/고양이/사람): 해당 객체가 실제 이미지에 보이는 구간만 relevant로 표시합니다.
- 물 마시기: 혀/입을 물에 대고 마시는 모습 또는 연속 프레임에서 음수 행동이 확인되는 구간만 relevant입니다. 물그릇 옆에 있다는 사실만으로는 relevant가 아닙니다.
- 사료 그릇 접근: 이전 위치보다 그릇 방향으로 이동하는 시간 흐름이 확인되어야 합니다. 이미 그릇 옆에 선 정지 frame만으로 접근이라고 판정하지 않습니다.
- 움직임/뛰기: 앞뒤 frame/video에서 위치 변화가 확인되어야 합니다. 서 있거나 걷는 모습은 ‘뛰는 장면’의 정답이 아닙니다.
- 소파 아래 진입: 실제로 소파 아래 영역으로 들어가는 과정 또는 소파 아래 위치가 명확해야 relevant입니다.
- 위치·대상(소파 근처, 급식기 주변): 반려동물이 해당 대상/위치에 실제로 있는 구간을 표시합니다.
- event-grounded query: event metadata를 정답으로 옮기지 말고 해당 영상 contact sheet의 이미지를 직접 확인합니다.
- 정지 frame만으로 행동을 확정하기 어렵거나 가려짐/해상도 문제로 확신할 수 없으면 `uncertain`으로 남깁니다.

## 시간 구간 입력

가능하면 프레임 하나보다 행동이 확인되는 구간을 `relevant_time_ranges`에 기록합니다. start/end는 chunk-local seconds이며 양 끝을 포함합니다. 각 range에는 반드시 `video_id`와 `chunk_stem`을 함께 입력합니다. 서로 다른 chunk의 같은 timestamp는 별도 구간입니다. tolerance는 적용하지 않습니다.

```json
{"chunk_stem":"petcam_xxx_000","video_id":"dog_drinking_water","start":10.0,"end":18.0,"relevance":"relevant"}
```

불확실한 범위는 `relevance: "uncertain"`으로 적고 notes에 이유를 기록합니다. 정확한 장면 한두 프레임만 확인되면 `relevant_frame_ids` 또는 `frame_judgments`를 사용할 수 있습니다.

## Scope 확인

현재 15개 query의 scope 값은 기존 `retrieval-ground-truth-2026-09-27.json`을 그대로 따릅니다. `video_id`가 없는 query는 전체 eligible corpus를 대상으로 하는 global query이고, event-grounded 3개는 지정된 단일 video scope입니다. 다만 과거 explicit-object-search artifact에는 고양이/사람 query를 각 video에 따로 고정 실행한 기록이 있습니다. 그 과거 per-video 결과를 이번 15개 global query로 바꾸어 해석하지 않도록 구분해 두었습니다. query를 추가하거나 scope를 임의 변경하지 않았습니다.

| query_id | labeling_mode | scope | query |
|---|---|---|---|
| object_cat_visible | frame_visual | global (전체 eligible corpus) | 고양이가 보이는 장면 보여줘 |
| object_dog_visible | frame_visual | global (전체 eligible corpus) | 강아지가 보이는 장면 보여줘 |
| object_person_visible | frame_visual | global (전체 eligible corpus) | 사람이 함께 보이는 장면 있어? |
| object_bowl_nearby | frame_visual | global (전체 eligible corpus) | 반려동물이 그릇 근처에 있는 장면 |
| behavior_drinking_water | video_temporal | global (전체 eligible corpus) | 물을 마신 장면이 있어? |
| behavior_approach_food_bowl | video_temporal | global (전체 eligible corpus) | 사료 그릇에 접근한 장면이 있어? |
| behavior_dog_running | video_temporal | global (전체 eligible corpus) | 개가 뛰는 장면 |
| behavior_moving | video_temporal | global (전체 eligible corpus) | 반려동물이 움직인 장면은 언제였어? |
| location_feeder_nearby | frame_visual | global (전체 eligible corpus) | 자동 급식기 근처에 있는 장면이 있어? |
| location_couch_movement | video_temporal | global (전체 eligible corpus) | 소파 근처에서 움직인 장면 찾아줘 |
| location_door_front | frame_visual | global (전체 eligible corpus) | 문 앞에 있는 장면 보여줘 |
| location_under_couch | video_temporal | global (전체 eligible corpus) | 소파 아래에 들어간 적이 있어? |
| event_feeder_grounded_img8450 | frame_visual | video: IMG_8450_2 | 자동 급식기 근처에 있는 장면이 있어? |
| event_food_bowl_grounded_img8450 | video_temporal | video: IMG_8450_2 | 사료 그릇에 접근한 장면이 있어? |
| event_water_grounded_dog_water | video_temporal | video: dog_drinking_water | 물을 마신 장면이 있어? |

`global` query는 네 영상 전체를 검수해야 완료할 수 있습니다. UI에서 각 video를 선택하고 ‘이 video 검수 완료’를 눌러 네 개 모두 기록하세요. event-grounded query는 지정된 video 하나만 확인하면 됩니다.

## Exact source chunk playback

- IMG_8450_2 / `petcam_20260926_173123_000`: `gcs_exact` — `/private/var/folders/g5/83j4lqks1w1gyzq149ljv2xc0000gn/T/retrieval-labeling-videos-2026-09-27/IMG_8450_2/petcam_20260926_173123_000.mp4`
- IMG_8450_2 / `petcam_20260926_173123_001`: `gcs_exact` — `/private/var/folders/g5/83j4lqks1w1gyzq149ljv2xc0000gn/T/retrieval-labeling-videos-2026-09-27/IMG_8450_2/petcam_20260926_173123_001.mp4`
- dog_drinking_water / `petcam_20260926_173602_000`: `gcs_exact` — `/private/var/folders/g5/83j4lqks1w1gyzq149ljv2xc0000gn/T/retrieval-labeling-videos-2026-09-27/dog_drinking_water/petcam_20260926_173602_000.mp4`
- dog_drinking_water / `petcam_20260926_173602_001`: `gcs_exact` — `/private/var/folders/g5/83j4lqks1w1gyzq149ljv2xc0000gn/T/retrieval-labeling-videos-2026-09-27/dog_drinking_water/petcam_20260926_173602_001.mp4`
- dog_drinking_water / `petcam_20260926_173602_002`: `gcs_exact` — `/private/var/folders/g5/83j4lqks1w1gyzq149ljv2xc0000gn/T/retrieval-labeling-videos-2026-09-27/dog_drinking_water/petcam_20260926_173602_002.mp4`
- dog_drinking_water / `petcam_20260926_173602_003`: `gcs_exact` — `/private/var/folders/g5/83j4lqks1w1gyzq149ljv2xc0000gn/T/retrieval-labeling-videos-2026-09-27/dog_drinking_water/petcam_20260926_173602_003.mp4`
- dog_escape / `petcam_20260926_173415_000`: `gcs_exact` — `/private/var/folders/g5/83j4lqks1w1gyzq149ljv2xc0000gn/T/retrieval-labeling-videos-2026-09-27/dog_escape/petcam_20260926_173415_000.mp4`
- two_dogs / `two_dogs_000`: `gcs_exact` — `/private/var/folders/g5/83j4lqks1w1gyzq149ljv2xc0000gn/T/retrieval-labeling-videos-2026-09-27/two_dogs/two_dogs_000.mp4`

정확한 (video_id, chunk_stem) 쌍이 확인된 chunk만 `/tmp/retrieval-labeling-videos-2026-09-27/`에 로컬 재생용으로 내려받습니다. browser가 재생하지 못하면 위 경로를 직접 video player로 열고 chunk-local timestamp를 참고하세요.

## Contact sheets

- [IMG_8450_2-contact-sheet-01.jpg](IMG_8450_2-contact-sheet-01.jpg) — IMG_8450_2, page 1, 16 frames
- [IMG_8450_2-contact-sheet-02.jpg](IMG_8450_2-contact-sheet-02.jpg) — IMG_8450_2, page 2, 16 frames
- [IMG_8450_2-contact-sheet-03.jpg](IMG_8450_2-contact-sheet-03.jpg) — IMG_8450_2, page 3, 1 frames
- [dog_drinking_water-contact-sheet-01.jpg](dog_drinking_water-contact-sheet-01.jpg) — dog_drinking_water, page 1, 16 frames
- [dog_drinking_water-contact-sheet-02.jpg](dog_drinking_water-contact-sheet-02.jpg) — dog_drinking_water, page 2, 16 frames
- [dog_drinking_water-contact-sheet-03.jpg](dog_drinking_water-contact-sheet-03.jpg) — dog_drinking_water, page 3, 16 frames
- [dog_drinking_water-contact-sheet-04.jpg](dog_drinking_water-contact-sheet-04.jpg) — dog_drinking_water, page 4, 16 frames
- [dog_drinking_water-contact-sheet-05.jpg](dog_drinking_water-contact-sheet-05.jpg) — dog_drinking_water, page 5, 16 frames
- [dog_drinking_water-contact-sheet-06.jpg](dog_drinking_water-contact-sheet-06.jpg) — dog_drinking_water, page 6, 9 frames
- [dog_escape-contact-sheet-01.jpg](dog_escape-contact-sheet-01.jpg) — dog_escape, page 1, 16 frames
- [dog_escape-contact-sheet-02.jpg](dog_escape-contact-sheet-02.jpg) — dog_escape, page 2, 12 frames
- [two_dogs-contact-sheet-01.jpg](two_dogs-contact-sheet-01.jpg) — two_dogs, page 1, 16 frames
- [two_dogs-contact-sheet-02.jpg](two_dogs-contact-sheet-02.jpg) — two_dogs, page 2, 16 frames
- [two_dogs-contact-sheet-03.jpg](two_dogs-contact-sheet-03.jpg) — two_dogs, page 3, 16 frames
- [two_dogs-contact-sheet-04.jpg](two_dogs-contact-sheet-04.jpg) — two_dogs, page 4, 16 frames
- [two_dogs-contact-sheet-05.jpg](two_dogs-contact-sheet-05.jpg) — two_dogs, page 5, 16 frames
- [two_dogs-contact-sheet-06.jpg](two_dogs-contact-sheet-06.jpg) — two_dogs, page 6, 10 frames

모든 query의 초기 상태는 `pending_human_review`입니다. 전체 scope를 확인하고, relevant range/frame을 입력하거나 ‘검수했지만 relevant 결과 없음’을 명시해야 `labeled`로 export됩니다. 후자는 `no_relevant_results: true`로 표현합니다. 빈 배열만으로는 검수 완료가 아닙니다.
