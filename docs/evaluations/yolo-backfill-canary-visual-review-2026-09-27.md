# YOLO backfill canary visual review

- 대상: eligible dry-run에서 선정된 15개 canary의 정확한 GCS frame
- 전체 이미지 contact sheet: `yolo-backfill-canary-contact-sheet-2026-09-27.jpg`
- 판정: 원본 해상도 이미지 직접 확인, bbox overlay 없음
- Chroma/MySQL/behavior_events/GCS write: 없음

## 판정 요약

| 결과 | 개수 |
|---|---:|
| new가 더 정확 | 9 |
| old가 더 정확 | 3 |
| 둘 다 일부 맞음 또는 주요 label 맞음 | 2 |
| 둘 다 틀림 | 1 |
| 판정 보류 | 0 |

## Cat / dog / person 비교

‘정답’은 해당 class가 눈에 보이는 positive frame에서 맞게 검출된 수(TP)이며, 이 15개 canary에만 해당합니다.

| class | GT positive | old correct | new correct | old FP | new FP | old FN | new FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat | 4 | 2 | 3 | 1 | 2 | 2 | 1 |
| dog | 10 | 4 | 4 | 2 | 1 | 6 | 6 |
| person | 1 | 0 | 1 | 1 | 0 | 1 | 0 |

## Frame별 검수

| # | video / 시각 | old → new | 육안 정답 | 판정 | 메모 |
|---:|---|---|---|---|---|
| 1 | IMG_8450_2 / 66.0s | ['cat'] → ['bowl', 'dog'] | cat, bowl | old가 더 정확 | 세 마리 고양이가 보인다. bowl은 보이지만 dog는 없다. old cat은 맞고 new dog는 오분류이며 new는 cat을 빠뜨렸다. |
| 2 | dog_drinking_water / 12.5s | ['dog'] → ['bowl', 'cat'] | dog, bowl | old가 더 정확 | 검정과 흰색 강아지가 물그릇에 머리를 대고 있다. old dog는 맞고 new cat은 명백한 오분류다. 그릇은 보인다. |
| 3 | dog_drinking_water / 24.0s | ['dog', 'person'] → ['bowl', 'dog'] | dog, bowl | new가 더 정확 | 강아지와 초록색 그릇이 보인다. 화면 안에 사람은 보이지 않는다. new는 dog와 bowl을 표시하고 old의 person은 오검출이다. |
| 4 | dog_escape / 48.0s | [] → ['person'] | person | new가 더 정확 | 전체 프레임은 영상 안내 화면이지만 삽입된 감시 영상 안에 사람이 분명히 보인다. new person 검출은 맞고 old empty는 누락이다. |
| 5 | two_dogs / 8.0s | [] → ['dog'] | dog, couch | new가 더 정확 | 소파 위에 두 마리 강아지가 분명히 보인다. new dog가 맞고 old empty는 누락이다. couch도 보이지만 새 label에는 포함되지 않았다. |
| 6 | dog_drinking_water / 7.5s | ['cat'] → ['bowl'] | dog, bowl | new가 더 정확 | 강아지가 초록색 물그릇에 입을 대고 있다. new의 bowl은 맞지만 dog label은 없다. old cat은 잘못된 종 label이다. |
| 7 | IMG_8450_2 / 2.0s | ['cat'] → ['cat'] | cat, bowl | 둘 다 일부 맞음 또는 주요 label 맞음 | 고양이와 사료 그릇이 보인다. old와 new 모두 cat은 맞지만 bowl은 실제로 보여도 confidence 0.493으로 threshold 0.5 미만이라 new labels에서 빠졌다. |
| 8 | dog_escape / 10.0s | ['bed'] → ['bed'] | dog, couch, bowl | 둘 다 틀림 | 흰 강아지와 소파, 하단의 그릇이 보이며 bed는 보이지 않는다. old와 new 모두 dog/couch를 놓치고 bed를 잘못 표시한다. |
| 9 | two_dogs / 10.0s | ['couch', 'dog'] → ['couch', 'dog'] | dog, couch | 둘 다 일부 맞음 또는 주요 label 맞음 | 강아지 두 마리와 소파가 보인다. old와 new의 dog/couch labels 모두 실제 장면과 맞는다. |
| 10 | IMG_8450_2 / 12.0s | ['dog'] → ['bowl', 'cat'] | cat, bowl | new가 더 정확 | 두 마리 고양이와 그릇이 보인다. new cat+bowl은 맞고 old dog는 잘못된 종 label이다. |
| 11 | dog_escape / 12.0s | ['bed'] → ['couch'] | dog, couch, bowl | new가 더 정확 | 작은 흰 강아지가 소파 앞 펜스 안에 보이고 그릇도 보인다. new couch는 맞으며 old bed는 잘못된 물체 분류다. 양쪽 모두 dog를 놓쳤다. |
| 12 | two_dogs / 0.0s | ['chair', 'tv'] → ['chair', 'couch', 'dog', 'tv'] | dog, couch, chair | new가 더 정확 | 화면에 강아지 두 마리, 왼쪽 소파, 식탁 의자들이 보인다. new가 dog와 couch를 추가하고 old의 chair/tv 일부는 맞지만 강아지를 놓쳤다. |
| 13 | IMG_8450_2 / 14.0s | ['dog'] → ['bowl', 'cat'] | cat, bowl | new가 더 정확 | 두 마리 고양이와 그릇이 보인다. new cat+bowl은 맞고 old dog는 잘못된 종 label이다. |
| 14 | dog_drinking_water / 14.5s | ['dog'] → ['bowl', 'cat'] | dog, bowl | old가 더 정확 | 강아지가 초록색 그릇에서 물을 마시는 장면이다. old dog는 맞고 new cat은 명백한 종 오분류다. bowl은 new에서 맞게 검출됐다. |
| 15 | dog_escape / 14.0s | ['bed'] → ['couch'] | dog, couch, bowl | new가 더 정확 | 흰 강아지와 소파가 보인다. new couch는 맞고 old bed는 잘못된 분류다. 두 결과 모두 dog를 놓쳤다. |

## 0.4~0.5 confidence 검출

| # | detection | 육안 확인 |
|---:|---|---|
| 1 | cat 0.470, chair 0.423 | 세 마리 고양이가 보인다. bowl은 보이지만 dog는 없다. old cat은 맞고 new dog는 오분류이며 new는 cat을 빠뜨렸다. |
| 5 | couch 0.494 | 소파 위에 두 마리 강아지가 분명히 보인다. new dog가 맞고 old empty는 누락이다. couch도 보이지만 새 label에는 포함되지 않았다. |
| 6 | cat 0.417 | 강아지가 초록색 물그릇에 입을 대고 있다. new의 bowl은 맞지만 dog label은 없다. old cat은 잘못된 종 label이다. |
| 7 | bowl 0.493 | 고양이와 사료 그릇이 보인다. old와 new 모두 cat은 맞지만 bowl은 실제로 보여도 confidence 0.493으로 threshold 0.5 미만이라 new labels에서 빠졌다. |
| 8 | couch 0.477 | 흰 강아지와 소파, 하단의 그릇이 보이며 bed는 보이지 않는다. old와 new 모두 dog/couch를 놓치고 bed를 잘못 표시한다. |
| 11 | dog 0.426 | 작은 흰 강아지가 소파 앞 펜스 안에 보이고 그릇도 보인다. new couch는 맞으며 old bed는 잘못된 물체 분류다. 양쪽 모두 dog를 놓쳤다. |
| 15 | bed 0.415 | 흰 강아지와 소파가 보인다. new couch는 맞고 old bed는 잘못된 분류다. 두 결과 모두 dog를 놓쳤다. |

## 핵심 확인 사례

- IMG_8450_2 66.0s: 실제로 고양이들이 보인다. old `cat`이 맞고 new `dog`는 오검출이다. 낮은 `cat 0.470`은 실제 cat이지만 threshold 때문에 labels에서 빠졌다.
- dog_drinking_water 12.5s: 실제 dog와 bowl이 보인다. old `dog`가 맞고 new `cat`은 오분류다.
- dog_drinking_water 24.0s: 실제 dog와 bowl이 있고 사람은 없다. new는 dog/bowl을 맞췄고 old `person`은 false positive다.
- dog_escape 48.0s: 삽입된 감시 영상에 사람이 실제 보인다. new `person`이 맞다.
- two_dogs 8.0s: 강아지가 명확히 보인다. old empty보다 new `dog`가 맞다.

낮은 confidence 중 실제 대상과 맞는 사례는 #1 cat 0.470, #5 couch 0.494, #7 bowl 0.493, #8 couch 0.477, #11 dog 0.426이다. 이 중 #1의 cat, #5 couch, #7 bowl, #8 couch는 0.5 미만이라 object_labels에서 제외됐지만 실제로 보인다. #6 cat 0.417은 실제 dog 장면에서 cat으로 오분류된 것으로 보여 제외가 적절하다. #1 chair 0.423은 해당 영역이 cat furniture/scratcher로 보여 chair 여부는 보수적으로 uncertain 처리했다. #15 bed 0.415는 보이는 가구가 couch/dog bed 성격이라 bed label은 신뢰하기 어렵다.

## 판단

15개 표본에서는 new 우세 9, old 우세 3, 둘 다 일부/대체로 맞음 2, 둘 다 틀림 1, uncertain 0이다. 그러나 new가 실제 dog를 cat으로 분류한 #2와 #14, 실제 cat들을 dog로 분류한 #1 같은 중대한 종 오분류가 남아 있다. 따라서 이 canary 15개 결과만으로 240개 전체의 정확도를 확정하지 말고, 특히 cat↔dog 변경 frame은 실제 저장 전 개별 검토가 필요하다. 이 단계에서는 어떤 저장소도 update하지 않았다.
