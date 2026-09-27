# 추천 질문 → behavior event 연결 평가 (2026-09-27)

## 재현 및 원인

수정 전 실제 DB 조회와 gpt-4o-mini 호출에서 `dog_drinking_water`의 LLM 응답은 `{"event_index": 1, "question": "물을 마신 장면이 있어?"}`였다. 이벤트 목록 첫 항목은 DB event #9(2.0~57.5초)였다. 최종 source_event_id는 #17(18.0~18.0초)로 변경됐다.

기존 흐름:
1. prompt의 event_index는 DB ID가 아니라 입력 목록의 1-based 번호다.
2. `_parse_behavior_questions()`는 번호와 문장을 유지한다.
3. `_validate_behavior_questions()`가 번호를 실제 event로 해석하지만 `list[str]`만 반환해 근거를 버린다.
4. `generate_questions_from_behavior_events()` → `suggest_query_items()`의 중복/category/LLM 우선/라벨 보충 선정도 문자열만 다룬다.
5. 최종 문장마다 `_best_event_for_question()`을 다시 실행한다. `(텍스트 점수, interestingness, confidence)` 최댓값이 source event가 된다.
6. `_build_question_item()` → `/suggestions`의 `question_sources`에 ID와 시간 범위가 담긴다.
7. 기존 React가 선택된 항목의 ID/시간을 `/query`에 전달한다.
8. `/query` → `run_rag_query()` → `_resolve_source_event()`에서 ID로 이벤트를 조회하고, 있으면 event-grounded 검색으로 간다. ID와 시간 범위가 모두 없으면 일반 검색을 한다.

점수는 공통 키워드마다 +1, 질문의 2글자 이상 토큰이 이벤트 텍스트에 포함되면 +0.2다. 형태소 분석은 없다. `마신`은 키워드 `마시`와 일치하지 않는다. #9는 `물`(+1), `물을`(+0.2)로 1.2점이다. #17은 evidence의 “위치해 있어” 때문에 질문 토큰 `있어`까지 일치하여 1.4점이다. 따라서 흥미도/신뢰도가 낮아도 #17이 이긴다. duration은 점수에 사용되지 않는다.

**정정:** #17도 DB action은 물 마시기다. 다른 행동으로 연결된 사례라고 단정할 수 없으며, 이번에 재현한 오류는 원본 #9 대신 다른 시점의 #17에 연결되는 것이다. #8은 물 마시기 action이 아니라 일반 motion 감지 이벤트다. 실제 저장 정보와 예시 ID를 구분해야 한다.

## 물 마시기 질문의 이벤트별 기존 점수

| event | action | target_object | start/end(초) | duration | score | interestingness / confidence |
|---|---|---|---|---:|---:|---|
| #9 | 반려동물이 물을 마시는 것으로 보임 | 물 그릇 | 2.0 / 57.5 | 55.5 | 1.2 | 0.9 / 0.8 |
| #15 | 반려동물이 물을 마시는 것으로 보임 | 물 그릇 | 6.0 / 6.0 | 0.0 | 1.2 | 0.9 / 0.8 |
| #14 | 반려동물이 물을 마시는 것으로 보임 | 물 그릇 | 9.5 / 12.5 | 3.0 | 1.2 | 0.9 / 0.8 |
| #7 | 반려동물이 물을 마시는 것으로 보임 | 물 그릇 | 12.5 / 27.5 | 15.0 | 1.2 | 0.9 / 0.8 |
| #11 | 반려동물이 물을 마시는 것으로 보임 | 물 그릇 | 16.5 / 22.0 | 5.5 | 1.2 | 0.9 / 0.8 |
| #13 | 반려동물이 물을 마시는 것으로 보임 | 물 그릇 | 23.0 / 31.5 | 8.5 | 1.2 | 0.9 / 0.8 |
| #10 | 반려동물이 물을 마시는 것으로 보임 | None | 28.0 / 58.0 | 30.0 | 1.2 | 0.9 / 0.8 |
| #6 | 반려동물이 물을 마시는 것으로 보임 | 여름의 물 그릇 | 30.0 / 64.5 | 34.5 | 1.2 | 0.9 / 0.8 |
| #16 | 반려동물이 물을 마시는 것으로 보임 | 물그릇 | 2.0 / 3.5 | 1.5 | 1.2 | 0.8 / 0.7 |
| #17 | 반려동물이 물을 마시는 것으로 보임 | 물 그릇 | 18.0 / 18.0 | 0.0 | 1.4 | 0.8 / 0.7 |
| #8 | 반려동물이 있는 구간으로 감지됨 | None | 3.5 / 9.5 | 6.0 | 1.0 | 0.55 / 0.45 |
| #12 | 반려동물이 있는 구간으로 감지됨 | None | 2.0 / 5.5 | 3.5 | 1.0 | 0.5 / 0.45 |

## 변경 방식

- 새 `generate_question_items_from_behavior_events()`는 검증을 통과한 질문과 원본 `source_event_id`, `original_event_id`, `event_start`, `event_end`, `question_source`를 함께 반환한다.
- 기존 `generate_questions_from_behavior_events()`는 문자열 API 호환 wrapper로 유지한다.
- LLM과 behavior rule fallback 모두 생성 시 사용한 이벤트의 ID/시간을 보존한다. `suggest_query_items()`는 기존 문장 선정 순서를 유지하면서 선택된 item을 그대로 반환한다. 문자열 재점수로 원본 이벤트를 교체하지 않는다.
- 라벨 질문은 원본 이벤트가 없으므로 기존 점수 매칭을 사용한다. 프레임 fallback은 source ID와 시간 범위가 없다.
- `_best_event_for_question()`는 양의 텍스트 일치 점수가 있는 이벤트만 비교한다. 0은 현재 수식에서 일치 근거가 하나도 없다는 정확한 경계다. 임의의 유사도 threshold를 추가하지 않았다. 모두 0이면 None이고 ID/시작/끝도 모두 None이다.
- duration=0 이벤트는 배제하지 않는다. 명시적인 원본 이벤트는 텍스트 점수가 0이어도 보존한다.
- 프롬프트, 금지 표현/반복 검증 조건, category/중복 제한은 유지했다. validator에는 통과한 후보의 metadata 반환 옵션만 추가했고, 매칭이 None인 경우 명시된 source를 사용하도록 방어했다. 기존 validator의 재매칭 검증은 유지되어 일부 후보를 보수적으로 탈락시킬 수 있지만 최종 원본을 교체하지 않는다.
- `/suggestions`, `/query`, React와 검색 코드는 수정하지 않았다. 기존 필드를 그대로 전달하며 추적용 `original_event_id`, `question_source`만 추가한다.

## 실제 수정 후 호출: limit=3

각 영상에 실제 DB/LLM을 사용해 전후 1회씩 호출했다. LLM 문장은 호출마다 달라질 수 있다. score는 최종 연결 이벤트에 대한 기존 수식의 값이다. source가 없는 경우 최고 후보 점수를 병기한다. '동일'은 생성 당시 근거 보존을 뜻하며 영상 픽셀 사실 검증은 아니다.

| 영상 | 질문 | 출처 | 원본 event | 최종 event | event action | target_object | start/end(초) | score | 동일 여부 |
|---|---|---|---|---|---|---|---|---|---|
| IMG_8450_2 | 자동 급식기 근처에 있는 장면이 있어? | LLM | 3 | 3 | 반려동물이 배식을 기다리는 것으로 보임 | 자동 급식기 | 4.5 / 9.0 | 0.6 | 동일 |
| IMG_8450_2 | 사료 그릇에 접근한 장면이 있어? | LLM | 2 | 2 | 반려동물이 사료 그릇에 접근하여 먹이를 먹는 것으로 보임 | 사료 그릇 | 12.0 / 66.5 | 2.4 | 동일 |
| IMG_8450_2 | 고양이가 보이는 장면 보여줘 | label fallback | None | 3 | 반려동물이 배식을 기다리는 것으로 보임 | 자동 급식기 | 4.5 / 9.0 | 0.2 | 원본 없음 |
| dog_drinking_water | 물 그릇에 가까이 다가간 장면이 있어? | LLM | 9 | 9 | 반려동물이 물을 마시는 것으로 보임 | 물 그릇 | 2.0 / 57.5 | 2.2 | 동일 |
| dog_drinking_water | 물을 마시는 장면이 있어? | LLM | 13 | 13 | 반려동물이 물을 마시는 것으로 보임 | 물 그릇 | 23.0 / 31.5 | 2.4 | 동일 |
| dog_drinking_water | 고양이가 보이는 장면 보여줘 | label fallback | None | 6 | 반려동물이 물을 마시는 것으로 보임 | 여름의 물 그릇 | 30.0 / 64.5 | 0.2 | 원본 없음 |
| dog_escape | 사람이 함께 보이는 장면 있어? | label fallback | None | None | — | — | None / None | — (최고 0.0) | 원본 없음 |
| dog_escape | 침대 근처에서 움직인 장면 찾아줘 | label fallback | None | 5 | 반려동물이 장애물을 넘어서는 것으로 보임 | 반려동물의 장난감이나 식기 | 2.0 / 56.06 | 1.0 | 원본 없음 |
| dog_escape | 침대을 살펴보는 장면 있어? | label fallback | None | None | — | — | None / None | — (최고 0.0) | 원본 없음 |
| two_dogs | 고양이가 보이는 장면 보여줘 | label fallback | None | None | — | — | None / None | — (최고 0.0) | 원본 없음 |
| two_dogs | 고양이가 움직인 장면 찾아줘 | label fallback | None | 1 | 반려동물이 있는 구간으로 감지됨 | None | 0.0 / 178.0 | 1.0 | 원본 없음 |
| two_dogs | 강아지가 보이는 장면 보여줘 | label fallback | None | None | — | — | None / None | — (최고 0.0) | 원본 없음 |

## 동일 LLM 응답 재생: 정확히 “물을 마신 장면이 있어?”

수정 후 실제 새 LLM 호출에서는 “물을 마시는 장면이 있어?”가 원본 #13 → 최종 #13(23.0~31.5초, action 물 마시기, target 물 그릇)으로 유지됐다.
별도로 수정 전 **실제로 수집한 원본 LLM JSON과 같은 이벤트 snapshot**을 수정 후 코드에 재생했다. 이 검증에서는 DB/LLM 호출을 snapshot/기록 응답으로 대체했고, 생성 파싱·validation·최종 선정·연결 코드는 실제 함수를 실행했다. 새 실제 생성 결과와 구분한다.

| 실행 | 질문 | 출처 | 원본 event | 최종 event | action | target | start/end | 최종 score | 동일 |
|---|---|---|---|---|---|---|---|---:|---|
| 수정 전 실제 호출 | 물을 마신 장면이 있어? | LLM | #9 | #17 | 물을 마시는 것으로 보임 | 물 그릇 | 18.0 / 18.0 | 1.4 | 아니오 |
| 수정 후 동일 응답 재생 | 물을 마신 장면이 있어? | LLM | #9 | #9 | 물을 마시는 것으로 보임 | 물 그릇 | 2.0 / 57.5 | 1.2 | 예 |

## 테스트 및 수정 파일

- `pipeline/question_generator.py`: 구조화된 item 생성 API, 기존 문자열 wrapper, 검증 통과 지점의 source metadata 보존, rule fallback metadata 보존.
- `pipeline/query_suggester.py`: item metadata 유지, 출처 표기, 0점 무연결 정책.
- `tests/test_question_event_linking.py`: 신규 10개 테스트. 원본 보존, 다른 이벤트의 더 높은 점수, rule/label/frame fallback, 0점 None, 0초/0점 원본 보존, 3개 제한과 LLM 우선+보충, 문자열 API 호환, HTTP /query 일반 검색, /suggestions→/query event-grounded 연결 검증.
- `.venv/bin/python -m unittest discover -s tests`: **46개 통과** (기존 36개 + 신규 10개). 기존 검색 관련 24개와 질문 품질 12개도 모두 통과.
- `git diff --check` 통과. 프롬프트/문장 거절 함수/LLM JSON 파서가 이번 변경 전과 동일함을 AST 비교로 확인.

## 범위 밖 관찰

- 기존 점수의 `물`은 `반려동물`에도 일치한다. #8/#12의 일반 감지 이벤트가 1.0점을 받는 이유다. 0점 정책이 양의 점수 오매칭까지 해결하지는 않는다. 점수 수식은 이번에 변경하지 않았다.
- LLM은 여전히 문장 변동이 있다. dog_escape의 실제 수정 후 응답 “장애물을 넘으려는 장면이 있어?”는 기존 validation에서 제거돼 라벨 질문으로 보충됐다.
- “침대을” 조사 오류와 라벨의 고양이/강아지 판별 문제는 수정하지 않았다.
- 기존 validation은 재매칭 결과도 중복/반복 검사에 쓰므로 원본이 서로 다른 후보를 탈락시킬 수 있다. validation 변경 금지 범위라 유지했다.
- 이벤트 생성, CLIP/ChromaDB ranking, event 범위, chunk, ±3초 buffer, React는 수정하지 않았다.
