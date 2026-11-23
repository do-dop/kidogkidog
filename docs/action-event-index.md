# 행동 이벤트 선색인

영상 처리 시 MySQL에 저장한 **모든 행동 이벤트**를 `text-embedding-3-small`로 한 번 임베딩해 별도 Chroma 컬렉션(`petcam_behavior_events_v1`)에 저장한다. 요약 화면에 표시되는 상위 5개만 색인하지 않는다. 이벤트 ID를 키로 사용하므로 재실행해도 중복되지 않는다.

자유 행동 질문에서는 같은 모델로 질문을 임베딩해 가까운 이벤트를 찾는다. 선택한 영상 안에서도, 영상을 선택하지 않은 전체 검색에서도 작동한다. UI의 기본 `00~24시` 값은 시간 제한으로 취급하지 않고, 날짜·시간을 실제로 좁혔을 때만 검증된 펫캠 촬영 시각으로 이벤트를 거른다. 벡터 순위는 후보 선정에만 쓰고, 기존 행동 비교 단계가 주체·대상·방향을 검증한다. 핵심 행동은 일치하지만 이벤트가 세부 조건을 입증하지 못하면 장면을 `related_event`로 보여주며 불확실성을 답변에 명시한다. 색인이 없거나 일부 이벤트가 빠졌으면 기존 검색 경로로 돌아간다. 원본 영상에서 추출되지 않은 행동은 이 색인으로 복구할 수 없다.

기존 영상 백필:

```bash
python -m dotenv -f .env run -- python -m scripts.backfill_action_event_index --all --dry-run
python -m dotenv -f .env run -- python -m scripts.backfill_action_event_index --all
```

`CHROMA_HOST`가 원격 서버이면 해당 서버에 연결할 수 있어야 한다. 백필은 저장된 행동 설명을 OpenAI 임베딩 API로 전송하며 비용이 든다. 새 청크는 DB 커밋 후 자동으로 색인한다. 색인에 실패해도 영상 처리는 완료되며 위 명령으로 누락분을 채울 수 있다.

추출 실패와 검색 실패를 나눠 평가할 때는 원본 영상을 사람이 확인한 JSON 사례를 `scripts.evaluate_action_search`에 넘긴다. 예를 들어 `[{"video_id":"smartbench_0608","query":"강아지가 몸을 터는 장면 보여줘","action_visible":true,"expected_event_ids":[157]}]`이다. 원본에 행동이 있지만 이벤트 ID가 없다면 추출 실패, 이벤트가 있는데 자유 질문 결과에 없다면 검색 실패로 분류한다.
