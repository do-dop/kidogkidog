# Retrieval quality evaluation ground-truth template

This template is for human relevance labeling before running `scripts/evaluate_retrieval.py`. It contains query wording found in existing tests and evaluation artifacts. It does not claim any frame is relevant.

The dataset is [retrieval-ground-truth-2026-09-27.json](retrieval-ground-truth-2026-09-27.json). It is scoped to the 240 exact eligible frames: IMG_8450_2 (33), dog_drinking_water (89), dog_escape (28), and two_dogs (90). Deleted stale frames are excluded.

## How to label

Review frames returned for each query and mark relevance from the visible video evidence. You can provide exact frame IDs, time ranges, or both. Time ranges use chunk-local seconds. When a video contains multiple chunks, set `chunk_stem` as well as `video_id`. Set `time_tolerance_seconds` only after choosing a consistent tolerance for the evaluation; `null` means no automatic expansion. Empty judgment arrays mean “not labeled,” not “no relevant frames.”

For species diagnostics, add reviewed result frames under `ground_truth.frame_judgments`. The IDs and labels below are placeholders showing syntax only, not judgments:

```json
"frame_judgments": {
  "REPLACE_WITH_REVIEWED_FRAME_ID": {
    "relevant": false,
    "visible_labels": ["REVIEWED_LABELS"]
  }
}
```

Only manually reviewed labels count toward `wrong_species_result_rate`. To calculate `wrong_video_result_rate` for a corpus-wide query, fill `expected_video_ids` with the video(s) where a person found relevant content. Leave it `null` if that has not been judged.

Each query has `ground_truth_status: "needs_human_ground_truth"`. After judgments are complete, change it to `"labeled"`. The evaluator ignores unlabeled queries for relevance-based aggregate metrics and reports missing judgments distinctly.

## Query groups

| Type | Existing query examples |
|---|---|
| Object | 고양이가 보이는 장면 보여줘; 강아지가 보이는 장면 보여줘; 사람이 함께 보이는 장면 있어? |
| Behavior | 물을 마신 장면이 있어?; 사료 그릇에 접근한 장면이 있어?; 개가 뛰는 장면; 반려동물이 움직인 장면은 언제였어? |
| Location/target | 그릇 근처; 자동 급식기 근처; 소파 근처; 문 앞; 소파 아래 |
| Event-grounded recommendation | 자동 급식기 근처 (#3); 사료 그릇 접근 (#2); 물 마시기 (#17) |

## Metric meanings

- **Precision@k**: relevant results among the first k, divided by k. It is reported only for queries with human-labeled relevant items.
- **Recall@k**: relevant retrieved items among all judged relevant eligible frames/range frames.
- **MRR**: reciprocal rank of the first relevant result.
- **nDCG@5**: ranking quality in the first five results, with binary relevance.
- **No-result rate**: fraction of executed queries that returned no in-scope frame; this can be reported before relevance labeling.
- **Wrong-species result rate**: among retrieved frames with human-reviewed visible cat/dog labels, the fraction showing only the opposite species for an explicit cat/dog query.
- **Wrong-video result rate**: among retrieved frames, the fraction outside the manually judged `expected_video_ids` set. It is unavailable until that set is labeled.

The runner calls `pipeline.rag_chain.run_rag_query`, the same retrieval entry point used by `/query`. It keeps query expansion, explicit object filtering, event-grounded candidate windows, and general CLIP retrieval in the existing path. It passes no `user_id` and suppresses answer prose generation, so the evaluation does not create search-log rows or depend on LLM wording.

## Run after labeling

```bash
python scripts/evaluate_retrieval.py --mode run --top-k 5
```

This writes `docs/evaluations/retrieval-evaluation-results-2026-09-27.json` and `.md`. To rescore a saved result file after editing judgments:

```bash
python scripts/evaluate_retrieval.py --mode score
```
