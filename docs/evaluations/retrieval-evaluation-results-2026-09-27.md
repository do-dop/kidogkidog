# Retrieval evaluation result

Status: `not_run_ground_truth_pending`

Evaluation scope: 240 eligible frames (IMG_8450_2: 33, dog_drinking_water: 89, dog_escape: 28, two_dogs: 90). Deleted stale frames are excluded.

No retrieval run or metric calculation has been performed. Human relevance labels must be added to [retrieval-ground-truth-2026-09-27.json](retrieval-ground-truth-2026-09-27.json) first. Blank judgments are not treated as negative examples.

After labeling, `scripts/evaluate_retrieval.py --mode run --top-k 5` will write the actual per-query results and metrics here.
