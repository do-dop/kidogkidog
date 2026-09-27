# YOLO cat/dog temporal consistency evaluation

- Scope: exact 240 eligible frames only; no stale `two_dogs` 85 or `cats_5min` 292 frames used.
- YOLO, thresholds, raw detections, Chroma, MySQL and behavior events were not changed.
- Chunks were grouped by `(video_id, exact filename stem before _frame_)`; no cross-chunk neighbors. The 240 eligible rows contain 8 exact chunks (7 multi-frame chunks; median interval 2.0 seconds).
- Ground truth: 107 previously visually reviewed frames plus 15 canary reviews, deduplicated to 110 unique frame IDs.
- Method labels use existing 0.5 threshold; no threshold retuning. Raw `object_detections` stayed unchanged.

## Baseline

| class | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| cat | 20 | 12 | 1 | 0.625 | 0.952 | 0.755 |
| dog | 45 | 6 | 32 | 0.882 | 0.584 | 0.703 |

Reviewed sample: 110 unique frames. Precision/recall are only for visually adjudicated labels and are not representative population estimates.

## Temporal candidate comparison

Rows report class TP/FP/FN (F1); last column is reviewed frames changed / improved / worsened versus baseline. Windows include the current frame plus same-chunk neighbors.

| candidate | window | rule | cat TP/FP/FN (F1) | dog TP/FP/FN (F1) | changed / improved / worsened |
|---|---|---|---|---|---:|
| majority_ordinal_1 | ± 1 | majority | cat 20/7/1 (0.833) | dog 47/6/30 (0.723) | 10/8/2 |
| confidence_mean_ordinal_1 | ± 1 | confidence_mean | cat 21/5/0 (0.894) | dog 44/5/33 (0.698) | 18/13/5 |
| confidence_weighted_ordinal_1 | ± 1 | confidence_weighted | cat 21/6/0 (0.875) | dog 41/6/36 (0.661) | 13/8/5 |
| majority_ordinal_2 | ± 2 | majority | cat 21/8/0 (0.840) | dog 47/6/30 (0.723) | 16/10/6 |
| confidence_mean_ordinal_2 | ± 2 | confidence_mean | cat 21/2/0 (0.955) | dog 39/5/38 (0.645) | 22/14/8 |
| confidence_weighted_ordinal_2 | ± 2 | confidence_weighted | cat 21/5/0 (0.894) | dog 40/6/37 (0.650) | 16/10/6 |
| majority_seconds_4 | ±s 4 | majority | cat 21/5/0 (0.894) | dog 47/6/30 (0.723) | 17/12/5 |
| confidence_mean_seconds_4 | ±s 4 | confidence_mean | cat 21/2/0 (0.955) | dog 39/5/38 (0.645) | 22/14/8 |
| confidence_weighted_seconds_4 | ±s 4 | confidence_weighted | cat 21/6/0 (0.875) | dog 40/6/37 (0.650) | 13/8/5 |
| majority_seconds_8 | ±s 8 | majority | cat 21/6/0 (0.875) | dog 47/5/30 (0.729) | 20/14/6 |
| confidence_mean_seconds_8 | ±s 8 | confidence_mean | cat 21/4/0 (0.913) | dog 41/5/36 (0.667) | 24/14/8 |
| confidence_weighted_seconds_8 | ±s 8 | confidence_weighted | cat 21/4/0 (0.913) | dog 41/6/36 (0.661) | 14/10/4 |

Definitions: majority votes independent binary per-class detections at 0.5 and requires vote share ≥0.5; confidence mean averages each class maximum raw confidence (missing=0) and applies the existing 0.5 threshold; confidence-weighted mean uses weight `1/(1+distance)` with distance measured in frame order or seconds. Cat and dog remain multi-label; no forced single-species classification. Majority exact ties are unresolved.

## Best observed candidate

Best by macro F1: `majority_seconds_4` (macro F1 0.808) versus baseline 0.729. This is a small, thresholded candidate search on the same reviewed sample, so its apparent win is optimistic and not a held-out result.

## Known misclassification frames

| frame | visual truth | baseline | best candidate output and scores |
|---|---|---|---|
| IMG_8450_2__petcam_20260926_173123_000_frame_66.00 | ['cat'] | ['dog'] / raw {'cat': 0.4697346091270447, 'dog': 0.8823254704475403} | ['cat', 'dog'] / scores {'cat': 0.6667, 'dog': 1.0} |
| dog_drinking_water__petcam_20260926_173602_000_frame_12.50 | ['dog'] | ['cat'] / raw {'cat': 0.7299715876579285, 'dog': 0.0} | ['cat'] / scores {'cat': 0.75, 'dog': 0.25} |
| dog_drinking_water__petcam_20260926_173602_000_frame_14.50 | ['dog'] | ['cat'] / raw {'cat': 0.627281904220581, 'dog': 0.0} | [] / scores {'cat': 0.5, 'dog': 0.5} |
| dog_escape__petcam_20260926_173415_000_frame_12.00 | ['dog'] | [] / raw {'cat': 0.0, 'dog': 0.42569711804389954} | [] / scores {'cat': 0.0, 'dog': 0.0} |

The named 66.0s, 12.5s and 14.5s frames are included in the canary visual labels; 12.5s and 14.5s share the exact `petcam_20260926_173602_000` chunk. The dog .426 case is dog_escape 12.0s, also canary-reviewed; it is below the label threshold but remains raw evidence.


### Requested known-error outcomes under the top temporal candidate

| frame | truth | baseline | majority ±4s | outcome |
|---|---|---|---|---|
| IMG_8450_2 66.0s | cat | dog | cat + dog | cat recovered, but erroneous dog remains; not fully corrected |
| dog_drinking_water 12.5s | dog | cat | cat | error remains |
| dog_drinking_water 14.5s | dog | cat | unresolved (no cat/dog label) | false cat cleared, dog not recovered |
| dog_escape 12.0s | dog | unresolved | unresolved | .426 dog evidence not recovered |

The top-scoring temporal method fully corrected **0 of these 4** cases. It partly helps at 66.0s by adding cat, and at 14.5s by removing the false cat, but does not reliably identify the actual species. Other windows/rules have per-case outputs in JSON; none should be treated as production-validated.

## Video-level prior (eligible frames only)

| dominance cutoff | cat TP/FP/FN (F1) | dog TP/FP/FN (F1) | changed / improved / worsened |
|---:|---|---|---:|
| 0.7 | 21/3/0 (0.933) | 54/5/23 (0.794) | 10/10/0 |
| 0.8 | 21/11/0 (0.792) | 46/5/31 (0.719) | 2/2/0 |
| 0.9 | 20/11/1 (0.769) | 46/6/31 (0.713) | 1/1/0 |

The video prior is an intentionally coarse comparison. It uses only eligible frames and changes single-species predictions only; it can still erase a minority species. No video-name prior is used.

## Unresolved policy

When both smoothed classes remain below 0.5, the candidate can emit no cat/dog label (`unresolved`) or retain the original per-frame result. Per-method outcomes and counts are in the JSON. Unresolved avoids inventing a class but can increase false negatives; retaining can preserve a single-frame mistake.

## Known examples and interpretation

- 66.0s `IMG_8450_2`: actual cat, raw dog .882 and cat .470. Temporal recovery is possible only if neighboring same-chunk cat evidence outweighs the dog evidence; inspect the method-by-method row in JSON.
- 12.5s and 14.5s `dog_drinking_water`: actual dog but baseline cat. They are same chunk and nearby; neighboring dog evidence may recover them, but can also propagate a wrong label across adjacent frames.
- 12.0s `dog_escape`: actual dog confidence .426, below object_labels threshold; neighborhood evidence tests whether it can be recovered without changing the threshold.
- The evaluation records every reviewed frame whose label set changes, including cases where a previously correct label becomes wrong. See `changes.worsened_frame_ids` in JSON.

## Fixed-threshold comparison (same reviewed frames)

Raw detections were collected at 0.4, so this comparison is limited to thresholds 0.4–0.8 and the same 110 reviewed frames. It does not change the operating threshold.

| confidence threshold | cat TP/FP/FN (F1) | dog TP/FP/FN (F1) |
|---:|---|---|
| 0.4 | 21/14/0 (0.750) | 51/6/26 (0.761) |
| 0.5 | 20/12/1 (0.755) | 45/6/32 (0.703) |
| 0.6 | 19/7/2 (0.808) | 39/6/38 (0.639) |
| 0.7 | 18/5/3 (0.818) | 36/6/41 (0.605) |
| 0.8 | 18/0/3 (0.923) | 29/3/48 (0.532) |

The fixed-threshold comparison helps separate threshold effects from temporal effects: temporal smoothing does not uniformly dominate a threshold choice, and each class trades precision against recall differently. These rows use the existing 0.4-collected raw detections only; sub-0.4 evidence is unavailable.

### Reading the threshold comparison

Among the tested **single common thresholds**, 0.4 had the highest macro F1 on this reviewed sample (0.756); majority ±4s was higher at 0.808, but it still worsened 5 reviewed frames and fully corrected none of the four named cases. The per-class threshold combination cat 0.8 / dog 0.4 gives sample macro F1 about 0.842, but selecting and evaluating those thresholds on the same small reviewed set is optimistic. The class tradeoff is visible: at 0.8 cat F1 rises to 0.923 while dog F1 falls to 0.532; at 0.4 dog F1 is 0.761 while cat F1 is 0.750. This is not evidence to change thresholds in this task.

### Unresolved versus retaining baseline

For majority ±4s, 53/240 frames have no cat/dog label after smoothing (36 are visually reviewed). Emitting unresolved on those frames gives macro F1 0.808; retaining their baseline labels instead gives macro F1 about 0.788, with fewer dog false negatives but more cat false positives. On the named 14.5s frame, unresolved is safer than keeping the known-wrong cat label, while the .426 dog at dog_escape 12.0s remains unresolved. The reviewed set is too small to decide a general policy.

## Conclusion

This is an exploratory evaluation on 110 unique visually labeled frames: 95 retained from the earlier 107-frame set and 15 canary reviews, with overlap deduplicated. It is not an independent holdout. Adjacent-frame smoothing may reduce isolated errors, but it can propagate a run of correlated errors and suppress real minority animals. The JSON records which proposed method has the top sample F1 and its regressions. Do not infer that the threshold should change or that a production rule is validated from this small sample.
