# Stale cleanup: two_dogs 85 and cats_5min scene audit

Generated 2026-09-27. The two_dogs deletion used only the exact 85 stale frame IDs in the ineligible manifest and their unique scene IDs. cats_5min was read-only.

## two_dogs cleanup verification

- Chroma collection count after cleanup: 532
- Stale Chroma frames remaining: 0/85
- Eligible Chroma frames preserved byte-for-byte (embedding float32 bytes, document, metadata): 90/90
- Stale MySQL scene rows remaining: 0/85
- Eligible MySQL scene rows preserved: 90/90
- Behavior event references remaining: 0

| Query | Candidate count after cleanup | Stale IDs in results |
|---|---:|---:|
| cat | 0 | 0 |
| dog | 74 | 0 |
| person | 9 | 0 |

Post-cleanup `suggest_query_items(video_id="two_dogs", limit=3)` returned:

- 강아지가 보이는 장면 보여줘 — label fallback; source_event_id=None; event=None–None
- 강아지가 움직인 장면 찾아줘 — label fallback; source_event_id=1; event=0.0–178.0
- 소파 근처에서 움직인 장면 찾아줘 — label fallback; source_event_id=1; event=0.0–178.0

Stale cat fallback present: **False**. Limit respected: **True**.

## cats_5min: 351 non-manifest scene rows

- Total scenes: 643; manifest stale rows: 292; other rows: 351.
- Other row chunk/stem counts: `{"cats_5min": 146, "petcam_20260528_042322_000": 30, "petcam_20260528_042322_001": 29, "petcam_20260528_042322_002": 29, "petcam_20260528_042322_003": 29, "petcam_20260528_042322_004": 29, "petcam_20260528_042623_000": 30, "petcam_20260528_042623_001": 29}`.
- Other rows have local frame files: 0/351; exact GCS frame objects: 0/351.
- Other rows with exact Chroma key match: 0/351; with same chunk stem: 0/351.
- Other rows span 175 unique `start_time` values; 59 values are repeated, accounting for 176 extra rows.
- Scene label counts before removing the 292 manifest rows: `{"cat": 630, "bowl": 23, "dog": 7}`.
- Scene label counts remaining if only those 292 rows were removed: `{"cat": 344, "bowl": 11, "dog": 3}`.
- GCS source video chunks: 0; behavior event rows/references: 0/0.

The exact 85 frame and scene rows can be restored from the backup JSON. `rollback_dry_run` found 85 Chroma records and 85 MySQL rows restorable; no rollback writes were performed.

Full row-level cats_5min scene metadata and frame/chunk matching details are in the adjacent JSON report.
