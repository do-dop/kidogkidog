# cats_5min legacy cleanup result

Generated 2026-09-27. Cleanup was limited to all Chroma rows whose `video_id` is `cats_5min` and all current MySQL scene rows with that exact video ID. Search history, behavior events, and other videos were preserved.

## Final reference audit

- MySQL 8.4.11 tables: `behavior_events`, `scenes`, `search_logs`, `user_frequent_queries`; no separate video/recording table.
- Before cleanup: Chroma 292, scenes 643, behavior events 0, local frames 0, GCS frames 0, GCS chunks 0.
- Scene foreign keys and other-table references to the target frame/scene data: 0.
- Historical search logs with `video_id=cats_5min`: 8 rows preserved. `user_frequent_queries`: 33 rows preserved.
- UI video choices come from recordings API data; with no GCS chunk, cats_5min is not a playable recording. The UI uses its existing demo list when no GCS recordings are returned.

## Backup and cleanup

- Backup: `docs/evaluations/backups/cats-5min-legacy-cleanup-before-2026-09-27.json`; SHA-256: `62a94794195928553f31b7ba17efdcf24fc8b5d7a4324766c97853843d4b754e`.
- Chroma records backed up and removed: 292 / 292.
- Full MySQL scene rows backed up and removed by exact scene IDs: 643 / 643.
- After cleanup: Chroma collection 240; cats_5min Chroma 0; cats_5min scenes 0.
- Verification: `passed`. All non-cats Chroma document/metadata/embedding snapshots and all other-video scene counts were unchanged.

## Search candidates before and after

Counts below are direct object-filter search candidates (`cat` / `dog` / `person`):

| video | before | after |
|---|---|---|
| cats_5min | 286 / 4 / 0 | 0 / 0 / 0 |
| IMG_8450_2 | 33 / 0 / 0 | 33 / 0 / 0 |
| dog_drinking_water | 2 / 84 / 1 | 2 / 84 / 1 |
| dog_escape | 0 / 7 / 6 | 0 / 7 / 6 |
| two_dogs | 0 / 74 / 9 | 0 / 74 / 9 |

All four other videos retained the same search frame IDs before and after cleanup.

## Recommendation and query context

- cats_5min suggestions: 3 stale frame-fallback questions before cleanup; 0 after cleanup.
- cats_5min scene context: before, dominant labels were `cat(630), bowl(23), dog(7)` with four extracted scene events; after, dominant objects and scene events are empty.
- Historical/global frequent queries remain in the prompt context because search history was preserved. The top global query still includes “고양이”; this is query-history text, not a cats_5min scene/frame reference.
- Other videos’ scene-context snapshots and search frame IDs were unchanged. Suggestions were smoke-tested with limit 3; LLM phrasing can vary between calls.
- cats_5min recordings API remains `count=0`; there is no thumbnail key or media item from a source chunk. No `/media` reference remains through search because the Chroma frames are gone.

## Rollback

- Dry-run found 292 Chroma IDs and 643 scene IDs restorable, with 0 collisions. No rollback writes were performed.
- The backup retains original vectors, documents, and metadata. During an earlier aborted cleanup attempt, Chroma re-add caused 14 stale embeddings to differ from the original backup by at most one adjacent float32 value (maximum absolute delta `2.98e-8`). Consequently, rollback can restore IDs, documents, metadata, and effectively identical float32 vectors, but byte-identical embedding restoration is not guaranteed for those 14 rows.

Full before/after snapshots, preserved historical references, and per-video question provenance are in the JSON report.
