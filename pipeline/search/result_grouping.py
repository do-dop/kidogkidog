"""Present ranked scene hits as one result per source video."""

from __future__ import annotations

from typing import Any


def group_results_by_video(results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep the best scene as the card and retain every scene for playback.

    Only a real video_id is safe to group on. Unknown IDs remain separate so
    unrelated recordings cannot be accidentally merged.
    """
    groups: dict[tuple[str, str | int], list[dict[str, Any]]] = {}
    for index, result in enumerate(results):
        video_id = result.get("video_id")
        key = ("video", str(video_id)) if video_id else ("unknown", index)
        groups.setdefault(key, []).append(result)

    grouped = []
    for scenes in groups.values():
        ranked = sorted(scenes, key=lambda row: float(row.get("score") or 0), reverse=True)
        representative = dict(ranked[0])
        representative["matched_scene_count"] = len(ranked)
        representative["matched_frame_count"] = sum(
            max(1, int(scene.get("merged_frame_count") or 1)) for scene in ranked
        )
        representative["related_scenes"] = [dict(scene) for scene in ranked]
        grouped.append(representative)

    grouped.sort(key=lambda row: float(row.get("score") or 0), reverse=True)
    return grouped[:max(0, limit)]
