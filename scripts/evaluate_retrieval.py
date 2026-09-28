#!/usr/bin/env python3
"""Read-only retrieval quality runner and scorer for the current RAG path.

This tool never writes to Chroma, MySQL, or behavior events. `run` calls the
same `pipeline.rag_chain.run_rag_query` function used by `/query`, without a
user_id (so it cannot create a search-log row), and suppresses only answer
generation because metrics use retrieved frames rather than prose.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "docs/evaluations/retrieval-ground-truth-2026-09-27.json"
DEFAULT_RESULTS = ROOT / "docs/evaluations/retrieval-evaluation-results-2026-09-27.json"
DEFAULT_REPORT = ROOT / "docs/evaluations/retrieval-evaluation-results-2026-09-27.md"
EXPECTED_COUNTS = {
    "IMG_8450_2": 33,
    "dog_drinking_water": 89,
    "dog_escape": 28,
    "two_dogs": 90,
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_scope(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    manifest_path = (ROOT / dataset["scope"]["eligible_manifest"]).resolve()
    manifest = _read_json(manifest_path)
    frames = manifest.get("frames", [])
    ids = {str(frame["frame_id"]) for frame in frames if frame.get("frame_id")}
    actual_counts: dict[str, int] = defaultdict(int)
    for frame in frames:
        actual_counts[str(frame.get("video_id") or "")] += 1
    if len(frames) != dataset["scope"]["expected_total"] or actual_counts != EXPECTED_COUNTS:
        raise ValueError(
            "Eligible manifest scope mismatch; refusing to run. "
            f"total={len(frames)}, video_counts={dict(actual_counts)}"
        )
    return frames, ids


def _chunk_stem(frame: dict[str, Any]) -> str | None:
    path = str(frame.get("s3_key") or frame.get("frame_path") or "")
    name = Path(path).stem
    marker = "_frame_"
    return name.split(marker, 1)[0] if marker in name else None


def _timestamp(frame: dict[str, Any]) -> float | None:
    try:
        return float(frame["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None


def _query_relevant_ids(
    query: dict[str, Any], manifest_frames: list[dict[str, Any]], eligible_ids: set[str]
) -> tuple[set[str], list[str]]:
    gt = query.get("ground_truth") or query
    frame_ids = gt.get("relevant_frame_ids", gt.get("relevant_frames", []))
    relevant = {str(value) for value in frame_ids if str(value) in eligible_ids}
    for frame_id, judgment in (gt.get("frame_judgments") or {}).items():
        is_relevant = judgment.get("relevant") is True or judgment.get("relevance") == "relevant"
        if is_relevant and str(frame_id) in eligible_ids:
            relevant.add(str(frame_id))
    warnings: list[str] = []
    unknown_ids = [str(value) for value in frame_ids if str(value) not in eligible_ids]
    if unknown_ids:
        warnings.append(f"relevant frame IDs outside eligible manifest: {unknown_ids}")

    query_video = query.get("video_id")
    for time_range in gt.get("relevant_time_ranges", []):
        if time_range.get("relevance", "relevant") != "relevant":
            continue
        chunk_stem = time_range.get("chunk_stem")
        video_id = time_range.get("video_id") or query_video
        if not chunk_stem:
            warnings.append(f"time range missing required chunk_stem; ignored: {time_range}")
            continue
        if not video_id:
            warnings.append(f"global time range missing video_id; ignored: {time_range}")
            continue
        start = float(time_range["start"])
        end = float(time_range["end"])
        if end < start:
            warnings.append(f"invalid time range with end < start: {time_range}")
            continue
        for frame in manifest_frames:
            frame_id = str(frame.get("frame_id") or "")
            timestamp = _timestamp(frame)
            if frame_id not in eligible_ids or timestamp is None:
                continue
            if video_id and str(frame.get("video_id")) != str(video_id):
                continue
            if chunk_stem and _chunk_stem(frame) != chunk_stem:
                continue
            if start <= timestamp <= end:
                relevant.add(frame_id)
    return relevant, warnings


def _precision_at(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return sum(frame_id in relevant for frame_id in ranked_ids[:k]) / k


def _recall_at(ranked_ids: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return sum(frame_id in relevant for frame_id in ranked_ids[:k]) / len(relevant)


def _mrr(ranked_ids: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    for rank, frame_id in enumerate(ranked_ids, start=1):
        if frame_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_5(ranked_ids: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    dcg = sum(1.0 / math.log2(rank + 1) for rank, frame_id in enumerate(ranked_ids[:5], start=1) if frame_id in relevant)
    ideal_count = min(5, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else None


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def score_query(
    query: dict[str, Any],
    response: dict[str, Any],
    manifest_frames: list[dict[str, Any]],
    eligible_ids: set[str],
) -> dict[str, Any]:
    gt = query.get("ground_truth") or query
    relevant, warnings = _query_relevant_ids(query, manifest_frames, eligible_ids)
    if query.get("ground_truth_status") not in {"labeled", "reviewed_no_relevant"}:
        relevant = set()
        warnings.append("human ground truth is pending; relevance metrics are not scored")
    raw_results = response.get("results") or []
    in_scope = [item for item in raw_results if str(item.get("frame_id") or "") in eligible_ids]
    out_of_scope = [item for item in raw_results if str(item.get("frame_id") or "") not in eligible_ids]
    ranked_ids = [str(item["frame_id"]) for item in in_scope if item.get("frame_id")]
    metrics: dict[str, float | None] = {
        "precision_at_1": _precision_at(ranked_ids, relevant, 1) if relevant else None,
        "precision_at_3": _precision_at(ranked_ids, relevant, 3) if relevant else None,
        "precision_at_5": _precision_at(ranked_ids, relevant, 5) if relevant else None,
        "recall_at_3": _recall_at(ranked_ids, relevant, 3),
        "recall_at_5": _recall_at(ranked_ids, relevant, 5),
        "mrr": _mrr(ranked_ids, relevant),
        "ndcg_at_5": _ndcg_at_5(ranked_ids, relevant),
    }

    expected_videos = query.get("expected_video_ids")
    wrong_video_rate = None
    if expected_videos:
        expected = {str(value) for value in expected_videos}
        wrong_video_rate = (
            sum(str(item.get("video_id")) not in expected for item in in_scope) / len(in_scope)
            if in_scope else None
        )

    expected_labels = {str(value).lower() for value in query.get("expected_object_labels", [])}
    target_species = expected_labels & {"cat", "dog"}
    wrong_species_rate = None
    if target_species:
        opposite_species = {"cat", "dog"} - target_species
        frame_judgments = gt.get("frame_judgments") or {}
        applicable = [
            item for item in in_scope
            if str(item.get("frame_id")) in frame_judgments
            and frame_judgments[str(item.get("frame_id"))].get("visible_labels") is not None
        ]
        wrong_species_rate = (
            sum(
                bool({str(label).lower() for label in frame_judgments[str(item.get("frame_id"))].get("visible_labels", [])} & opposite_species)
                and not bool({str(label).lower() for label in frame_judgments[str(item.get("frame_id"))].get("visible_labels", [])} & target_species)
                for item in applicable
            ) / len(applicable)
            if applicable else None
        )

    return {
        "query_id": query["query_id"],
        "query": query["query"],
        "query_type": query["query_type"],
        "returned_count": len(in_scope),
        "no_result": len(in_scope) == 0,
        "ranked_frame_ids": ranked_ids,
        "relevant_frame_count": len(relevant),
        "relevant_hits": [frame_id for frame_id in ranked_ids if frame_id in relevant],
        "metrics": metrics,
        "wrong_species_result_rate": wrong_species_rate,
        "wrong_video_result_rate": wrong_video_rate,
        "out_of_scope_frame_ids": [str(item.get("frame_id")) for item in out_of_scope],
        "warnings": warnings,
        "ground_truth_status": query.get("ground_truth_status", "needs_human_ground_truth"),
        "species_judged_result_count": len(applicable) if target_species else 0,
    }


def _labels(frame: dict[str, Any]) -> list[str]:
    value = frame.get("object_labels", [])
    if isinstance(value, str):
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip().lower() for part in value if str(part).strip()]
    return []


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = (
        "precision_at_1", "precision_at_3", "precision_at_5",
        "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5",
    )
    scored = [row for row in rows if row["ground_truth_status"] == "labeled" and row["relevant_frame_count"] > 0]
    return {
        "query_count": len(rows),
        "scored_query_count": len(scored),
        "no_result_rate": sum(row["no_result"] for row in rows) / len(rows) if rows else None,
        **{key: _mean([row["metrics"][key] for row in scored]) for key in numeric_keys},
        "wrong_species_result_rate": _mean([row["wrong_species_result_rate"] for row in rows]),
        "wrong_video_result_rate": _mean([row["wrong_video_result_rate"] for row in rows]),
    }


def calculate_metrics(
    queries: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    manifest_frames: list[dict[str, Any]],
    eligible_ids: set[str],
) -> dict[str, Any]:
    rows = [score_query(q, responses.get(q["query_id"], {}), manifest_frames, eligible_ids) for q in queries]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["query_type"]].append(row)
    return {
        "overall": _aggregate(rows),
        "by_query_type": {key: _aggregate(value) for key, value in sorted(by_type.items())},
        "per_query": rows,
    }


def _normalize_frame(item: dict[str, Any]) -> dict[str, Any]:
    keys = ("frame_id", "video_id", "timestamp", "frame_path", "s3_key", "object_labels", "score", "clip_similarity", "retrieval_source", "event_window")
    normalized = {key: item.get(key) for key in keys if key in item}
    for key in ("timestamp", "score", "clip_similarity"):
        if normalized.get(key) is not None:
            try:
                normalized[key] = float(normalized[key])
            except (TypeError, ValueError):
                normalized[key] = None
    labels = normalized.get("object_labels")
    if isinstance(labels, set):
        normalized["object_labels"] = sorted(labels)
    return normalized


def run_retrieval(dataset: dict[str, Any], queries: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    if top_k < 5 or top_k > 10:
        raise ValueError("top_k must be between 5 and 10 to match the /query cap and support @5 metrics")
    # Follow the repository's documented .env launch convention, without printing values.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    from pipeline import rag_chain, vector_store

    # The production helper uses get_or_create_collection(). Prime its cache
    # through read-only get_collection() first, and refuse to create a missing
    # collection/local database during an evaluation run.
    import chromadb

    if vector_store.CHROMA_HOST:
        client = chromadb.HttpClient(
            host=vector_store.CHROMA_HOST,
            port=vector_store.CHROMA_PORT,
            ssl=vector_store.CHROMA_SSL,
        )
    else:
        if not vector_store.CHROMA_DIR.exists():
            raise RuntimeError("Configured local Chroma path does not exist; refusing to initialize a database")
        client = chromadb.PersistentClient(path=str(vector_store.CHROMA_DIR))
    collection = client.get_collection(name=vector_store.CHROMA_COLLECTION)
    vector_store._client = client
    vector_store._collection = collection

    # Keep the production retrieval and event-selection path; skip only answer prose.
    rag_chain._generate_answer_with_langchain = lambda **_: rag_chain.DEFAULT_FALLBACK_MESSAGE
    manifest_frames, eligible_ids = load_scope(dataset)
    responses: dict[str, dict[str, Any]] = {}
    for index, query in enumerate(queries, start=1):
        args = {
            "query": query["query"],
            "video_id": query.get("video_id"),
            "top_k": top_k,
            "user_id": None,
            "recording_date": query.get("recording_date"),
            "time_range": query.get("time_range"),
            "source_event_id": query.get("source_event_id"),
            "event_start": query.get("event_start"),
            "event_end": query.get("event_end"),
        }
        args = {key: value for key, value in args.items() if value is not None}
        result = rag_chain.run_rag_query(**args)
        responses[query["query_id"]] = {
            "results": [_normalize_frame(item) for item in result.get("results", [])],
            "used_llm": False,
        }
        print(f"[{index}/{len(queries)}] {query['query_id']}: {len(responses[query['query_id']]['results'])} results")

    metrics = calculate_metrics(queries, responses, manifest_frames, eligible_ids)
    return {
        "status": "scored" if any(q.get("ground_truth_status") == "labeled" for q in queries) else "retrieval_only_ground_truth_pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": dataset["scope"],
        "retrieval": {"entrypoint": "pipeline.rag_chain.run_rag_query", "top_k": top_k, "user_id": None, "answer_generation": "suppressed; retrieval path unchanged"},
        "responses": responses,
        "metrics": metrics,
    }


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def render_report(result: dict[str, Any]) -> str:
    metrics = result.get("metrics", {})
    lines = [
        "# Retrieval evaluation result",
        "",
        f"Status: `{result.get('status', 'not_run')}`",
        f"Created: `{result.get('created_at', 'not run')}`",
        "",
        "This report scores only human-labeled judgments. Empty judgments are not treated as negative examples.",
        "",
    ]
    if not metrics:
        lines += ["Retrieval has not been run; no metric has been calculated.", ""]
        return "\n".join(lines)
    lines += ["## Overall and by query type", "", "| 유형 | query 수 | no-result | P@1 | P@3 | P@5 | R@3 | R@5 | MRR | nDCG@5 | wrong species | wrong video |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    aggregates = [("전체", metrics.get("overall", {})), *metrics.get("by_query_type", {}).items()]
    for name, row in aggregates:
        lines.append("| " + " | ".join([str(name), str(row.get("query_count", 0)), *[_fmt(row.get(key)) for key in (
            "no_result_rate", "precision_at_1", "precision_at_3", "precision_at_5", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5", "wrong_species_result_rate", "wrong_video_result_rate")]]) + " |")
    lines += ["", "## Per-query judgments", "", "| query_id | 질문 | GT relevant frames | P@1 | R@5 | MRR | nDCG@5 | 상태 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in metrics.get("per_query", []):
        lines.append("| " + " | ".join([str(row.get("query_id")), str(row.get("query", "")).replace("|", "\\|"), str(row.get("relevant_frame_count", 0)), _fmt(row.get("metrics", {}).get("precision_at_1")), _fmt(row.get("metrics", {}).get("recall_at_5")), _fmt(row.get("metrics", {}).get("mrr")), _fmt(row.get("metrics", {}).get("ndcg_at_5")), str(row.get("ground_truth_status"))]) + " |")
    lines.append("")
    return "\n".join(lines)


def _validate_queries(queries: list[dict[str, Any]]) -> None:
    ids = [query.get("query_id") for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("query_id values must be unique")
    allowed = {"object", "behavior", "location_target", "event_grounded"}
    for query in queries:
        query_type = query.get("type", query.get("query_type"))
        if query_type not in allowed:
            raise ValueError(f"Unsupported query type in {query.get('query_id')}: {query_type}")
        if not query.get("query"):
            raise ValueError(f"Missing query text in {query.get('query_id')}")


def _validate_labeled_ground_truth(queries: list[dict[str, Any]]) -> None:
    """Reject incomplete human-completed entries before metric scoring."""
    for query in queries:
        if query.get("ground_truth_status") != "labeled":
            continue
        query_id = query.get("query_id")
        scope = query.get("scope") or {}
        required_videos = set(scope.get("video_ids") or ([query["video_id"]] if query.get("video_id") else []))
        reviewed_videos = set(query.get("reviewed_video_ids") or [])
        if query.get("human_review_complete") is not True:
            raise ValueError(f"{query_id}: labeled requires human_review_complete=true")
        if not required_videos or not required_videos.issubset(reviewed_videos):
            raise ValueError(f"{query_id}: labeled requires review of every scoped video {sorted(required_videos)}")

        ranges = query.get("relevant_time_ranges") or []
        frame_ids = set(query.get("relevant_frame_ids") or [])
        judgments = query.get("frame_judgments") or {}
        frame_ids.update(fid for fid, row in judgments.items() if row.get("relevance") == "relevant" or row.get("relevant") is True)
        has_positive = bool(frame_ids) or any(row.get("relevance") == "relevant" for row in ranges)
        no_relevant = query.get("no_relevant_results")
        if not isinstance(no_relevant, bool):
            raise ValueError(f"{query_id}: labeled requires explicit no_relevant_results true/false")
        if no_relevant and has_positive:
            raise ValueError(f"{query_id}: cannot combine relevant judgments with no_relevant_results=true")
        if not no_relevant and not has_positive:
            raise ValueError(f"{query_id}: mark relevant result(s) or set no_relevant_results=true")

        for row in ranges:
            if row.get("relevance") not in {"relevant", "not_relevant", "uncertain"}:
                raise ValueError(f"{query_id}: invalid time-range relevance {row.get('relevance')!r}")
            if not row.get("video_id") or not row.get("chunk_stem"):
                raise ValueError(f"{query_id}: each range requires video_id and chunk_stem")
            if float(row.get("start", -1)) < 0 or float(row.get("end", -1)) < float(row.get("start", -1)):
                raise ValueError(f"{query_id}: invalid time range bounds")
            if row.get("relevance") == "uncertain" and not str(row.get("note") or "").strip():
                raise ValueError(f"{query_id}: uncertain time ranges require a note")
        for fid, row in judgments.items():
            if row.get("relevance") not in {"relevant", "not_relevant", "uncertain"}:
                raise ValueError(f"{query_id}: invalid frame judgment for {fid}")
            if row.get("relevance") == "uncertain" and not str(row.get("note") or "").strip():
                raise ValueError(f"{query_id}: uncertain frame {fid} requires a note")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "score"), default="run", help="run current retrieval or score a saved retrieval result")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = _read_json(args.dataset)
    queries = dataset.get("queries", [])
    _validate_queries(queries)
    _validate_labeled_ground_truth(queries)
    manifest_frames, eligible_ids = load_scope(dataset)
    if args.mode == "run":
        result = run_retrieval(dataset, queries, args.top_k)
    else:
        run_data = _read_json(args.results)
        if not run_data.get("retrieval") or "responses" not in run_data:
            raise ValueError("--mode score requires a saved retrieval run; no retrieval run was found")
        responses = run_data.get("responses", {})
        metrics = calculate_metrics(queries, responses, manifest_frames, eligible_ids)
        result = {**run_data, "status": "scored", "metrics": metrics}
    _write_json(args.results, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    print(f"Results JSON: {args.results}")
    print(f"Results Markdown: {args.report}")
    print(f"Eligible frame scope: {len(eligible_ids)}")
    if result.get("status") != "scored":
        print("Ground truth is not fully labeled; retrieval metrics remain pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
