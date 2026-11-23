"""Precompute multilingual vectors for every stored behavior event.

The vector search only narrows candidates. The existing action matcher still
checks subject, object and direction before a scene is presented as evidence.
"""

from __future__ import annotations

import os

from pipeline.storage.vector_store import get_event_collection


EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64


def _event_id(event: dict) -> str:
    return f"behavior-event-{int(event['id'])}"


def _event_text(event: dict) -> str:
    return "\n".join(
        f"{label}: {event.get(key) or ''}"
        for label, key in (("주체", "subject"), ("행동", "action"),
                           ("대상", "target_object"), ("장면", "summary"))
    )


def _embed(client, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def index_behavior_events(events: list[dict]) -> int:
    """Idempotently index committed database events; never index unsaved rows."""
    rows = [event for event in events if event.get("id") is not None and event.get("video_id")]
    if not rows or not os.getenv("OPENAI_API_KEY"):
        return 0
    from openai import OpenAI

    collection = get_event_collection()
    client = OpenAI()
    indexed = 0
    for offset in range(0, len(rows), BATCH_SIZE):
        group = rows[offset:offset + BATCH_SIZE]
        existing = set(collection.get(ids=[_event_id(event) for event in group], include=[]).get("ids") or [])
        batch = [event for event in group if _event_id(event) not in existing]
        if not batch:
            continue
        texts = [_event_text(event) for event in batch]
        embeddings = _embed(client, texts)
        if len(embeddings) != len(batch):
            raise RuntimeError("행동 이벤트 임베딩 수가 저장할 이벤트 수와 다릅니다.")
        collection.upsert(
            ids=[_event_id(event) for event in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"video_id": str(event["video_id"]),
                        "event_id": int(event["id"]),
                        "start_time": float(event.get("start_time") or 0)} for event in batch],
        )
        indexed += len(batch)
    return indexed


def semantic_event_candidates(query: str, events: list[dict], limit: int = 48,
                              video_id: str | None = None) -> list[dict] | None:
    """Return ranked candidates, or None when the index is incomplete/unavailable."""
    if not events or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        ids = [_event_id(event) for event in events]
        collection = get_event_collection()
        indexed = set(collection.get(ids=ids, include=[]).get("ids") or [])
        if not set(ids).issubset(indexed):
            return None

        from openai import OpenAI
        vector = _embed(OpenAI(), [query])[0]
        query_args = dict(
            query_embeddings=[vector],
            n_results=min(max(1, limit), len(events)),
            include=["distances"],
        )
        if video_id:
            query_args["where"] = {"video_id": video_id}
        result = collection.query(**query_args)
        by_id = {_event_id(event): event for event in events}
        found = [by_id[item] for item in (result.get("ids") or [[]])[0] if item in by_id]
        return found or None
    except Exception as exc:
        print(f"행동 이벤트 의미 색인 조회 실패: {exc}", flush=True)
        return None
