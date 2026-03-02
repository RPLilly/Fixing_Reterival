from __future__ import annotations

import os
import threading
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.services.ingest_jobs import (
    append_error,
    create_job,
    get_job,
    mark_finished,
    mark_started,
    stable_chunk_key,
    update_job,
)
from app.services.llm_gateway import get_access_token, get_embeddings
from app.services.vector_store import embedding_exists, insert_embedding

router = APIRouter(prefix="/ingest", tags=["ingest"])


class ChunkMetadata(BaseModel):
    keywords: str | None = None
    entities: str | None = None
    entities_list: list[Any] = Field(default_factory=list)
    filename: str | None = None


class ChunkIn(BaseModel):
    text: str
    source: str | None = None
    metadata: ChunkMetadata | None = None


class IngestResponse(BaseModel):
    job_id: str
    status_url: str


def _check_admin_key(provided_key: str | None) -> None:
    expected = os.getenv("ADMIN_KEY")
    if expected:
        if not provided_key or provided_key != expected:
            raise HTTPException(status_code=401, detail="Invalid admin_key")


def _run_ingest_job(job_id: str, chunks: list[dict[str, Any]]) -> None:
    """Background worker to embed + store chunks.

    Notes:
    - Uses a per-process token cache in llm_gateway.get_access_token.
    - Dedupes within the request (by content hash) and against DB (by text+filename).
    """

    mark_started(job_id)

    # 1) Request-level dedupe: keep first occurrence.
    seen: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    for chunk in chunks:
        key = stable_chunk_key(chunk)
        if key in seen:
            update_job(job_id, skipped_duplicates=get_job(job_id).skipped_duplicates + 1)  # type: ignore[union-attr]
            continue
        seen.add(key)
        deduplicated.append(chunk)

    update_job(job_id, total=len(deduplicated))
    # 2) Filter out obvious invalid chunks.
    filtered: list[dict[str, Any]] = []
    for idx, chunk in enumerate(deduplicated):
        text = (chunk.get("text") or "").strip()
        if not text:
            append_error(job_id, index=idx, message="Empty text; skipped")
            update_job(job_id, failed=get_job(job_id).failed + 1)  # type: ignore[union-attr]
            continue
        filtered.append(chunk)

    if not filtered:
        mark_finished(job_id, status="completed")
        return

    access_token = get_access_token()

    # 3) Batch embeddings.
    texts = [chunk["text"] for chunk in filtered]
    try:
        embeddings = get_embeddings(texts, access_token)
    except Exception as e:
        append_error(job_id, index=-1, message="Embedding request failed", detail=str(e))
        mark_finished(job_id, status="failed")
        return

    # 4) Store, with best-effort dedupe against DB.
    embedded_count = 0
    stored_count = 0
    failed_count = 0

    for idx, (chunk, embedding_obj) in enumerate(zip(filtered, embeddings)):
        try:
            embedding = embedding_obj.get("embedding") if isinstance(embedding_obj, dict) else embedding_obj
            if embedding is None:
                failed_count += 1
                append_error(job_id, index=idx, message="Missing embedding in response")
                continue

            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            filename = metadata.get("filename")

            # DB-level dedupe: avoid re-inserting identical text+filename.
            if embedding_exists(chunk_text=chunk["text"], filename=filename):
                update_job(job_id, skipped_duplicates=get_job(job_id).skipped_duplicates + 1)  # type: ignore[union-attr]
                embedded_count += 1
                continue

            insert_embedding(chunk["text"], embedding, filename=filename)
            embedded_count += 1
            stored_count += 1
        except Exception as e:
            failed_count += 1
            append_error(job_id, index=i, message="Failed to store embedding", detail=str(e))

        # Update progress occasionally.
        if (idx + 1) % 25 == 0:
            update_job(job_id, embedded=embedded_count, stored=stored_count, failed=failed_count)

    update_job(job_id, embedded=embedded_count, stored=stored_count, failed=failed_count)
    mark_finished(job_id, status="completed" if failed_count == 0 else "failed")


@router.post("/chunks", response_model=IngestResponse)
async def ingest_chunks(payload: list[ChunkIn], background_tasks: BackgroundTasks, admin_key: str | None = None):
    """Accept chunks payload, enqueue embedding+insert job, return job_id.

    Security:
    - If ADMIN_KEY is set, requires matching admin_key query param.
    """

    _check_admin_key(admin_key)

    job = create_job(total=len(payload))
    chunks = [item.model_dump() for item in payload]

    # FastAPI BackgroundTasks execute after response is returned.
    background_tasks.add_task(_run_ingest_job, job.job_id, chunks)

    return IngestResponse(job_id=job.job_id, status_url=f"/ingest/status/{job.job_id}")


@router.get("/status/{job_id}")
async def ingest_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "total": job.total,
        "embedded": job.embedded,
        "stored": job.stored,
        "skipped_duplicates": job.skipped_duplicates,
        "failed": job.failed,
        "errors": job.errors,
    }
