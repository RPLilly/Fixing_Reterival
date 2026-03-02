from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class IngestJobSnapshot:
    job_id: str
    status: str  # queued|running|completed|failed
    created_at: float
    started_at: float | None
    finished_at: float | None
    total: int
    embedded: int
    stored: int
    skipped_duplicates: int
    failed: int
    errors: list[dict[str, Any]]


# In-memory job store (per-process). Good enough for a single-replica deployment.
# If you run multiple replicas, move this to Postgres/Redis.
_JOBS: dict[str, IngestJobSnapshot] = {}
_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def create_job(*, total: int) -> IngestJobSnapshot:
    job = IngestJobSnapshot(
        job_id=str(uuid.uuid4()),
        status="queued",
        created_at=_now(),
        started_at=None,
        finished_at=None,
        total=total,
        embedded=0,
        stored=0,
        skipped_duplicates=0,
        failed=0,
        errors=[],
    )
    with _LOCK:
        _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> IngestJobSnapshot | None:
    with _LOCK:
        return _JOBS.get(job_id)


def update_job(job_id: str, **patch) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        for key, value in patch.items():
            setattr(job, key, value)


def append_error(job_id: str, *, index: int, message: str, detail: Any | None = None) -> None:
    err = {"index": index, "message": message}
    if detail is not None:
        err["detail"] = detail
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job.errors.append(err)


def mark_started(job_id: str) -> None:
    update_job(job_id, status="running", started_at=_now())


def mark_finished(job_id: str, *, status: str) -> None:
    update_job(job_id, status=status, finished_at=_now())


def stable_chunk_key(chunk: dict[str, Any]) -> str:
    """Create a stable key for dedupe.

    We include:
    - text (dominant signal)
    - source
    - filename

    Using a hash keeps comparisons cheap.
    """

    text = str(chunk.get("text") or "")
    source = str(chunk.get("source") or "")
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    filename = str(metadata.get("filename") or "")

    canonical = json.dumps({"text": text, "source": source, "filename": filename}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
