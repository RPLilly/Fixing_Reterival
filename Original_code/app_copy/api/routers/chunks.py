from __future__ import annotations

from fastapi import APIRouter, Query

from app.services.vector_store import count_chunk_embeddings, list_chunks


router = APIRouter(prefix="/chunks", tags=["chunks"])

@router.get("")
def get_chunks(
    filename: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Fetch chunks from the DB.

    Returns chunk text and filename but not embedding vectors.

    Query params:
    - filename: optional filter
    - limit/offset: pagination
    """

    total = count_chunk_embeddings(filename=filename)
    items = list_chunks(filename=filename, limit=limit, offset=offset)
    return {
        "total": total,
        "count": len(items),
        "limit": limit,
        "offset": offset,
        "filename": filename,
        "items": items,
    }
