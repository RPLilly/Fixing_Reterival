from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vector_store import clear_chunk_embeddings, count_chunk_embeddings

router = APIRouter(prefix="/admin", tags=["admin"])


class ClearChunksRequest(BaseModel):
    # If provided, only delete chunks for this filename; otherwise clear all.
    filename: str | None = None
    # When clearing all, whether to reset SERIAL identity.
    reset_identity: bool = True
    # Simple guard to prevent accidental clearing.
    confirm: bool = False
    # Optional admin key validation.
    # NOTE: This is only required if the server has ADMIN_KEY set in env.
    admin_key: str | None = None


def _check_admin_key(provided_key: str | None) -> None:
    """Enforce admin key only when ADMIN_KEY is configured.

    If ADMIN_KEY is unset/empty, calls are allowed without any admin_key.
    """
    expected = os.getenv("ADMIN_KEY")
    if expected:
        if not provided_key or provided_key != expected:
            raise HTTPException(status_code=401, detail="Invalid admin_key")


@router.post("/clear-chunks")
def clear_chunks(req: ClearChunksRequest):
    """Clear chunk_embeddings rows.

    Safety:
    - Requires confirm=true
    - If ADMIN_KEY is set in env, requires matching admin_key
    """
    if req.confirm is not True:
        raise HTTPException(status_code=400, detail="confirm must be true to clear chunks")

    _check_admin_key(req.admin_key)

    before = count_chunk_embeddings(filename=req.filename)
    deleted = clear_chunk_embeddings(filename=req.filename, reset_identity=req.reset_identity)
    after = count_chunk_embeddings(filename=req.filename)

    return {
        "scope": "filename" if req.filename else "all",
        "filename": req.filename,
        "before": before,
        "deleted": deleted,
        "after": after,
    }
