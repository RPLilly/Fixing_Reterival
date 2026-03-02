from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from app.services.hybrid import hybrid_retrieve

router = APIRouter()


@router.get("/retrieve-chunks-hybrid")
async def retrieve_chunks_hybrid_get(
    query: str = Query(..., description="User query"),
    limit: int = Query(5, ge=1, le=50),
    alpha: float = Query(0.6, ge=0.0, le=1.0),
    file_filter: str | None = Query(None, description="Optional filename filter (metadata.filename)"),
    mode: str = Query("rrf", description="Fusion mode: rrf (default) or blend"),
    rrf_k: int = Query(60, ge=1, le=10000, description="RRF k constant (higher = less top-heavy)"),
):

    try:
        return await run_in_threadpool(
            hybrid_retrieve,
            query=query,
            limit=limit,
            alpha=alpha,
            file_filter=file_filter,
            mode=mode,
            rrf_k=rrf_k,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# @router.post("/retrieve-chunks-hybrid")
# async def retrieve_chunks_hybrid_post(request: Request):
#     try:
#         try:
#             body = await request.json()
#         except JSONDecodeError:
#             return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
#
#         query = body.get("query")
#         if not query or not isinstance(query, str):
#             return JSONResponse(status_code=400, content={"error": "Missing query"})
#
#         limit = int(body.get("limit", 5))
#         limit = max(1, min(limit, 50))
#
#         alpha = body.get("alpha", 0.6)
#         try:
#             alpha = float(alpha)
#         except Exception:
#             return JSONResponse(status_code=400, content={"error": "alpha must be a number"})
#         alpha = max(0.0, min(alpha, 1.0))
#
#         file_filter = body.get("file_filter")
#         if file_filter is not None and not isinstance(file_filter, str):
#             return JSONResponse(status_code=400, content={"error": "file_filter must be a string"})
#
#         return hybrid_retrieve(query=query, limit=limit, alpha=alpha, file_filter=file_filter)
#
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
#
