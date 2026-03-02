from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.llm_gateway import get_access_token, get_embeddings
from app.services.vector_store import fetch_top_k

router = APIRouter()


@router.get("/retrieve-chunks")
async def retrieve_chunks_get(query: str):
    try:
        access_token = get_access_token()
        query_embedding = get_embeddings([query], access_token)[0]["embedding"]
        chunks = fetch_top_k(query_embedding, top_k=5)
        return {"results": [{"chunk_text": chunk_text} for chunk_text, _ in chunks]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# NOTE: POST /retrieve-chunks is intentionally disabled.
# We keep the handler here for reference/backward compatibility, but it is not registered
# in FastAPI (and therefore won't show up in the OpenAPI docs).
#
# from fastapi import Request
# from json import JSONDecodeError
#
# @router.post("/retrieve-chunks")
# async def retrieve_chunks_post(request: Request):
#     try:
#         try:
#             body = await request.json()
#         except JSONDecodeError:
#             return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
#
#         query = body.get("query")
#         if not query:
#             return JSONResponse(status_code=400, content={"error": "Missing query"})
#
#         access_token = get_access_token()
#         query_embedding = get_embeddings([query], access_token)[0]["embedding"]
#         chunks = fetch_top_k(query_embedding, top_k=5)
#         return {"results": [{"chunk_text": chunk_text} for chunk_text, _ in chunks]}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
