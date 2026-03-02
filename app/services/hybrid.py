from __future__ import annotations
import logging
from typing import List

from sqlalchemy import Column, Integer, Text, func # type: ignore
from sqlalchemy.orm import declarative_base # type: ignore

from app.db.sqlalchemy import get_session_local
from app.services.llm_gateway import get_embeddings_v2
from app.services.vector_store import fetch_top_k
 
logger = logging.getLogger(__name__)
 
Base = declarative_base()
SessionLocal = get_session_local()


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    id = Column(Integer, primary_key=True)
    text = Column("chunk_text", Text, nullable=True)
    filename = Column(Text, nullable=True)


def _sanitize_tsquery(user_query: str) -> str:
    terms = [query.strip() for query in user_query.replace("\n", " ").split() if query.strip()]
    escaped = []
    for term in terms:
        safe = "".join(character for character in term if character.isalnum() or character in {"_", "-"})
        if safe:
            escaped.append(f"{safe}:*")
    return " & ".join(escaped) if escaped else ""


def _full_text_search(*, user_query: str, limit: int, file_filter: str | None = None) -> List[dict]:
    textsearch_query = _sanitize_tsquery(user_query)
    if not textsearch_query:
        logger.warning(f"Full-text search failed: invalid query after sanitization. Original query: '{user_query}'")
        return []

    db = SessionLocal()
    try:
        query = db.query(
            ChunkEmbedding,
            func.ts_rank_cd(
                func.to_tsvector("english", ChunkEmbedding.text),
                func.to_tsquery("english", textsearch_query),
            ).label("tsv_score"),
        ).filter(
            func.to_tsvector("english", ChunkEmbedding.text).op("@@")(
                func.to_tsquery("english", textsearch_query)
            )
        )

        if file_filter:
            query = query.filter(ChunkEmbedding.filename == file_filter)


        results = (
            query.order_by(
                func.ts_rank_cd(
                    func.to_tsvector("english", ChunkEmbedding.text),
                    func.to_tsquery("english", textsearch_query),
                ).desc()
            )
            .limit(limit)
            .all()
        )

        out: List[dict] = []
        for emb_row, tsv_score in results:
            out.append(
                {
                    "id": int(emb_row.id),
                    "chunk_text": emb_row.text,
                    "filename": emb_row.filename,
                    "lexical_score": float(tsv_score) if tsv_score is not None else 0.0,
                }
            )
        logger.info(f"Full-text search successful: retrieved {len(out)} chunks. Query: '{user_query}', File filter: {file_filter}")
        return out
    except Exception as e:
        logger.error(f"Full-text search failed: {type(e).__name__}: {str(e)}. Query: '{user_query}', File filter: {file_filter}", exc_info=True)
        return []
    finally:
        db.close()


def _rrf_score(*, rank: int | None, k: int) -> float:
    """Reciprocal Rank Fusion contribution for a single list.

    rank is 1-based (1 is best). If rank is None (doc not in list) -> 0.
    """

    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(k + rank)


def hybrid_retrieve(
    *,
    query: str,
    limit: int = 5,
    alpha: float = 0.6,
    file_filter: str | None = None,
    mode: str = "rrf",
    rrf_k: int = 60,
) -> dict:
    logger.info(f"Starting hybrid retrieval")
    try:
        embeddings_client = get_embeddings_v2()
        query_embedding = embeddings_client.embed_query(query)
        logger.info(f"Embedding generation successful OpenAIEmbeddings client.")
    except Exception as e:
        logger.error(f"Hybrid retrieval failed during embedding generation: {type(e).__name__}: {str(e)}", exc_info=True)
        raise
 
    candidate_k = max(limit * 3, 10)
    try:
        vector_candidates = fetch_top_k(query_embedding, top_k=candidate_k, filename=file_filter)
        logger.info(f"Vector search successful: retrieved {len(vector_candidates)} candidates")
    except Exception as e:
        logger.error(f"Hybrid retrieval failed during vector search: {type(e).__name__}: {str(e)}", exc_info=True)
        raise
   
    tsv_rows = _full_text_search(user_query=query, limit=candidate_k, file_filter=file_filter)  ##text search results

    mode = (mode or "blend").lower().strip()
    if mode not in {"blend", "rrf"}:
        logger.error(f"Invalid mode: {mode}. Must be one of: blend, rrf")
        raise ValueError("mode must be one of: blend, rrf")
    if rrf_k <= 0:
        logger.error(f"Invalid rrf_k: {rrf_k}. Must be > 0")
        raise ValueError("rrf_k must be > 0")

    if mode == "rrf":
        # Rank maps (1-based)
        vector_rank: dict[str, int] = {}
        for i, (chunk_text, _dist) in enumerate(vector_candidates, start=1):
            if not chunk_text:
                continue
            vector_rank.setdefault(chunk_text, i)

        lexical_rank: dict[str, int] = {}
        filename_map: dict[str, str] = {}

        for i, row in enumerate(tsv_rows, start=1):
            chunk_text = row.get("chunk_text")
            if not chunk_text:
                continue
            lexical_rank.setdefault(chunk_text, i)
            if row.get("filename"):
                filename_map.setdefault(chunk_text, row.get("filename"))

        # Union of candidates
        merged: dict[str, dict] = {}
        for chunk_text_vector, rank_vector in vector_rank.items():
            merged.setdefault(chunk_text_vector, {"chunk_text": chunk_text_vector, "filename": None})
            merged[chunk_text_vector]["vector_rank"] = rank_vector

        for chunk_text_lexical, rank_lexical in lexical_rank.items():
            merged.setdefault(chunk_text_lexical, {"chunk_text": chunk_text_lexical, "filename": None})
            merged[chunk_text_lexical]["lexical_rank"] = rank_lexical
            if chunk_text_lexical in filename_map and not merged[chunk_text_lexical].get("filename"):
                merged[chunk_text_lexical]["filename"] = filename_map[chunk_text_lexical]

        items = list(merged.values())
        for item in items:
            vector_rank = item.get("vector_rank")
            lexical_rank = item.get("lexical_rank")
            item["rrf_score"] = (float(alpha) * _rrf_score(rank=vector_rank, k=int(rrf_k))) + (
                (1.0 - float(alpha)) * _rrf_score(rank=lexical_rank, k=int(rrf_k))
            )

        items.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        items = items[:limit]

        # Tighten filename consistency for top results.
        top_texts = [item.get("chunk_text") for item in items if item.get("chunk_text")]
        if top_texts:
            db = SessionLocal()
            try:
                query = db.query(ChunkEmbedding.text, ChunkEmbedding.filename).filter(ChunkEmbedding.text.in_(top_texts))
                if file_filter:
                    query = query.filter(ChunkEmbedding.filename == file_filter)
                filename_by_text = {chunk_text: fn for chunk_text, fn in query.all() if chunk_text}
                for item in items:
                    item["filename"] = filename_by_text.get(item.get("chunk_text"))
            finally:
                db.close()
 
        logger.info(f"Hybrid retrieval (RRF mode) successful: returned {len(items)} results")
        return {
            "results": items,
            "params": {
                "limit": limit,
                "alpha": alpha,
                "file_filter": file_filter,
                "fusion_mode": "rrf",
                "rrf_k": int(rrf_k),
            },
        }

    # --- default: score blending (existing behavior) ---

    vector_distances = [float(distance) for _, distance in vector_candidates] or [1.0]
    vector_dist_min, vector_dist_max = min(vector_distances), max(vector_distances)
    vector_dist_denom = (vector_dist_max - vector_dist_min) if (vector_dist_max - vector_dist_min) != 0 else 1.0

    def vec_score(vec_distance: float) -> float:
        return 1.0 - ((vec_distance - vector_dist_min) /vector_dist_denom)
    vector_map: dict[str, float] = {}
    for chunk_text, dist in vector_candidates:
        if not chunk_text:
            continue
        vector_map[chunk_text] = max(vector_map.get(chunk_text, 0.0), vec_score(float(dist)))

    lexical_scores = [float(row["lexical_score"]) for row in tsv_rows] or [0.0]
    lexical_score_min, lexical_score_max = min(lexical_scores), max(lexical_scores)
    lexical_score_denom = (lexical_score_max - lexical_score_min) if (lexical_score_max - lexical_score_min) != 0 else 1.0

    def lexical_score_norm(s: float) -> float:
        if lexical_score_max == lexical_score_min:
            return 1.0 if s > 0 else 0.0
        return (s - lexical_score_min) / lexical_score_denom

    lexical_map: dict[str, float] = {}
    filename_map: dict[str, str] = {}
    for tsv_row in tsv_rows:
        tsv_chunk_text = tsv_row.get("chunk_text")  ##text search chunk text
        if not tsv_chunk_text:
            continue
        lexical_map[tsv_chunk_text] = max(
            lexical_map.get(tsv_chunk_text, 0.0),
            lexical_score_norm(float(tsv_row.get("lexical_score", 0.0))),
        )
        if tsv_row.get("filename"):
            filename_map.setdefault(tsv_chunk_text, tsv_row.get("filename"))

    merged: dict[str, dict] = {}
    for chunk_text, vector_score in vector_map.items():
        merged.setdefault(chunk_text, {"chunk_text": chunk_text, "vector_score": 0.0, "lexical_score": 0.0, "filename": None})
        merged[chunk_text]["vector_score"] = vector_score

    for chunk_text, lexical_score in lexical_map.items():
        merged.setdefault(chunk_text, {"chunk_text": chunk_text, "vector_score": 0.0, "lexical_score": 0.0, "filename": None})
        merged[chunk_text]["lexical_score"] = lexical_score
        if chunk_text in filename_map and not merged[chunk_text].get("filename"):
            merged[chunk_text]["filename"] = filename_map[chunk_text]

    items = list(merged.values())
    for item in items:
        item["hybrid_score"] = alpha * float(item["vector_score"]) + (1.0 - alpha) * float(item["lexical_score"])

    items.sort(key=lambda x: x["hybrid_score"], reverse=True)
    items = items[:limit]

    # Tighten filename consistency for top results.
    top_texts = [item.get("chunk_text") for item in items if item.get("chunk_text")]
    if top_texts:
        db = SessionLocal()
        try:
            query = db.query(ChunkEmbedding.text, ChunkEmbedding.filename).filter(ChunkEmbedding.text.in_(top_texts))
            if file_filter:
                query = query.filter(ChunkEmbedding.filename == file_filter)
            filename_by_text = {chunk_text: fn for chunk_text, fn in query.all() if chunk_text}
            for item in items:
                item["filename"] = filename_by_text.get(item.get("chunk_text"))
        except Exception:
            pass
        finally:
            db.close()
 
    logger.info(f"Hybrid retrieval (blend mode) successful: returned {len(items)} results")
    return {
        "results": items,
        "params": {"limit": limit, "alpha": alpha, "file_filter": file_filter, "fusion_mode": "blend"},
    }