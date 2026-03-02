"""
Comprehensive testing script for retrieval pipeline.
Tests: Vector Search, Full-Text Search, and RRF (Reciprocal Rank Fusion)
Metrics: Relevance ranking, Hit rate, and Mean Reciprocal Rank (MRR)
"""

from __future__ import annotations
import json
import os
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import logging

# Force DB_HOST to localhost for local development
os.environ['DB_HOST'] = 'localhost'

# Clear all app-related modules to force reimport
modules_to_clear = [key for key in sys.modules.keys() if key.startswith('app')]
for module in modules_to_clear:
    del sys.modules[module]

from sqlalchemy import Column, Integer, Text, func
from sqlalchemy.orm import declarative_base
from app.db.sqlalchemy import get_session_local
from app.services.llm_gateway import get_access_token, get_embeddings
from app.services.vector_store import fetch_top_k

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()
SessionLocal = get_session_local()


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    id = Column(Integer, primary_key=True)
    text = Column("chunk_text", Text, nullable=True)
    filename = Column(Text, nullable=True)


@dataclass
class RetrievalMetrics:
    """Metrics for a single test case"""
    question: str
    method: str  # "vector", "lexical", "rrf"
    answer_chunk_ids: List[int]
    retrieved_chunk_ids: List[int]
    hit_count: int  # Number of correct chunks in top results
    recall: float  # Recall: hit_count / total expected chunks
    avg_rr: float  # Averaged Reciprocal Rank for all hits
    normalized_avg_rr: float  # Normalized Avg RR (1.0 if all expected chunks found in first K positions)
    top_k: int = 5


def _sanitize_tsquery(user_query: str) -> str:
    """Sanitize user query for PostgreSQL full-text search"""
    terms = [query.strip() for query in user_query.replace("\n", " ").split() if query.strip()]
    escaped = []
    for term in terms:
        safe = "".join(character for character in term if character.isalnum() or character in {"_", "-"})
        if safe:
            escaped.append(f"{safe}:*")
    return " & ".join(escaped) if escaped else ""


def _full_text_search(*, user_query: str, limit: int, file_filter: str | None = None) -> List[dict]:
    """Perform full-text search using PostgreSQL"""
    textsearch_query = _sanitize_tsquery(user_query)
    if not textsearch_query:
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
        return out
    except Exception as e:
        logger.error(f"Full-text search error: {e}")
        return []
    finally:
        db.close()


def _vector_search(query_embedding: list, limit: int, file_filter: str | None = None) -> List[Tuple[str, float]]:
    """Perform vector search using embeddings"""
    try:
        results = fetch_top_k(query_embedding, top_k=limit, filename=file_filter)
        return results
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []


def _rrf_score(*, rank: int | None, k: int) -> float:
    """Reciprocal Rank Fusion score"""
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(k + rank)


def _calculate_normalized_avg_rr(reciprocal_ranks: List[float], answer_chunk_ids: List[int]) -> float:
    """
    Calculate normalized average reciprocal rank.
    
    If all expected chunks are found:
        - Ideal case: all at positions 1,2,3,... gives NARR = 1.0
        - Actual RR divided by ideal RR
    If not all chunks found:
        - Penalized by dividing by number of expected chunks
    
    NARR ranges from 0 to 1.0, where 1.0 = perfect ranking
    """
    if not answer_chunk_ids:
        return 0.0
    
    num_expected = len(answer_chunk_ids)
    
    if not reciprocal_ranks:
        # No chunks found
        return 0.0
    
    # Sum of actual reciprocal ranks
    actual_sum = sum(reciprocal_ranks)
    
    # Ideal sum: if all expected chunks were at positions 1,2,3,...,num_expected
    # Ideal RR values would be [1/1, 1/2, 1/3, ..., 1/num_expected]
    ideal_sum = sum(1.0 / (i + 1) for i in range(num_expected))
    
    # Normalized: actual / ideal, penalized by number of hits vs expected
    num_hits = len(reciprocal_ranks)
    normalized_avg_rr = (actual_sum / ideal_sum) * (num_hits / num_expected)
    
    return min(normalized_avg_rr, 1.0)  # Cap at 1.0


def _get_chunk_id_from_text(chunk_text: str) -> int | None:
    """Retrieve chunk ID from database by chunk text"""
    db = SessionLocal()
    try:
        result = db.query(ChunkEmbedding.id).filter(ChunkEmbedding.text == chunk_text).first()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error retrieving chunk ID: {e}")
        return None
    finally:
        db.close()


def test_vector_search(query: str, answer_chunk_ids: List[int], limit: int = 5) -> RetrievalMetrics:
    """Test vector search and calculate metrics"""
    try:
        access_token = get_access_token()
        query_embedding = get_embeddings([query], access_token)[0]["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        return RetrievalMetrics(query, "vector", answer_chunk_ids, [], 0, 0.0, 0.0, 0.0, limit)

    vector_results = _vector_search(query_embedding, limit)
    
    retrieved_chunk_ids = []
    hit_count = 0
    reciprocal_ranks = []

    for rank, (chunk_text, _distance) in enumerate(vector_results, start=1):
        chunk_id = _get_chunk_id_from_text(chunk_text)
        if chunk_id:
            retrieved_chunk_ids.append(chunk_id)
            if chunk_id in answer_chunk_ids:
                hit_count += 1
                reciprocal_ranks.append(1.0 / rank)

    # Calculate Avg RR penalizing missing chunks: average across ALL expected chunks
    avg_rr = sum(reciprocal_ranks) / len(answer_chunk_ids) if answer_chunk_ids else 0.0
    recall = hit_count / len(answer_chunk_ids) if answer_chunk_ids else 0.0
    normalized_avg_rr = _calculate_normalized_avg_rr(reciprocal_ranks, answer_chunk_ids)
    return RetrievalMetrics(query, "vector", answer_chunk_ids, retrieved_chunk_ids, hit_count, recall, avg_rr, normalized_avg_rr, limit)


def test_lexical_search(query: str, answer_chunk_ids: List[int], limit: int = 5) -> RetrievalMetrics:
    """Test full-text search and calculate metrics"""
    lexical_results = _full_text_search(user_query=query, limit=limit)
    
    retrieved_chunk_ids = []
    hit_count = 0
    reciprocal_ranks = []

    for rank, result in enumerate(lexical_results, start=1):
        chunk_id = result.get("id")
        retrieved_chunk_ids.append(chunk_id)
        if chunk_id in answer_chunk_ids:
            hit_count += 1
            reciprocal_ranks.append(1.0 / rank)

    # Calculate Avg RR penalizing missing chunks: average across ALL expected chunks
    avg_rr = sum(reciprocal_ranks) / len(answer_chunk_ids) if answer_chunk_ids else 0.0
    recall = hit_count / len(answer_chunk_ids) if answer_chunk_ids else 0.0
    normalized_avg_rr = _calculate_normalized_avg_rr(reciprocal_ranks, answer_chunk_ids)
    return RetrievalMetrics(query, "lexical", answer_chunk_ids, retrieved_chunk_ids, hit_count, recall, avg_rr, normalized_avg_rr, limit)


def test_rrf_search(query: str, answer_chunk_ids: List[int], limit: int = 5, alpha: float = 0.6, rrf_k: int = 60) -> RetrievalMetrics:
    """Test RRF (combined vector + lexical) search and calculate metrics"""
    try:
        access_token = get_access_token()
        query_embedding = get_embeddings([query], access_token)[0]["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        return RetrievalMetrics(query, "rrf", answer_chunk_ids, [], 0, 0.0, limit)

    candidate_k = max(limit * 3, 10)
    
    # Get vector candidates
    vector_candidates = _vector_search(query_embedding, candidate_k)
    vector_rank: dict[str, int] = {}
    for i, (chunk_text, _dist) in enumerate(vector_candidates, start=1):
        if chunk_text:
            vector_rank.setdefault(chunk_text, i)

    # Get lexical candidates
    lexical_results = _full_text_search(user_query=query, limit=candidate_k)
    lexical_rank: dict[str, int] = {}
    for i, result in enumerate(lexical_results, start=1):
        chunk_text = result.get("chunk_text")
        if chunk_text:
            lexical_rank.setdefault(chunk_text, i)

    # Merge and score
    merged: dict[str, dict] = {}
    
    for chunk_text, rank in vector_rank.items():
        merged.setdefault(chunk_text, {"chunk_text": chunk_text})
        merged[chunk_text]["vector_rank"] = rank

    for chunk_text, rank in lexical_rank.items():
        merged.setdefault(chunk_text, {"chunk_text": chunk_text})
        merged[chunk_text]["lexical_rank"] = rank

    # Calculate RRF scores
    items = list(merged.values())
    for item in items:
        v_rank = item.get("vector_rank")
        l_rank = item.get("lexical_rank")
        item["rrf_score"] = (float(alpha) * _rrf_score(rank=v_rank, k=int(rrf_k))) + (
            (1.0 - float(alpha)) * _rrf_score(rank=l_rank, k=int(rrf_k))
        )

    items.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    items = items[:limit]

    # Get chunk IDs from database
    retrieved_chunk_ids = []
    hit_count = 0
    reciprocal_ranks = []

    for rank, item in enumerate(items, start=1):
        chunk_text = item.get("chunk_text")
        chunk_id = _get_chunk_id_from_text(chunk_text)
        if chunk_id:
            retrieved_chunk_ids.append(chunk_id)
            if chunk_id in answer_chunk_ids:
                hit_count += 1
                reciprocal_ranks.append(1.0 / rank)

    # Calculate Avg RR penalizing missing chunks: average across ALL expected chunks
    avg_rr = sum(reciprocal_ranks) / len(answer_chunk_ids) if answer_chunk_ids else 0.0
    recall = hit_count / len(answer_chunk_ids) if answer_chunk_ids else 0.0
    normalized_avg_rr = _calculate_normalized_avg_rr(reciprocal_ranks, answer_chunk_ids)
    return RetrievalMetrics(query, "rrf", answer_chunk_ids, retrieved_chunk_ids, hit_count, recall, avg_rr, normalized_avg_rr, limit)


def print_metrics_table(metrics_list: List[RetrievalMetrics]):
    """Print metrics in a formatted table"""
    print("\n" + "=" * 210)
    print(f"{'Question':<45} | {'Method':<10} | {'Expected IDs':<35} | {'Retrieved IDs':<35} | {'Hits':<6} | {'Recall':<8} | {'Norm RR':<8}")
    print("=" * 210)
    
    for metric in metrics_list:
        question_short = metric.question[:42] + "..." if len(metric.question) > 45 else metric.question
        expected_ids_str = str(metric.answer_chunk_ids)
        retrieved_ids_str = str(metric.retrieved_chunk_ids)
        print(f"{question_short:<45} | {metric.method:<10} | {expected_ids_str:<35} | {retrieved_ids_str:<35} | {metric.hit_count:<6} | {metric.recall:<8.4f} | {metric.normalized_avg_rr:<8.4f}")
    
    print("=" * 210)


def print_summary_stats(metrics_list: List[RetrievalMetrics]):
    """Print summary statistics for each method"""
    methods = {}
    
    for metric in metrics_list:
        method = metric.method
        if method not in methods:
            methods[method] = {"recalls": [], "normalized_avg_rrs": []}
        
        methods[method]["recalls"].append(metric.recall)
        methods[method]["normalized_avg_rrs"].append(metric.normalized_avg_rr)

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for method, data in sorted(methods.items()):
        avg_recall = sum(data["recalls"]) / len(data["recalls"]) if data["recalls"] else 0
        avg_norm_rr = sum(data["normalized_avg_rrs"]) / len(data["normalized_avg_rrs"]) if data["normalized_avg_rrs"] else 0
        
        print(f"\n{method.upper()}:")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average Norm RR: {avg_norm_rr:.4f}")
    
    print("\n" + "=" * 70)


def main():
    """Main testing function"""
    questions_file = "questions_with_chunks.json"
    
    if not os.path.exists(questions_file):
        logger.error(f"File {questions_file} not found")
        return

    # Load questions
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    questions = data.get("questions", [])
    logger.info(f"Loaded {len(questions)} questions")

    all_metrics: List[RetrievalMetrics] = []
    limit = 5
    
    for idx, q_data in enumerate(questions, start=1):
        question = q_data.get("question", "")
        answer_chunk_ids = q_data.get("answer_chunk_ids", [])
        
        if not question or not answer_chunk_ids:
            logger.warning(f"Skipping question {idx}: missing question or answer_chunk_ids")
            continue
        
        logger.info(f"\n--- Question {idx}/{len(questions)} ---")
        logger.info(f"Q: {question[:60]}...")
        logger.info(f"Expected chunk IDs: {answer_chunk_ids}")
        
        # Test all three methods
        try:
            vector_metrics = test_vector_search(question, answer_chunk_ids, limit)
            all_metrics.append(vector_metrics)
            logger.info(f"  Vector: {vector_metrics.hit_count}/{len(answer_chunk_ids)} hits, Recall={vector_metrics.recall:.4f}, Avg RR={vector_metrics.avg_rr:.4f}")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        try:
            lexical_metrics = test_lexical_search(question, answer_chunk_ids, limit)
            all_metrics.append(lexical_metrics)
            logger.info(f"  Lexical: {lexical_metrics.hit_count}/{len(answer_chunk_ids)} hits, Recall={lexical_metrics.recall:.4f}, Avg RR={lexical_metrics.avg_rr:.4f}")
        except Exception as e:
            logger.error(f"Lexical search failed: {e}")
        
        try:
            rrf_metrics = test_rrf_search(question, answer_chunk_ids, limit)
            all_metrics.append(rrf_metrics)
            logger.info(f"  RRF: {rrf_metrics.hit_count}/{len(answer_chunk_ids)} hits, Recall={rrf_metrics.recall:.4f}, Avg RR={rrf_metrics.avg_rr:.4f}")
        except Exception as e:
            logger.error(f"RRF search failed: {e}")

    # Print results
    print_metrics_table(all_metrics)
    print_summary_stats(all_metrics)

    # Save results to JSON
    results_file = "retrieval_test_results.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
