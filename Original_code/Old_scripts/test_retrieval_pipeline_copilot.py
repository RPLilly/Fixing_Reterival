"""
Comprehensive testing script for retrieval pipeline.
Tests: Vector Search, Full-Text Search, and RRF (Reciprocal Rank Fusion)
Metrics: Relevance ranking, Hit rate, and Mean Reciprocal Rank (MRR)
"""

from __future__ import annotations
import json
import os
import sys
import time
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

# Common English stopwords to filter out for better relevance
COMMON_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
    'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
}

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
    retrieval_time: float = 0.0  # Time taken for retrieval in seconds
    top_k: int = 5


def _sanitize_tsquery(user_query: str, use_and: bool = True) -> str:
    """
    Sanitize user query for PostgreSQL full-text search.
    
    Args:
        user_query: The raw user query
        use_and: If True, use AND operator (all terms must match). 
                 If False, use OR operator (any term matches).
    
    Returns:
        Sanitized PostgreSQL tsquery string
    """
    # Normalize whitespace
    normalized = user_query.replace("\n", " ").lower().strip()
    
    # Split into terms
    raw_terms = [term.strip() for term in normalized.split() if term.strip()]
    
    # Clean and filter terms
    valid_terms = []
    for term in raw_terms:
        # Remove special characters, keep only alphanumeric, underscore, hyphen
        safe = "".join(char for char in term if char.isalnum() or char in {"_", "-"})
        
        # Filter out stopwords and very short terms (noise)
        if safe and len(safe) > 2 and safe not in COMMON_STOPWORDS:
            # Weight longer terms higher with prefix matching
            valid_terms.append(f"{safe}:*")
        elif safe and len(safe) > 2:  # Include slightly longer stopwords
            # For longer terms, even if stopword, include them
            if len(safe) >= 5:
                valid_terms.append(f"{safe}:*")
    
    if not valid_terms:
        # Fallback: if all terms were filtered, use original terms without prefix matching
        for term in raw_terms:
            safe = "".join(char for char in term if char.isalnum() or char in {"_", "-"})
            if safe and len(safe) > 1:
                valid_terms.append(safe)
    
    # Combine terms with AND operator (all must match) for better precision
    operator = " & " if use_and else " | "
    return operator.join(valid_terms) if valid_terms else ""


def _full_text_search(*, user_query: str, limit: int, file_filter: str | None = None) -> List[dict]:
    """
    Perform full-text search using PostgreSQL with improved relevance.
    
    Strategy:
    1. First try AND operator (all terms must match) for precision
    2. If no results, fall back to OR operator (any term matches) for recall
    3. Use ts_rank_cd with normalization for better scoring
    """
    db = SessionLocal()
    try:
        # First attempt: AND operator for better precision
        textsearch_query = _sanitize_tsquery(user_query, use_and=True)
        if not textsearch_query:
            return []

        query = db.query(
            ChunkEmbedding,
            func.ts_rank_cd(
                func.to_tsvector("english", ChunkEmbedding.text),
                func.to_tsquery("english", textsearch_query),
                # Normalization: 2 = rank / (1 + log(length))
                # Improves ranking for shorter, more focused chunks
                32  # Use normalization flag 32 for length normalization
            ).label("tsv_score"),
        ).filter(
            func.to_tsvector("english", ChunkEmbedding.text).op("@@")(
                func.to_tsquery("english", textsearch_query)
            )
        )

        if file_filter:
            query = query.filter(ChunkEmbedding.filename == file_filter)

        results = query.order_by(
            func.ts_rank_cd(
                func.to_tsvector("english", ChunkEmbedding.text),
                func.to_tsquery("english", textsearch_query),
                32
            ).desc()
        ).limit(limit).all()

        # If we got good results with AND, return them
        if len(results) >= limit * 0.5:  # At least 50% of requested results
            return _format_results(results)

        # Fallback: Try with OR operator for better recall
        logger.info(f"AND query returned few results ({len(results)}), trying OR operator")
        textsearch_query = _sanitize_tsquery(user_query, use_and=False)
        if not textsearch_query:
            return _format_results(results)

        query = db.query(
            ChunkEmbedding,
            func.ts_rank_cd(
                func.to_tsvector("english", ChunkEmbedding.text),
                func.to_tsquery("english", textsearch_query),
                32
            ).label("tsv_score"),
        ).filter(
            func.to_tsvector("english", ChunkEmbedding.text).op("@@")(
                func.to_tsquery("english", textsearch_query)
            )
        )

        if file_filter:
            query = query.filter(ChunkEmbedding.filename == file_filter)

        or_results = query.order_by(
            func.ts_rank_cd(
                func.to_tsvector("english", ChunkEmbedding.text),
                func.to_tsquery("english", textsearch_query),
                32
            ).desc()
        ).limit(limit).all()

        # Combine results, preferring AND results
        combined_results = results + or_results
        return _format_results(combined_results[:limit])

    except Exception as e:
        logger.error(f"Full-text search error: {e}")
        return []
    finally:
        db.close()


def _format_results(results: List[Tuple]) -> List[dict]:
    """Format database results into dictionary list"""
    out: List[dict] = []
    seen_ids = set()
    
    for emb_row, tsv_score in results:
        # Avoid duplicates
        if emb_row.id in seen_ids:
            continue
        seen_ids.add(emb_row.id)
        
        out.append({
            "id": int(emb_row.id),
            "chunk_text": emb_row.text,
            "filename": emb_row.filename,
            "lexical_score": float(tsv_score) if tsv_score is not None else 0.0,
        })
    
    return out


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
    start_time = time.time()
    try:
        access_token = get_access_token()
        query_embedding = get_embeddings([query], access_token)[0]["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        retrieval_time = time.time() - start_time
        return RetrievalMetrics(query, "vector", answer_chunk_ids, [], 0, 0.0, 0.0, 0.0, retrieval_time, limit)

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
    retrieval_time = time.time() - start_time
    return RetrievalMetrics(query, "vector", answer_chunk_ids, retrieved_chunk_ids, hit_count, recall, avg_rr, normalized_avg_rr, retrieval_time, limit)


def test_lexical_search(query: str, answer_chunk_ids: List[int], limit: int = 5) -> RetrievalMetrics:
    """Test full-text search and calculate metrics"""
    start_time = time.time()
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
    retrieval_time = time.time() - start_time
    return RetrievalMetrics(query, "lexical", answer_chunk_ids, retrieved_chunk_ids, hit_count, recall, avg_rr, normalized_avg_rr, retrieval_time, limit)


def test_rrf_search(query: str, answer_chunk_ids: List[int], limit: int = 5, alpha: float = 0.6, rrf_k: int = 60) -> RetrievalMetrics:
    """Test RRF (combined vector + lexical) search and calculate metrics"""
    start_time = time.time()
    try:
        access_token = get_access_token()
        query_embedding = get_embeddings([query], access_token)[0]["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        retrieval_time = time.time() - start_time
        return RetrievalMetrics(query, "rrf", answer_chunk_ids, [], 0, 0.0, 0.0, 0.0, retrieval_time, limit)

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
    retrieval_time = time.time() - start_time
    return RetrievalMetrics(query, "rrf", answer_chunk_ids, retrieved_chunk_ids, hit_count, recall, avg_rr, normalized_avg_rr, retrieval_time, limit)


def print_metrics_table(metrics_list: List[RetrievalMetrics]):
    """Print metrics in a formatted table"""
    print("\n" + "=" * 250)
    print(f"{'Question':<45} | {'Method':<10} | {'Expected IDs':<35} | {'Retrieved IDs':<35} | {'Hits':<6} | {'Recall':<8} | {'Norm RR':<8} | {'Time (s)':<10}")
    print("=" * 250)
    
    for metric in metrics_list:
        question_short = metric.question[:42] + "..." if len(metric.question) > 45 else metric.question
        expected_ids_str = str(metric.answer_chunk_ids)
        retrieved_ids_str = str(metric.retrieved_chunk_ids)
        print(f"{question_short:<45} | {metric.method:<10} | {expected_ids_str:<35} | {retrieved_ids_str:<35} | {metric.hit_count:<6} | {metric.recall:<8.4f} | {metric.normalized_avg_rr:<8.4f} | {metric.retrieval_time:<10.4f}")
    
    print("=" * 250)


def print_summary_stats(metrics_list: List[RetrievalMetrics]):
    """Print summary statistics for each method"""
    methods = {}
    
    for metric in metrics_list:
        method = metric.method
        if method not in methods:
            methods[method] = {"recalls": [], "normalized_avg_rrs": [], "retrieval_times": []}
        
        methods[method]["recalls"].append(metric.recall)
        methods[method]["normalized_avg_rrs"].append(metric.normalized_avg_rr)
        methods[method]["retrieval_times"].append(metric.retrieval_time)

    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)
    
    for method, data in sorted(methods.items()):
        avg_recall = sum(data["recalls"]) / len(data["recalls"]) if data["recalls"] else 0
        avg_norm_rr = sum(data["normalized_avg_rrs"]) / len(data["normalized_avg_rrs"]) if data["normalized_avg_rrs"] else 0
        avg_time = sum(data["retrieval_times"]) / len(data["retrieval_times"]) if data["retrieval_times"] else 0
        
        print(f"\n{method.upper()}:")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average Norm RR: {avg_norm_rr:.4f}")
        print(f"  Average Retrieval Time: {avg_time:.4f}s")
    
    print("\n" + "=" * 90)


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
            logger.info(f"  Vector: {vector_metrics.hit_count}/{len(answer_chunk_ids)} hits, Recall={vector_metrics.recall:.4f}, Avg RR={vector_metrics.avg_rr:.4f}, Time={vector_metrics.retrieval_time:.4f}s")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        try:
            lexical_metrics = test_lexical_search(question, answer_chunk_ids, limit)
            all_metrics.append(lexical_metrics)
            logger.info(f"  Lexical: {lexical_metrics.hit_count}/{len(answer_chunk_ids)} hits, Recall={lexical_metrics.recall:.4f}, Avg RR={lexical_metrics.avg_rr:.4f}, Time={lexical_metrics.retrieval_time:.4f}s")
        except Exception as e:
            logger.error(f"Lexical search failed: {e}")
        
        try:
            rrf_metrics = test_rrf_search(question, answer_chunk_ids, limit)
            all_metrics.append(rrf_metrics)
            logger.info(f"  RRF: {rrf_metrics.hit_count}/{len(answer_chunk_ids)} hits, Recall={rrf_metrics.recall:.4f}, Avg RR={rrf_metrics.avg_rr:.4f}, Time={rrf_metrics.retrieval_time:.4f}s")
        except Exception as e:
            logger.error(f"RRF search failed: {e}")

    # Print results
    print_metrics_table(all_metrics)
    print_summary_stats(all_metrics)

    # Save results to JSON
    results_file = "retrieval_test_results_copilot.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
