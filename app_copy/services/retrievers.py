from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.services.llm_gateway import get_access_token, get_embeddings
from app.services.vector_store import fetch_top_k


class PostgresEmbeddingRetriever(BaseRetriever):
    """LangChain retriever backed by Postgres pgvector table chunk_embeddings."""

    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        access_token = get_access_token()
        query_embedding = get_embeddings([query], access_token)[0]["embedding"]
        chunks = fetch_top_k(query_embedding, top_k=self.top_k)
        return [Document(page_content=chunk_text) for chunk_text, _ in chunks]

