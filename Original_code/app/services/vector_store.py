from __future__ import annotations

from typing import Any, Sequence

from app.db.pg_pool import get_pg_connection, put_pg_connection


def insert_embedding(chunk_text: str, embedding: Sequence[float] | Any, filename: str | None = None) -> None:
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chunk_embeddings (chunk_text, embedding, filename) VALUES (%s, %s, %s)",
        (chunk_text, embedding, filename),
    )
    conn.commit()
    cursor.close()
    put_pg_connection(conn)


def embedding_exists(*, chunk_text: str, filename: str | None = None) -> bool:
    """Best-effort dedupe check.

    Since we can't change schema (no unique constraint), we do a simple existence query.
    """
    conn = get_pg_connection()
    cursor = conn.cursor()
    try:
        if filename is None:
            cursor.execute(
                "SELECT 1 FROM chunk_embeddings WHERE chunk_text = %s LIMIT 1",
                (chunk_text,),
            )
        else:
            cursor.execute(
                "SELECT 1 FROM chunk_embeddings WHERE chunk_text = %s AND filename = %s LIMIT 1",
                (chunk_text, filename),
            )
        return cursor.fetchone() is not None
    finally:
        cursor.close()
        put_pg_connection(conn)


def list_chunks(*, filename: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    """Return chunk rows (without embedding vectors) with pagination.

    We intentionally don't return the embedding vector to keep payloads small.
    """
    # Defensive bounds
    if limit < 1:
        limit = 1
    if limit > 1000:
        limit = 1000
    if offset < 0:
        offset = 0

    conn = get_pg_connection()
    cursor = conn.cursor()
    try:
        if filename:
            cursor.execute(
                """
                SELECT id, chunk_text, filename
                FROM chunk_embeddings
                WHERE filename = %s
                ORDER BY id ASC
                LIMIT %s OFFSET %s
                """,
                (filename, limit, offset),
            )
        else:
            cursor.execute(
                """
                SELECT id, chunk_text, filename
                FROM chunk_embeddings
                ORDER BY id ASC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
        rows = cursor.fetchall()
        return [{"id": row[0], "text": row[1], "filename": row[2]} for row in rows]
    finally:
        cursor.close()
        put_pg_connection(conn)


def fetch_top_k(query_embedding: Sequence[float], *, top_k: int = 5, filename: str | None = None):
    conn = get_pg_connection()
    cursor = conn.cursor()
    embedding_str = '[' + ','.join(str(value) for value in query_embedding) + ']'

    if filename:
        cursor.execute(
            """
            SELECT chunk_text, embedding <#> %s::vector AS distance
            FROM chunk_embeddings
            WHERE filename = %s
            ORDER BY distance ASC
            LIMIT %s
            """,
            (embedding_str, filename, top_k),
        )
    else:
        cursor.execute(
            """
            SELECT chunk_text, embedding <#> %s::vector AS distance
            FROM chunk_embeddings
            ORDER BY distance ASC
            LIMIT %s
            """,
            (embedding_str, top_k),
        )

    rows = cursor.fetchall()
    cursor.close()
    put_pg_connection(conn)
    return [(row[0], row[1]) for row in rows]


def clear_chunk_embeddings(*, filename: str | None = None, reset_identity: bool = True) -> int:
    """Clear embeddings from Postgres.

    - If filename is provided, deletes only rows for that filename.
    - Otherwise clears the whole table (TRUNCATE).

    Returns number of rows deleted (best-effort; TRUNCATE returns 0).
    """
    conn = get_pg_connection()
    cursor = conn.cursor()
    try:
        if filename:
            cursor.execute("DELETE FROM chunk_embeddings WHERE filename = %s", (filename,))
            deleted = cursor.rowcount
        else:
            if reset_identity:
                cursor.execute("TRUNCATE TABLE chunk_embeddings RESTART IDENTITY")
            else:
                cursor.execute("TRUNCATE TABLE chunk_embeddings")
            deleted = 0
        conn.commit()
        return int(deleted)
    finally:
        cursor.close()
        put_pg_connection(conn)


def count_chunk_embeddings(*, filename: str | None = None) -> int:
    """Count rows in chunk_embeddings (optionally filtered by filename)."""
    conn = get_pg_connection()
    cursor = conn.cursor()
    try:
        if filename:
            cursor.execute("SELECT COUNT(*) FROM chunk_embeddings WHERE filename = %s", (filename,))
        else:
            cursor.execute("SELECT COUNT(*) FROM chunk_embeddings")
        return int(cursor.fetchone()[0])
    finally:
        cursor.close()
        put_pg_connection(conn)
