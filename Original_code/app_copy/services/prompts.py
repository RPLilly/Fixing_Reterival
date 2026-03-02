from __future__ import annotations
 
import logging
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from app.db.pg_pool import get_pg_connection, put_pg_connection

logger = logging.getLogger(__name__) 
# Database-backed prompt management
# Each prompt has a unique ID and can be user-specific


def get_prompt_by_id(prompt_id: int) -> dict | None:
    """Get a prompt by its ID."""
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM prompts WHERE id = %s", (prompt_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    except psycopg2.Error as e:
        logger.error(f"Database error retrieving prompt {prompt_id}: {type(e).__name__}: {str(e)}")
        raise
    finally:
        put_pg_connection(conn)


def get_active_prompt_by_type(prompt_type: str, user_id: str | None = None) -> dict | None:
    """
    Get the most recent prompt of a given type for a user.
    Falls back to system default if user-specific not found.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Try user-specific first
            if user_id:
                cur.execute(
                    """SELECT * FROM prompts 
                       WHERE type = %s AND user_id = %s
                       ORDER BY updated_at DESC LIMIT 1""",
                    (prompt_type, user_id)
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
            
            # Fall back to system default
            cur.execute(
                """SELECT * FROM prompts 
                   WHERE type = %s AND user_id = 'system'
                   ORDER BY updated_at DESC LIMIT 1""",
                (prompt_type,)
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except psycopg2.Error as e:
        logger.error(f"Database error retrieving active prompt type '{prompt_type}': {type(e).__name__}: {str(e)}")
        raise
    finally:
        put_pg_connection(conn)


def list_prompts(
    prompt_type: str | None = None,
    user_id: str | None = None,
    limit: int = 100,
    offset: int = 0
) -> tuple[list[dict], int]:
    """List prompts with optional filters. Returns (prompts, total_count)."""
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Build query
            conditions = []
            params = []
            
            if prompt_type:
                conditions.append("type = %s")
                params.append(prompt_type)
            if user_id:
                conditions.append("user_id = %s")
                params.append(user_id)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Get total count
            cur.execute(f"SELECT COUNT(*) FROM prompts WHERE {where_clause}", params)
            total = cur.fetchone()['count']
            
            # Get paginated results
            cur.execute(
                f"""SELECT * FROM prompts WHERE {where_clause} 
                    ORDER BY updated_at DESC LIMIT %s OFFSET %s""",
                params + [limit, offset]
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows], total
    finally:
        put_pg_connection(conn)


def create_prompt(
    prompt_type: str,
    template: str,
    user_id: str
) -> dict:
    """Create a new prompt."""
    if prompt_type not in ('generator', 'validator'):
        raise ValueError("type must be 'generator' or 'validator'")
    
    if not template or not template.strip():
        raise ValueError("template must be a non-empty string")
    
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO prompts (type, template, user_id, created_at, updated_at)
                   VALUES (%s, %s, %s, NOW(), NOW())
                   RETURNING *""",
                (prompt_type, template, user_id)
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row)
    finally:
        put_pg_connection(conn)


def update_prompt(
    prompt_id: int,
    template: str | None = None
) -> dict | None:
    """Update an existing prompt."""
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            updates = []
            params = []
            
            if template is not None:
                if not template.strip():
                    raise ValueError("template must be a non-empty string")
                updates.append("template = %s")
                params.append(template)
            
            if not updates:
                # No updates, just return current
                return get_prompt_by_id(prompt_id)
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(prompt_id)
            
            cur.execute(
                f"""UPDATE prompts SET {', '.join(updates)}
                    WHERE id = %s
                    RETURNING *""",
                params
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row) if row else None
    finally:
        put_pg_connection(conn)


def delete_prompt(prompt_id: int) -> bool:
    """Delete a prompt by ID."""
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prompts WHERE id = %s", (prompt_id,))
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted
    finally:
        put_pg_connection(conn)

