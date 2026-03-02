from __future__ import annotations

import os

import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

from app.core.config import config


POSTGRES_MINCONN = int(os.getenv("POSTGRES_MINCONN", "1"))
POSTGRES_MAXCONN = int(os.getenv("POSTGRES_MAXCONN", "10"))
_pg_pool: pool.SimpleConnectionPool | None = None


def get_pool() -> pool.SimpleConnectionPool:
    global _pg_pool
    if _pg_pool is None:
        conn_info = {
            "dbname": config.postgres_db,
            "user": config.postgres_user,
            "password": config.postgres_password,
            "host": config.postgres_host,
            "port": config.postgres_port,
        }
        if POSTGRES_MAXCONN < POSTGRES_MINCONN:
            raise RuntimeError(
                f"Invalid pool sizing: POSTGRES_MAXCONN={POSTGRES_MAXCONN} must be >= POSTGRES_MINCONN={POSTGRES_MINCONN}"
            )
        _pg_pool = psycopg2.pool.SimpleConnectionPool(POSTGRES_MINCONN, POSTGRES_MAXCONN, **conn_info)
    return _pg_pool


def get_pg_connection():
    conn = get_pool().getconn()
    register_vector(conn)
    return conn


def put_pg_connection(conn):
    get_pool().putconn(conn)
