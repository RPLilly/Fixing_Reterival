"""
SQLAlchemy ORM Models for Database Tables

This is the SOURCE OF TRUTH for the database schema.
When you modify models here, generate a migration with:
    docker-compose exec app alembic revision --autogenerate -m "description"
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, Integer, Text, String, Boolean, DateTime, CheckConstraint, UniqueConstraint
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

# Base class for all models
Base = declarative_base()


class ChunkEmbedding(Base):
    """
    Table for storing document chunks and their embeddings.
    
    Purpose: Stores text chunks from documents along with their vector embeddings
    for semantic search and retrieval.
    """
    
    __tablename__ = "chunk_embeddings"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    chunk_text = Column(Text, nullable=True)
    embedding = Column(Vector(3072), nullable=True)  # text-embedding-3-large dimension
    filename = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<ChunkEmbedding(id={self.id}, filename={self.filename})>"


class Prompt(Base):
    """
    Table for storing prompt templates for the QA system.
    
    Purpose: Allows dynamic prompt management with user-specific and system-level prompts.
    Supports both 'generator' and 'validator' prompt types.
    """
    
    __tablename__ = "prompts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    type = Column(String(50), nullable=False)  # 'generator' or 'validator'
    template = Column(Text, nullable=False)
    user_id = Column(String(255), nullable=False)  # User ID or 'system' for defaults
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        CheckConstraint("type IN ('generator', 'validator')", name='check_prompt_type'),
    )
    
    def __repr__(self):
        return f"<Prompt(id={self.id}, type={self.type}, user_id={self.user_id})>"