from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import config


def get_sqlalchemy_engine():
    url = config.RDS_URI
    return create_engine(url, pool_pre_ping=True)


def get_session_local():
    engine = get_sqlalchemy_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

