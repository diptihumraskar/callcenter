"""Engine, session-factory, and transactional scope helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base

_session_factories: dict[int, sessionmaker[Session]] = {}


def get_engine(db_path: str = "data/calls.db", db_encryption_key: str | None = None) -> Engine:
    """Create a SQLite engine, adding a PRAGMA key listener if encrypted."""
    if db_path != ":memory:":
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path}", future=True)

    if db_encryption_key:

        @event.listens_for(engine, "connect")
        def _set_encryption_key(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute(f"PRAGMA key = '{db_encryption_key}'")
            cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create every table if it does not already exist."""
    Base.metadata.create_all(engine)


def get_session(engine: Engine) -> Session:
    """Return a session from a cached sessionmaker, keyed by engine identity."""
    key = id(engine)
    if key not in _session_factories:
        _session_factories[key] = sessionmaker(bind=engine, expire_on_commit=False)
    return _session_factories[key]()


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Commit on success, roll back on exception, always close."""
    session = get_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
