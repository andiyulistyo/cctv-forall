"""SQLite database setup via SQLAlchemy.

WAL mode is enabled so that the FastAPI process and the per-source detection
worker processes can read/write concurrently without locking each other out.
"""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

DATABASE_URL = f"sqlite:///{settings.db_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    future=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ANN001
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Columns added after a release, as (table, column, SQLite type). Adding one to
# a model is not enough on its own: create_all() only creates missing *tables*,
# so an existing database keeps its old shape and every query touching the new
# column fails at runtime. SQLite's ALTER TABLE ADD COLUMN is the whole
# migration story here -- it is cheap, and appending a nullable column never
# rewrites the table.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("sources", "alpr_zone", "JSON"),
    ("plate_reads", "frame_path", "TEXT"),
    ("face_sightings", "frame_path", "TEXT"),
)


def _add_missing_columns() -> None:
    with engine.begin() as conn:
        for table, column, column_type in _ADDED_COLUMNS:
            rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            if not rows:
                continue  # table does not exist yet; create_all just made it
            if column in {r[1] for r in rows}:
                continue
            conn.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
            )


def init_db() -> None:
    """Create tables and bring an existing database up to the current shape."""
    from . import models  # noqa: F401  (ensures models are imported)

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
