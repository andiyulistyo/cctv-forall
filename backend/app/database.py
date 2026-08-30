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


# Columns added after a release, as (table, column, SQLite column definition).
# Adding one to a model is not enough on its own: create_all() only creates
# missing *tables*, so an existing database keeps its old shape and every query
# touching the new column fails at runtime. SQLite's ALTER TABLE ADD COLUMN is
# the whole migration story here -- it is cheap, and appending a column with a
# constant default never rewrites the table.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("sources", "alpr_zone", "JSON"),
    # NOT NULL DEFAULT 0 rather than a bare INTEGER: existing rows would
    # otherwise read back as None, and SourceOut.auto_start is a bool.
    ("sources", "auto_start", "INTEGER NOT NULL DEFAULT 0"),
    ("plate_reads", "frame_path", "TEXT"),
    # Human review of a read. All three nullable: an existing row has not been
    # reviewed, and NULL is exactly how that is spelled (see models.PlateRead).
    ("plate_reads", "corrected_text", "VARCHAR(30)"),
    ("plate_reads", "reviewed_at", "DATETIME"),
    ("plate_reads", "reviewed_by", "VARCHAR(120)"),
    ("face_sightings", "frame_path", "TEXT"),
)


# Indexes for those columns. create_all() builds a table's indexes when it
# builds the table, and ALTER TABLE ADD COLUMN builds none -- so on a database
# that already existed, an index=True in the model is a statement about what a
# *fresh* install gets and nothing else. Named exactly as SQLAlchemy names
# them, so a fresh install and an upgraded one end up with the same schema
# rather than two indexes doing one job.
_ADDED_INDEXES: tuple[tuple[str, str, str], ...] = (
    ("ix_plate_reads_corrected_text", "plate_reads", "corrected_text"),
    ("ix_plate_reads_reviewed_at", "plate_reads", "reviewed_at"),
)


def _add_missing_indexes() -> None:
    with engine.begin() as conn:
        for name, table, column in _ADDED_INDEXES:
            rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            if column not in {r[1] for r in rows}:
                continue  # the column itself is not there yet
            conn.exec_driver_sql(
                f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({column})"
            )


def _add_missing_columns() -> None:
    with engine.begin() as conn:
        for table, column, definition in _ADDED_COLUMNS:
            rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            if not rows:
                continue  # table does not exist yet; create_all just made it
            if column in {r[1] for r in rows}:
                continue
            conn.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )


def init_db() -> None:
    """Create tables and bring an existing database up to the current shape."""
    from . import models  # noqa: F401  (ensures models are imported)

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    _add_missing_indexes()
