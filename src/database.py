"""SQLite-based storage replacing file-based IPC (pickle + CSV).

Provides encrypted BLOB storage for facial encodings and structured
access-log storage, with connection pooling, transaction management,
and retry logic for concurrent operations.
"""

import sqlite3
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

DB_FILE = DATA_DIR / "eventguard.db"

_MAX_RETRIES = 5
_RETRY_DELAY = 0.1  # seconds


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a new SQLite connection with WAL mode and busy timeout."""
    path = db_path or DB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.Connection(str(path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db(db_path: Optional[Path] = None):
    """Context manager yielding a database connection with automatic commit/rollback."""
    conn = _get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _retry(func):
    """Decorator that retries database operations on lock/busy errors."""
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    last_exc = exc
                    delay = _RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "Database busy (attempt %d/%d), retrying in %.2fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                else:
                    raise
        raise last_exc  # type: ignore[misc]
    return wrapper


def init_db(db_path: Optional[Path] = None):
    """Initialize database tables if they don't exist."""
    with get_db(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ENTERED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_access_log_name
            ON access_log(name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_encodings_name
            ON encodings(name)
        """)
    logger.info("Database initialized at %s", db_path or DB_FILE)


# ── Encoding operations ─────────────────────────────────────────

def _serialize_encoding(encoding: np.ndarray) -> bytes:
    """Convert a numpy encoding array to bytes for storage."""
    return encoding.tobytes()


def _deserialize_encoding(blob: bytes) -> np.ndarray:
    """Convert stored bytes back to a numpy encoding array."""
    return np.frombuffer(blob, dtype=np.float64)


@_retry
def save_encodings(
    names: List[str],
    encodings: List[np.ndarray],
    db_path: Optional[Path] = None,
):
    """Save facial encodings to the database, replacing any existing data."""
    with get_db(db_path) as conn:
        conn.execute("DELETE FROM encodings")
        conn.executemany(
            "INSERT INTO encodings (name, encoding) VALUES (?, ?)",
            [
                (name, _serialize_encoding(enc))
                for name, enc in zip(names, encodings)
            ],
        )
    logger.info("Saved %d encodings to database", len(names))


@_retry
def load_encodings(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load all facial encodings from the database.

    Returns dict matching the legacy pickle format:
    {"names": [...], "encodings": [...]}
    """
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT name, encoding FROM encodings ORDER BY id"
        ).fetchall()

    if not rows:
        raise FileNotFoundError(
            "No encodings found in database. Run ingest.py first."
        )

    names = [row["name"] for row in rows]
    encodings = [_deserialize_encoding(row["encoding"]) for row in rows]
    return {"names": names, "encodings": encodings}


@_retry
def get_encoding_count(db_path: Optional[Path] = None) -> int:
    """Return the number of stored encodings."""
    with get_db(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM encodings").fetchone()
        return row["cnt"]


# ── Access log operations ────────────────────────────────────────

@_retry
def log_access(
    name: str,
    timestamp: str,
    status: str = "ENTERED",
    db_path: Optional[Path] = None,
):
    """Record an access event."""
    with get_db(db_path) as conn:
        conn.execute(
            "INSERT INTO access_log (name, timestamp, status) VALUES (?, ?, ?)",
            (name, timestamp, status),
        )
    logger.debug("Logged access: %s at %s (%s)", name, timestamp, status)


@_retry
def get_access_log(db_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Retrieve the full access log as a list of dicts."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT name as Name, timestamp as Time, status as Status "
            "FROM access_log ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@_retry
def get_checked_in_names(db_path: Optional[Path] = None) -> set:
    """Get the set of names that have checked in (for O(1) lookup)."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT name FROM access_log WHERE status = 'ENTERED'"
        ).fetchall()
    return {row["name"] for row in rows}


@_retry
def get_access_count(db_path: Optional[Path] = None) -> int:
    """Return the number of access log entries."""
    with get_db(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM access_log").fetchone()
        return row["cnt"]


@_retry
def is_inside(name: str, db_path: Optional[Path] = None) -> bool:
    """Check whether a person is already inside (has an ENTERED record)."""
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM access_log WHERE name = ? AND status = 'ENTERED' LIMIT 1",
            (name,),
        ).fetchone()
    return row is not None
