"""Integration tests for src/database.py."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.database import (
    get_access_count,
    get_access_log,
    get_checked_in_names,
    get_encoding_count,
    init_db,
    is_inside,
    load_encodings,
    log_access,
    save_encodings,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return db_path


class TestEncodingOperations:
    def test_save_and_load(self, tmp_db):
        names = ["Alice", "Bob"]
        encodings = [np.random.rand(128) for _ in range(2)]
        save_encodings(names, encodings, db_path=tmp_db)

        result = load_encodings(db_path=tmp_db)
        assert result["names"] == names
        assert len(result["encodings"]) == 2
        np.testing.assert_array_almost_equal(
            result["encodings"][0], encodings[0]
        )

    def test_load_empty_raises(self, tmp_db):
        with pytest.raises(FileNotFoundError):
            load_encodings(db_path=tmp_db)

    def test_count(self, tmp_db):
        assert get_encoding_count(db_path=tmp_db) == 0
        save_encodings(
            ["Alice"], [np.random.rand(128)], db_path=tmp_db
        )
        assert get_encoding_count(db_path=tmp_db) == 1

    def test_save_replaces_existing(self, tmp_db):
        save_encodings(["Alice"], [np.random.rand(128)], db_path=tmp_db)
        save_encodings(["Bob", "Carol"], [np.random.rand(128)] * 2, db_path=tmp_db)
        assert get_encoding_count(db_path=tmp_db) == 2
        result = load_encodings(db_path=tmp_db)
        assert result["names"] == ["Bob", "Carol"]


class TestAccessLogOperations:
    def test_log_and_retrieve(self, tmp_db):
        log_access("Alice", "12:00:00", "ENTERED", db_path=tmp_db)
        log_access("Bob", "12:01:00", "ENTERED", db_path=tmp_db)

        logs = get_access_log(db_path=tmp_db)
        assert len(logs) == 2
        assert logs[0]["Name"] == "Bob"  # newest first

    def test_checked_in_names(self, tmp_db):
        log_access("Alice", "12:00:00", "ENTERED", db_path=tmp_db)
        names = get_checked_in_names(db_path=tmp_db)
        assert "Alice" in names

    def test_is_inside(self, tmp_db):
        assert not is_inside("Alice", db_path=tmp_db)
        log_access("Alice", "12:00:00", "ENTERED", db_path=tmp_db)
        assert is_inside("Alice", db_path=tmp_db)

    def test_count(self, tmp_db):
        assert get_access_count(db_path=tmp_db) == 0
        log_access("Alice", "12:00:00", "ENTERED", db_path=tmp_db)
        assert get_access_count(db_path=tmp_db) == 1

    def test_empty_log(self, tmp_db):
        logs = get_access_log(db_path=tmp_db)
        assert logs == []


class TestDatabaseInit:
    def test_init_creates_tables(self, tmp_db):
        assert tmp_db.exists()
        assert get_encoding_count(db_path=tmp_db) == 0
        assert get_access_count(db_path=tmp_db) == 0

    def test_init_idempotent(self, tmp_db):
        init_db(tmp_db)
        init_db(tmp_db)
        assert get_encoding_count(db_path=tmp_db) == 0
