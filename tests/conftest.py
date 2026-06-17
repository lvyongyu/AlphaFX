from __future__ import annotations

import pytest

import alphafx.database as database_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect the default Database path to a per-test temp file.

    Without this, every test that constructs `Database()` (directly or via an
    agent default) writes the real `data/alphafx.db`, so the suite is not
    hermetic and pollutes the developer's working data. Tests that pass an
    explicit path are unaffected.
    """
    monkeypatch.setattr(database_module, "DB_PATH", tmp_path / "alphafx_test.db")
