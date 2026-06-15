# Root conftest so pytest puts the repo root on sys.path — this makes
# `import alphafx` work under any invocation (the `pytest` console script in CI,
# not only `python -m pytest`). Sibling helpers in tests/ (e.g. `factories`)
# keep working via the tests/ directory insertion.
