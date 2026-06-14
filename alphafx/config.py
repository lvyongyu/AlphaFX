from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "alphafx.db"


@dataclass(frozen=True)
class Symbols:
    audusd: str = "AUDUSD=X"
    dxy: str = "DX-Y.NYB"
    vix: str = "^VIX"


DEFAULT_SYMBOLS = Symbols()

