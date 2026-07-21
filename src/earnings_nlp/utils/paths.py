"""Central path definitions so every module agrees on where things live."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

TRANSCRIPTS_RAW = DATA_RAW / "transcripts"
DOWNLOAD_LOG = TRANSCRIPTS_RAW / "download_log.csv"

CONFIG_DIR = ROOT / "config"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
