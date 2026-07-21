# Data directory

Contents of `raw/`, `interim/`, and `processed/` are gitignored — this
project does not redistribute transcript text or price data. Reproduce them
locally with the scripts in `src/earnings_nlp/data/` and `scripts/`.

- `raw/transcripts/` — one JSON file per call, named `TICKER_QUARTER.json`
  (e.g. `AAPL_2024Q1.json`), plus a `download_log.csv` audit trail. Produced
  by `src/earnings_nlp/data/download_transcripts.py`.
- `raw/prices/` — daily price data (added in a later phase).
- `interim/` — intermediate, not-yet-final outputs (e.g. parsed but
  unlabeled dataframes).
- `processed/` — final, analysis-ready tables (e.g. the parsed transcript
  dataframe, the eventual per-call feature table).
