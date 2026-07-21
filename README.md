# earnings-call-nlp-alpha

**Management–Analyst Language Divergence as an Equity Signal**

## 1. Project summary

This project tests whether a shift in management's tone between the
*prepared remarks* portion of an earnings call and the *Q&A* portion —
"management–analyst divergence" — predicts abnormal post-earnings stock
returns. It combines earnings call transcripts, FinBERT sentiment scoring,
event-study analysis, and a walk-forward long-short backtest.

**Status: Version 0.1.** Only transcript download and parsing are
implemented so far — see [Roadmap](#roadmap--current-status) below.

## 2. Research hypothesis

Prepared remarks are scripted and vetted; Q&A answers are improvised under
pressure from analysts. If management's tone deteriorates measurably from
script to Q&A — more hedging, less confidence, a more negative FinBERT
score — that gap may reveal information not yet in the stock price, ahead
of it showing up in subsequent guidance or results.

## 3. Data

- **Transcripts**: [Alpha Vantage `EARNINGS_CALL_TRANSCRIPT`](https://www.alphavantage.co/documentation/#earnings-call-transcript)
  endpoint, requested per `(ticker, quarter)` pair and cached locally as JSON.
- **Prices**: not yet integrated (planned: daily adjusted close via
  `yfinance`, plus a market/sector benchmark).
- **Current coverage (Version 0.1 milestone)**: 4 calls —
  AAPL 2024Q1, AAPL 2024Q2, MSFT 2024Q1, MSFT 2024Q2.
- **Planned initial universe**: 20 large-cap US companies across sectors,
  8 quarters each (~160 calls) — see `config/config.yaml`.

Raw transcript text is not committed to this repository (see
[Limitations](#7-limitations)); the download script reproduces it locally
from the API.

## 4. Methodology

Full intended pipeline:

```
Transcripts
    ↓
Speaker and section parsing        <- implemented (Version 0.1)
    ↓
FinBERT sentiment probabilities    <- not yet implemented
    ↓
Divergence and language-change features
    ↓
Event-return construction
    ↓
Walk-forward modelling
    ↓
Long-short portfolio
```

At the parsing stage, each spoken turn in a transcript becomes one row,
labeled with:

- **section**: `prepared` or `qa` — inferred from the Operator's
  announcement of the Q&A session (or, failing that, the first Analyst turn)
- **role**: `CEO`, `CFO`, `Other Management`, `Analyst`, `Operator`, or
  `Unknown` — inferred from the speaker's title string

Operator instructions, empty turns, exact-duplicate turns, page/formatting
artifacts, and very short non-substantive remarks (e.g. "Thanks.") are
dropped. Punctuation, hedging language, and sentence structure are left
untouched, since they may carry signal for later sentiment/linguistic
features.

## 5. Main results

Not applicable yet — no sentiment features, event study, or backtest have
been run. This section will report only genuine out-of-sample results once
those phases exist.

## 6. Visualizations

None yet.

## 7. Limitations

- **Transcript licensing/availability**: only transcripts reachable through
  the Alpha Vantage API are used; raw transcript text is not redistributed
  in this repository.
- **Section-boundary heuristic**: the prepared/Q&A split is inferred from
  operator language, not an explicit field in the source data, and should
  be spot-checked on a sample of calls before trusting downstream results.
- Sample size, execution timing, survivorship bias, and transaction costs
  are not yet relevant at this stage but will be documented here as later
  phases are added.

## 8. Reproduction instructions

### Install

Requires Python 3.11+ (developed against 3.12).

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in an
[Alpha Vantage API key](https://www.alphavantage.co/support/#api-key):

```
ALPHA_VANTAGE_API_KEY=your_key_here
```

### Usage

Download the Version 0.1 milestone transcripts (AAPL 2024Q1/Q2, MSFT
2024Q1/Q2) — skips any file already present in `data/raw/transcripts/`:

```
python -m earnings_nlp.data.download_transcripts
```

Parse them into a single clean dataframe and view descriptive statistics:
open `notebooks/02_transcript_parsing.ipynb`, or in Python:

```python
from earnings_nlp.processing.clean_transcripts import parse_transcript_files
from earnings_nlp.utils.paths import TRANSCRIPTS_RAW

files = sorted(TRANSCRIPTS_RAW.glob("*.json"))
df = parse_transcript_files(files)
print(df.groupby(["ticker", "quarter"]).size())
print(df["role"].value_counts())
```

### Run tests

```
pytest
```

## 9. Repository structure

```
config/              Project configuration (tickers, API settings)
data/                Raw/interim/processed data (gitignored, see data/README.md)
notebooks/           Exploration notebooks, numbered by pipeline stage
src/earnings_nlp/    Installable package: data download, processing,
                     features, models, backtest, shared utils
tests/               Unit tests
reports/             Figures and the final write-up (later phases)
scripts/             End-to-end pipeline/backtest entry points (later phases)
```

## Roadmap / current status

- [x] Phase 1–2: GitHub repo, folder structure
- [x] Phase 3: Python environment
- [x] Phase 5–6 (Version 0.1): download 4 milestone transcripts, parse into
      a labeled dataframe, descriptive stats, parser test
- [ ] Phase 7: interpretable linguistic features
- [ ] Phase 8–9: FinBERT sentiment + divergence features
- [ ] Phase 10–11: event returns + event study
- [ ] Phase 12–13: predictive modelling + backtest
- [ ] Phase 14–15: robustness tests + final presentation
