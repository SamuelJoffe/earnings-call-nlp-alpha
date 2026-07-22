# earnings-call-nlp-alpha

**Management–Analyst Language Divergence as an Equity Signal**

## 1. Project summary

This project tests whether a shift in management's tone between the
*prepared remarks* portion of an earnings call and the *Q&A* portion —
"management–analyst divergence" — predicts abnormal post-earnings stock
returns. It combines earnings call transcripts, FinBERT sentiment scoring,
event-study analysis, and a walk-forward long-short backtest.

**Status: in progress, past Version 0.1.** Transcript parsing, linguistic
and FinBERT sentiment features, divergence features, and event returns are
implemented — see [Roadmap](#roadmap--current-status) below.

## 2. Research hypothesis

Prepared remarks are scripted and vetted; Q&A answers are improvised under
pressure from analysts. If management's tone deteriorates measurably from
script to Q&A — more hedging, less confidence, a more negative FinBERT
score — that gap may reveal information not yet in the stock price, ahead
of it showing up in subsequent guidance or results.

## 3. Data

- **Transcripts**: [Alpha Vantage `EARNINGS_CALL_TRANSCRIPT`](https://www.alphavantage.co/documentation/#earnings-call-transcript)
  endpoint, requested per `(ticker, quarter)` pair and cached locally as JSON.
- **Prices**: daily adjusted close via `yfinance`, for each call's ticker
  plus SPY as the market benchmark.
- **Current coverage (Version 0.1 milestone)**: 4 calls —
  AAPL 2024Q1, AAPL 2024Q2, MSFT 2024Q1, MSFT 2024Q2.
- **Earnings dates/timing**: Alpha Vantage's transcript response doesn't
  include a call date (only `{symbol, quarter, transcript}`), so the 4
  milestone calls' real calendar dates and before/during/after-market
  timing are curated in `config/config.yaml` from `yfinance`'s
  `Ticker.get_earnings_dates()`, cross-checked against each transcript's
  own reported EPS figure (e.g. AAPL 2024Q1's transcript says "EPS was
  $2.18", matching the 2024-02-01 row exactly). This doesn't scale to the
  full universe yet — see [Limitations](#7-limitations).
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
FinBERT sentiment probabilities    <- implemented (Phase 8)
    ↓
Divergence and language-change features  <- implemented (Phase 9)
    ↓
Event-return construction          <- implemented (Phase 10)
    ↓
Event study (quintile sort)        <- implemented (Phase 11), not yet run for real (see below)
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

Before FinBERT, `src/earnings_nlp/features/linguistic_features.py`
computes a simple interpretable baseline per turn (word count, average
sentence length, question count, positive/negative/uncertainty word
counts, first-person-plural count, numeric token count), using a small
hand-curated lexicon inspired by the Loughran-McDonald financial-sentiment
categories. These are aggregated per call into five groups — prepared
remarks, management Q&A answers, analyst questions, CEO speech, CFO
speech — as rates per 100 words so they're comparable across sections of
very different length, plus two prepared-to-Q&A change features
(`qa_length_change`, `uncertainty_change`). This establishes the pipeline
end to end before a large language model is introduced.

`src/earnings_nlp/features/finbert_features.py` then adds
[FinBERT](https://huggingface.co/ProsusAI/finbert) sentiment. FinBERT's
position limit is 512 tokens, so `src/earnings_nlp/processing/chunk_text.py`
splits each turn on actual tokenizer boundaries (not word/character counts)
into <=510-token chunks first. Each chunk gets a positive/negative/neutral
probability and a continuous `sentiment_score = positive_probability -
negative_probability`; chunk scores are aggregated per call into
`prepared_management_sentiment`, `qa_management_sentiment`,
`analyst_question_sentiment`, `ceo_sentiment`, `cfo_sentiment`,
`overall_sentiment`, `sentiment_dispersion` (std dev of sentiment_score
across all chunks in the call), `negative_chunk_percentage`, and
`qa_negative_probability` (mean negative probability across the whole
Q&A section).

`src/earnings_nlp/features/change_features.py` combines those per-call
sentiment features into the signature research features:

- **`management_qa_divergence`** = `prepared_management_sentiment -
  qa_management_sentiment` — the core hypothesis variable. Positive means
  management sounded more upbeat in its script than under questioning;
  near zero means consistent tone; negative means tone improved during
  Q&A.
- **`analyst_management_gap`** = `qa_management_sentiment -
  analyst_question_sentiment` — how much more/less positive management's
  answers are than the questions being asked.
- **`ceo_cfo_gap`** = `ceo_sentiment - cfo_sentiment`.
- **`quarterly_sentiment_change`** = each ticker's `overall_sentiment`
  minus its own previous quarter's (NaN for a ticker's first available
  quarter).
- **`qa_negativity_change`** = each ticker's `qa_negative_probability`
  minus its own previous quarter's.

`build_divergence_table()` also adds `prepared_sentiment`/`qa_sentiment`
aliases matching the final research table's column names from the
project plan.

`src/earnings_nlp/data/download_prices.py` and
`src/earnings_nlp/backtest/event_returns.py` then bring in market data.
For each call:

```
R_{i,t->t+k}  = P_{i,t+k} / P_{i,t} - 1
AR_{i,t->t+k} = R_{i,t->t+k} - R_{m,t->t+k}
```

evaluated at k = 1, 5, and 20 trading days, where `m` is the SPY
benchmark. The **signal-entry date** `t` — which close can first possibly
reflect the call's information — depends on when the call happened:

- **after_close**: that day's close already happened before the call, so
  it can't reflect it. `t` = the *next* trading day's close.
- **before_open** / **during_hours**: the regular session that day already
  had the chance to react. `t` = that same day's close.

All 4 milestone calls happen to be `after_close` (true for both AAPL and
MSFT historically), so `event_returns.py` is tested against synthetic
price series covering all three timing branches, including a case where
the call falls on a Friday to confirm the "next trading day" rule
correctly skips the weekend rather than naively adding one calendar day.

`src/earnings_nlp/backtest/event_study.py` tests `management_qa_divergence`
directly against forward abnormal returns, before any predictive model:
sort calls into divergence quintiles (Q1 = lowest, Q5 = highest), compare
mean/median abnormal return by quintile, check whether the relationship is
monotonic, compute the long-Q1/short-Q5 spread, check whether it survives
winsorizing extreme returns, and break results out by an arbitrary group
column (sector, year, ...). **It refuses to form quintiles from fewer than
5 calls** rather than silently reporting a 2- or 3-group split as if it
were a real quintile sort — see [Main results](#5-main-results).

## 5. Main results

Not applicable yet, and this is the phase where the Version 0.1 sample
size (n=4) actually bites: a quintile event study needs at least 5 calls,
and a meaningful one needs dozens per bucket to separate signal from
noise. `assign_divergence_quintiles()` correctly raises rather than
faking a smaller split on the 4 milestone calls (see
`notebooks/04_event_study.ipynb`). The quintile/monotonicity/spread logic
itself is validated against a 30-observation synthetic sample with a known
planted relationship (`tests/test_event_study.py`), so the code is ready
to run for real once Phase 4's full universe (20 companies x 8 quarters)
is downloaded.

What n=4 *can* show — not a finding, just what's currently computable —
is a plain correlation between `management_qa_divergence` and
`abnormal_return_5d` of 0.34, and the underlying scatter in
`reports/figures/divergence_vs_abnormal_return_5d.png`. (Also, MSFT's 2024
fiscal-Q1 call — 2023-10-24 — shows a -3.75% next-day abnormal return,
consistent with the well-known market reaction to that call's
cloud-growth commentary; that's a pipeline sanity check, not a finding.)

## 6. Visualizations

- `reports/figures/divergence_vs_abnormal_return_5d.png` — scatter of
  `management_qa_divergence` against `abnormal_return_5d` for the 4
  milestone calls.
- `reports/figures/divergence_distribution.png` — histogram of
  `management_qa_divergence` across the 4 milestone calls.

Both are illustrative at n=4, not yet evidence of anything.

## 7. Limitations

- **Transcript licensing/availability**: only transcripts reachable through
  the Alpha Vantage API are used; raw transcript text is not redistributed
  in this repository.
- **Section-boundary heuristic**: the prepared/Q&A split is inferred from
  operator language, not an explicit field in the source data, and should
  be spot-checked on a sample of calls before trusting downstream results.
- **Earnings date/timing is manually curated, not automated**: Alpha
  Vantage doesn't return a call date, so the 4 milestone calls' dates and
  before/during/after-market timing were sourced from `yfinance` and
  hand-verified against each transcript's reported EPS. Scaling to the
  full universe (Phase 4) will need this automated — matching
  `yfinance.Ticker.get_earnings_dates()` rows to Alpha Vantage's
  fiscal-quarter labels programmatically, which isn't always a 1:1
  calendar-quarter mapping.
- **Execution timing is approximate**: the before/during/after-market
  bucketing relies on `yfinance`'s reported release time (06:00 for
  before-open, 16:00 for after-close, observed consistently across
  tickers), which is itself an approximation of the actual release
  time, not a verified timestamp.
- **Sample size**: the Version 0.1 milestone has only 4 calls, too few for
  a meaningful quintile event study (see [Main results](#5-main-results)).
  Any correlation reported at this stage is a pipeline sanity check, not
  evidence.
- Survivorship bias and transaction costs are not yet relevant at this
  stage but will be documented here as later phases are added.

## 8. Reproduction instructions

### Install

Requires Python 3.11+ (developed against 3.12).

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e . --no-deps    # makes `earnings_nlp` importable
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

Compute the Phase 7 baseline linguistic features, aggregated per call:

```python
from earnings_nlp.features.linguistic_features import aggregate_call_features

call_features = aggregate_call_features(df)
print(call_features[["ticker", "quarter", "qa_length_change", "uncertainty_change"]])
```

Compute Phase 8 FinBERT sentiment features, aggregated per call (downloads
the ~440MB `ProsusAI/finbert` model on first run):

```python
from earnings_nlp.features.finbert_features import add_chunk_sentiment, aggregate_call_sentiment

chunk_df = add_chunk_sentiment(df)
call_sentiment = aggregate_call_sentiment(chunk_df)
print(call_sentiment)
```

Compute the Phase 9 divergence features:

```python
from earnings_nlp.features.change_features import build_divergence_table

divergence = build_divergence_table(call_sentiment)
print(divergence[["ticker", "quarter", "management_qa_divergence", "analyst_management_gap", "ceo_cfo_gap"]])
```

Download prices (each call's ticker plus SPY) and compute forward/abnormal
event returns at 1/5/20 trading days:

```
python -m earnings_nlp.data.download_prices
```

```python
from earnings_nlp.backtest.event_returns import load_price_series, build_event_return_table
from earnings_nlp.data.download_transcripts import load_config

config = load_config()
calls = config["milestone_calls"]
prices_by_ticker = {c["ticker"]: load_price_series(c["ticker"]) for c in calls}
benchmark_prices = load_price_series(config["price_source"]["benchmark_ticker"])

event_returns = build_event_return_table(calls, prices_by_ticker, benchmark_prices)
print(event_returns)
```

Run the Phase 11 event study: open `notebooks/04_event_study.ipynb`, which
merges divergence + event returns, shows what a correlation/scatter can
say at n=4, and demonstrates that quintile assignment correctly refuses
to run below 5 calls. Or in Python:

```python
from earnings_nlp.backtest.event_study import plot_divergence_scatter, assign_divergence_quintiles

research_table = divergence.merge(event_returns, on=["ticker", "quarter"])
plot_divergence_scatter(research_table, "management_qa_divergence", "abnormal_return_5d")

assign_divergence_quintiles(research_table)  # raises ValueError at n=4 -- expected
```

### Run tests

```
pytest                      # full suite, ~15s (downloads FinBERT on first run)
pytest -m "not integration" # skip the one test that loads the real model
```

## 9. Repository structure

```
config/              Project configuration (tickers, API settings)
data/                Raw/interim/processed data (gitignored, see data/README.md)
notebooks/           Exploration notebooks, numbered by pipeline stage
src/earnings_nlp/    Installable package: data download, processing,
                     features, models, backtest, shared utils
tests/               Unit tests
reports/             Figures (see reports/figures/) and the final write-up (later phases)
scripts/             End-to-end pipeline/backtest entry points (later phases)
```

## Roadmap / current status

- [x] Phase 1–2: GitHub repo, folder structure
- [x] Phase 3: Python environment
- [x] Phase 5–6 (Version 0.1): download 4 milestone transcripts, parse into
      a labeled dataframe, descriptive stats, parser test
- [x] Phase 7: interpretable linguistic features
- [x] Phase 8: FinBERT sentiment
- [x] Phase 9: divergence features
- [x] Phase 10: prices + event returns
- [x] Phase 11: event study code (quintile/monotonicity/spread logic
      validated on synthetic data; not yet run for real -- needs Phase 4's
      full universe, since n=4 can't support 5 quintiles)
- [ ] Phase 12–13: predictive modelling + backtest
- [ ] Phase 14–15: robustness tests + final presentation
