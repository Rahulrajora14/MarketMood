# MarketMood: module-by-module walkthrough

This document traces every design decision in the research plan (Section 3.3) to the
code that implements it. Read it top to bottom once; afterwards use it as a map.

---

## 0. The one-paragraph version

We ask whether the *tone of news* about a company helps forecast that company's stock,
over and above what its own price history already tells us. We build one row per
company per trading day (a "firm-day"), compute price features and news-sentiment
features for that row using only information available at that day's 16:00 close,
and train the **same** model twice: once on price features only and once on price +
sentiment. The difference in out-of-sample performance is the value of sentiment. We
then check whether that difference is statistically real, whether it depends on how
much policy uncertainty there is in the economy, and whether it would survive trading
costs.

---

## 1. The flow

```
                  ┌──────────────── data layer ────────────────┐
                  │ FNSPID news   Yahoo prices   EPU   Reddit  │   marketmood/data/*
                  └───────────────────────┬────────────────────┘
                                          │
                  ┌──────────── calendar layer ────────────────┐
                  │ which trading session can each item inform? │   marketmood/calendar.py
                  └───────────────────────┬────────────────────┘
                                          │
          ┌──────────────────────── feature layer ───────────────────────────┐
          │ price.py: returns, HAR vol, range, turnover, market return       │
          │ sentiment.py: FinBERT P(pos/neg/neu), Loughran-McDonald counts    │   marketmood/features/*
          │ aggregate.py: per firm-day mean tone, disagreement, volume        │
          └───────────────────────┬──────────────────────────────────────────┘
                                  │
                  ┌──────────── panel layer ───────────────────┐
                  │ targets, EPU join, walk-forward folds       │   marketmood/panel.py
                  └───────────────────────┬────────────────────┘
                                          │
                  ┌──────────── model layer ───────────────────┐
                  │ baselines, logistic, LightGBM, HAR          │   marketmood/models.py
                  │ price-only vs price+text: same grid          │
                  └───────────────────────┬────────────────────┘
                                          │
                  ┌──────────── evaluation layer ──────────────┐
                  │ metrics, Diebold-Mariano, bootstrap,        │   marketmood/evaluate/*
                  │ EPU terciles, interaction, backtest         │   marketmood/report.py
                  └───────────────────────┬────────────────────┘
                                          │
                         FastAPI (api/main.py) ─► Streamlit (dashboard/app.py)
```

Every stage is a function in `marketmood/pipeline.py` and a command in
`python -m marketmood <stage>`. Each stage reads the previous stage's Parquet file, so you
can rerun any stage on its own.

---

## 2. Data layer — `marketmood/data/`

### 2.1 News: `news.py`, `fnspid_ranges.py`

**Source.** FNSPID (Dong, Fan & Peng, 2024) on Hugging Face. Two files:

| file | size | what is in it |
|---|---|---|
| `All_external.csv` | 5.7 GB | short wire headlines (Benzinga etc.), mostly 2009–2020 |
| `nasdaq_exteral_data.csv` | 23 GB | Nasdaq.com articles with full text, up to 2023 |

The big file is read in parallel byte ranges (`fnspid_ranges.py`): each worker
downloads ~50 MB, cuts it at row boundaries, parses it and keeps only our tickers and
years. Nothing else is saved.

**Cleaning** (`combine_parts`):
1. duplicates removed (same ticker, same day, same headline);
2. *round-ups* removed: a headline that appears under five or more tickers on the same
   day ("Stocks That Hit 52-Week Highs On Friday") says nothing about any one firm.

Counts at each step: `reports/tables/fnspid_cleaning.csv`.

**The timestamp problem (important — expect a viva question).**
FNSPID's `Date` column looks like `2020-05-22 08:45:06 UTC`. We checked whether those
times are trustworthy (`audit_timestamps`, output `reports/tables/fnspid_timestamp_audit.csv`):

* 99.6% of rows have time `00:00:00` → only the date is known;
* for rows with a time, recurring stories show the times are **not UTC**. A
  "52-Week Highs" story must be published during market hours (09:30–16:00 New York),
  but read as UTC its median time is ~02:45 New York. If the stated time is instead
  read as "New York time minus 4 hours", 100% of them fall inside market hours.

So we cannot place an item before or after the 16:00 close. **Decision:** use only the
date, and treat each item as published at 23:59 New York on that date. An item dated
D can then only inform the next trading session after D. This throws away same-day
news (weakening our own signal) but guarantees no after-close story leaks into the
forecast made at that day's close. Function: `end_of_stated_day`.

### 2.2 Prices: `prices.py`

* Yahoo Finance through `yfinance`, `auto_adjust=True` (split and dividend adjusted).
* Cross-check: Stooq was not reachable from our environment, so daily returns are
  compared with FNSPID's own price archive instead. Correlation of daily returns is
  ≥ 0.98 for most firms and over 99% of days agree within 0.5%
  (`reports/tables/price_crosscheck.csv`). GE and GME are lower because of the GE
  spin-offs and the GME split, which the two sources adjust differently.

### 2.3 Policy uncertainty: `epu.py`

Daily US Economic Policy Uncertainty index (Baker, Bloom & Davis), FRED series
`USEPUINDXD`. Built from that day's newspapers, so the value dated D is **not known at
D's close**. `panel.attach_epu` uses the latest value dated strictly before the
session.

### 2.4 Reddit: `reddit.py`

Submission titles from r/stocks, r/investing, r/StockMarket, r/wallstreetbets via the
Arctic Shift archive. Titles must contain the ticker as an upper-case word (or as
`$T` for one- and two-letter tickers). The archive often timed out, and only 3,284 posts
(2.3% of firm-days) were collected. Below the 10% coverage threshold in the config, so,
as the plan allows, Reddit is reported in the data inspection but not used as a feature set.

### 2.5 The universe (`pipeline.stage_universe`)

1. Take the S&P 500 members on **1 Jan 2012** (public membership history, fja05680/sp500).
2. Measure news coverage per firm = share of trading days with at least one item,
   **using 2012–2015 only** (the first training window), so the choice of firms uses no
   test-period information.
3. Walk down that ranking, keeping firms whose Yahoo price history spans the whole
   sample, until 30 are kept.

Result: 30 firms across tech, healthcare, energy, financials, industrials and consumer
names. Table: `reports/tables/universe_price_availability.csv`.

*Honest limitation:* free price sources drop delisted firms (Time Warner Cable,
Broadcom Corp., Whole Foods, DirecTV and others were in the index in 2012 but were
acquired later). Our panel therefore conditions on survival. Because the price-only
and sentiment models are compared on exactly the same firms and days, this bias
affects both sides equally.

---

## 3. Calendar layer — `marketmood/calendar.py`

The only place where "which day can this information be used on" is decided.

* `TradingCalendar` wraps the official NYSE calendar (`exchange_calendars`),
  including holidays and 13:00 early closes.
* `assign_session(ts)`: the first session whose close is **strictly after** `ts`.
  An item at exactly 16:00:00 goes to the next day.
* `decision_time(session)`: that session's close in UTC.

Tested in `tests/test_calendar.py`: before / at / after close, weekends, holidays,
early-close days, summer vs winter time, and the FNSPID date rule.

---

## 4. Feature layer — `marketmood/features/`

### 4.1 Price features (`price.py`) — all known at the close of day t

| feature | meaning |
|---|---|
| `ret_1d … ret_20d` | log return over the last 1, 2, 3, 5, 10, 20 days |
| `har_daily/weekly/monthly` | log volatility over the last 1, 5, 22 days (the HAR components of Corsi, 2009) |
| `range_vol` | log of log(High/Low): intraday range, a cleaner daily volatility proxy |
| `turnover_ratio` | today's volume relative to its 20-day average |
| `mkt_ret`, `mkt_ret_5d` | equal-weight average return of the 30 firms (market factor) |
| `dow_1..4`, `month_end` | calendar effects |

`tests/test_panel.py::test_price_features_do_not_peek` changes every price after a
given day and checks that no feature on or before that day changes.

### 4.2 Item-level sentiment (`sentiment.py`)

* **FinBERT** (`ProsusAI/finbert`): BERT further trained on financial text; for each
  headline gives P(positive), P(negative), P(neutral). Polarity = P(pos) − P(neg).
  Runs on CPU (~50 headlines/second); results are cached in
  `data/interim/finbert_cache.parquet` by a hash of the text, so it runs once and resumes
  if interrupted. Headlines only: article bodies exist for the Nasdaq subset only, so
  using them would make the two sources inconsistent.
* **Loughran–McDonald** word lists (finance-specific positive/negative/uncertainty
  words): net tone = (pos − neg) / (pos + neg). Cheap and fully interpretable; tells us
  whether a transformer is needed at all.

### 4.3 Firm-day aggregation (`aggregate.py`)

For each (firm, session): number of items, mean FinBERT polarity, disagreement (std of
polarity), share of clearly negative items, mean LM tone, LM uncertainty, a 3-session
count-weighted tone, and *abnormal attention* (today's log item count minus its own
22-day average). Days with no news get count 0 and tone 0, plus a `has_news` flag so the
model can tell "no news" from "neutral news".

---

## 5. Panel layer — `marketmood/panel.py`

### 5.1 Targets (`make_targets`) — all defined on **changes**, never price levels

* `y_dir` = 1 if log(C[t+1]/C[t]) > 0 else 0. Binary classification.
* `y_logvol` = log of the square root of the mean squared daily return over t+1…t+5.
* `target_time` = the close at which the last ingredient of the target is known.

Why not predict the price? Tomorrow's price ≈ today's price, so a model that copies
today's close scores R² ≈ 0.99 and has no skill. Many published "95% accuracy" results
are this.

### 5.2 Walk-forward folds (`walk_forward_folds`)

```
fold 2016: train 2012-01 … 2015-12 (minus 5-day embargo) │ test 2016
fold 2017: train 2012-01 … 2016-12 (minus 5-day embargo) │ test 2017
...
fold 2023: train 2012-01 … 2022-12 (minus 5-day embargo) │ test 2023
```

* Expanding window: the model always trains on everything before the test year.
* **Embargo** of 5 sessions: the volatility target looks 5 days ahead, so the last 5
  training rows would otherwise have targets that overlap the test year.
* Inside each training window the last 20% (again after an embargo) is used to choose
  hyper-parameters; the chosen configuration is then refitted on the full window.
* Never shuffled. Tested in `test_folds_are_ordered_and_embargoed`.

---

## 6. Model layer — `marketmood/models.py`

| target | baselines | learners (each run price-only and price+text) |
|---|---|---|
| direction | majority class; persistence (tomorrow's sign = today's) | regularised logistic regression; LightGBM |
| volatility | last week's volatility (random walk) | HAR regression (OLS); LightGBM |

**Fairness rules**
* identical learner, identical hyper-parameter grid (`param_grid`, same seed) and
  identical tuning budget for every feature set;
* preprocessing (median imputation, scaling) lives inside a scikit-learn `Pipeline`
  that is re-created and fitted inside each fold (`test_scaler_is_fitted_on_training_rows_only`);
* the decision threshold for "up" is the training-set share of up days, so a model
  cannot look good on balanced accuracy by always calling "up" in a bull market.

**Feature sets compared**

| name | columns |
|---|---|
| `price` | price features only |
| `attention` | price + news count, has-news, abnormal attention (no tone) |
| `lm` | price + attention + Loughran-McDonald tone |
| `finbert` | price + attention + FinBERT tone |
| `all_text` | finbert + Reddit features (only if Reddit coverage is usable) |

`attention` vs `finbert` separates "news happened" from "news was positive/negative".
`lm` vs `finbert` asks whether the transformer earns its cost.

---

## 7. Evaluation layer — `marketmood/evaluate/`, `marketmood/report.py`

* **Metrics** (`metrics.py`): balanced accuracy, macro-F1, ROC-AUC for direction;
  QLIKE and RMSE on log volatility for volatility. Reported as mean ± s.d. over the
  8 yearly folds, never one split.
* **Diebold–Mariano** (`stats.py`): losses are averaged across firms for each day, the
  daily difference (text model − price model) is tested with a Newey–West variance and
  the Harvey–Leybourne–Newbold correction. Negative = text model better.
* **Stationary block bootstrap** for a 95% interval on the change in balanced accuracy,
  resampling whole days in blocks (mean length 20) to keep serial dependence.
* **Multiple comparisons**: all comparisons are listed and Holm-adjusted p-values are reported.
* **EPU regimes** (`regime.py`): each test day is labelled low/mid/high using
  cut-points from that fold's *training* window only; metrics recomputed per regime.
  Plus a direct interaction regression: target ~ controls + sentiment + EPU +
  sentiment × EPU, standard errors clustered by date.
* **Backtest** (`backtest.py`): each day buy the top 20% of firms by predicted P(up),
  sell the bottom 20%, trade one day late, pay 5 bp per unit turnover.

---

## 8. Verification as code — `tests/`

`pytest` runs 27 checks in a few seconds on synthetic data. The important ones:

| test | what it guarantees |
|---|---|
| `test_calendar.py` (12 tests) | timestamps map to the right session, including holidays, early closes, DST, FNSPID dates |
| `test_price_features_do_not_peek` | future prices cannot change today's features |
| `test_sentiment_features_use_only_text_before_decision_point` | every text item used on day t was published before t's close |
| `test_epu_is_lagged` | EPU used on day t is dated before t |
| `test_folds_are_ordered_and_embargoed` | no overlap, ≥5-day gap, no training target realised after the test period starts |
| `test_scaler_is_fitted_on_training_rows_only` | preprocessing never sees test rows |
| `test_shuffled_labels_destroy_performance` | with real signal the model works; with permuted labels it falls to 50% |

The shuffled-label check is also run on the real data (`python -m marketmood control`):
if a pipeline still scores above chance after the labels are shuffled, something is leaking.

`python -m marketmood leakdemo` reruns the same models with the **naive** alignment
(news dated D joined to day D), which is what much of the literature does. The gap
between the two is the size of the leak.

---

## 9. Demo — `api/main.py`, `dashboard/app.py`

```
uvicorn api.main:app --port 8000        # terminal 1
streamlit run dashboard/app.py          # terminal 2  ->  http://localhost:8501
```

The dashboard has four tabs: Findings (headline numbers, significance table, backtest),
Firm explorer (price, daily FinBERT tone, out-of-sample P(up) and the actual headlines
for any day), Policy uncertainty (results by EPU tercile, interaction test), and Data and
checks (coverage, honesty check). It reads everything through the API, so what is shown
is exactly what the pipeline produced.

---

## 10. Results

See `docs/RESULTS.md` for the findings in plain language and `docs/RESULTS_TABLES.md`
for every table. Figures are in `reports/figures/`.
