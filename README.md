# MarketMood

**Sentiment-augmented prediction of US equity movement under varying policy-uncertainty regimes**

B.Tech Project-I, Department of Computer Science and Engineering, Delhi Technological University
Group DTU/CSE/BTP/GP136 · Guide: Dr. Nipun Bansal

## Research questions

| | Question | Target | Main metric |
|---|---|---|---|
| RQ1 | Does news / social sentiment improve next-day direction over a price-only model with the same tuning budget? | sign of next-day log return | balanced accuracy, macro-F1 |
| RQ2 | Does sentiment improve short-horizon volatility forecasts over a price-only (HAR) baseline? | log realised vol, next 5 days | QLIKE, RMSE |
| RQ3 | Does any gain from sentiment change with economic-policy uncertainty? | both | metrics within EPU terciles + interaction test |

## Pipeline

```
 FNSPID news ─┐                                         ┌─ price-only model ──┐
 Reddit posts ┤─► calendar ─► FinBERT + LM ─► firm-day ─┤                     ├─► walk-forward ─► DM test, bootstrap,
 Yahoo prices ┤   (16:00 ET     scores        panel     └─ price + sentiment ─┘    folds          EPU terciles, backtest
 EPU index  ──┘    cut-off)                                (same learner, same grid)                      │
                                                                                                FastAPI ─► Streamlit
```

## Layout

```
marketmood/            the package, one module per layer
  data/                loaders: prices.py, news.py (FNSPID), epu.py, reddit.py
  calendar.py          trading calendar + timestamp -> session mapping (the leakage boundary)
  features/            price.py, sentiment.py (FinBERT, Loughran-McDonald), aggregate.py
  panel.py             targets, firm-day panel, walk-forward folds with embargo
  models.py            baselines and learners behind one interface
  evaluate/            metrics, Diebold-Mariano + bootstrap, EPU regimes, backtest
  pipeline.py          stage functions that connect the layers
  report.py            tables and figures
api/main.py            FastAPI over the trained artefacts
dashboard/app.py       Streamlit dashboard (talks only to the API)
config/default.yaml    every setting that changes a number
tests/                 verification suite (leakage checks)
reports/               generated tables, figures, data-inspection report
docs/                  walkthrough notes
```

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt

pytest                            # verification suite, runs on synthetic data in seconds

python -m marketmood all          # raw data -> figures (first run downloads and scores everything)
# or stage by stage:
python -m marketmood universe     # freeze ticker list from FNSPID coverage
python -m marketmood download     # prices, EPU, Reddit
python -m marketmood sentiment    # FinBERT + LM (cached; resumes if interrupted)
python -m marketmood build        # panel + data inspection report
python -m marketmood train        # walk-forward, both targets
python -m marketmood control      # shuffled-label honesty check
python -m marketmood evaluate     # tables, tests, regimes, backtest, figures
python -m marketmood export       # refit headline models for the API

uvicorn api.main:app --port 8000  # terminal 1
streamlit run dashboard/app.py    # terminal 2
```

The FinBERT pass is the only slow step. It is cached in `data/interim/finbert_cache.parquet`,
so it runs once. On Google Colab, run `python -m marketmood sentiment` with a GPU runtime and
copy the cache file back.

## Data sources

| Source | What | Access |
|---|---|---|
| FNSPID (Dong, Fan & Peng, 2024) | timestamped financial news aligned to S&P 500 tickers | Hugging Face `Zihan1004/FNSPID` |
| Yahoo Finance via `yfinance` | daily OHLCV, split and dividend adjusted | public |
| Stooq | independent price series for the cross-check | public |
| Economic Policy Uncertainty, daily US | Baker, Bloom & Davis | FRED `USEPUINDXD` |
| Reddit (Arctic Shift archive) | submission titles from investing subreddits | public |
| fja05680/sp500 | historical S&P 500 membership, to freeze the universe at the sample start | GitHub |
| Loughran-McDonald Master Dictionary | finance word lists | `marketmood/resources/lm_dictionary.csv` |

Raw corpora are not committed (see `.gitignore`); the repository holds code, configuration and derived tables.

This is a research artefact for academic evaluation. It does not place orders and nothing in it is investment advice.
