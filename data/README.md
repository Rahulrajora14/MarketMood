# data/

| folder | contents | in git? |
|---|---|---|
| `raw/` | `prices.parquet` (Yahoo), `epu.parquet` (FRED), `news.parquet` (FNSPID, filtered to the 30 firms), `reddit.parquet`, `sp500_members_start.csv` | no |
| `raw/fnspid/`, `raw/fnspid_parts/` | the large FNSPID downloads and their filtered pieces (only needed to re-pick the universe) | no |
| `interim/` | `finbert_cache.parquet` (one row per unique headline), `news_scored.parquet`, `reddit_scored.parquet` | no |
| `processed/` | `panel.parquet` (firm-day panel) and every model's out-of-sample predictions | no |

The project folder you received already contains `raw/` (small files), `interim/` and
`processed/`, so `python -m marketmood train`, `evaluate`, the API and the dashboard all run
without downloading anything. To rebuild from scratch, delete these folders and run
`python -m marketmood all` (several GB of downloads, ~1 hour of FinBERT on a laptop CPU).
