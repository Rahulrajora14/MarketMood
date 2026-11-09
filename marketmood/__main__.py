"""Command line entry point.

    python -m marketmood all                 # whole pipeline, raw data to figures
    python -m marketmood train --config config/default.yaml

Stages, in order:
    universe   freeze the ticker list from FNSPID coverage
    download   prices (Yahoo + Stooq check), EPU, Reddit
    sentiment  FinBERT + Loughran-McDonald scores, cached
    build      join everything into the firm-day panel, inspection report
    train      walk-forward training for both targets
    control    shuffled-label honesty check
    leakdemo   same models with the naive (leaky) news alignment, for contrast
    evaluate   tables, significance tests, regimes, backtest, figures
    export     refit headline models on all data for the API
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import pipeline as PL
from . import report as R
from .config import load_config

STAGES = ["universe", "download", "sentiment", "build", "train", "control", "leakdemo",
          "evaluate", "export"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="marketmood")
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)

    todo = STAGES if args.stage == "all" else [args.stage]
    for st in todo:
        logging.getLogger("marketmood").info("== %s ==", st)
        if st == "universe":
            if cfg.tickers and args.stage == "all":
                continue                      # already frozen; never re-pick
            PL.stage_universe(cfg)
            cfg = load_config(args.config)
        elif st == "download":
            PL.stage_download(cfg)
        elif st == "sentiment":
            PL.stage_sentiment(cfg)
        elif st == "build":
            PL.stage_build(cfg)
            R.data_inspection(cfg)
        elif st == "train":
            PL.stage_train(cfg)
        elif st == "control":
            PL.stage_train(cfg, shuffle_labels=True,
                           only=["logistic:finbert", "lightgbm:finbert", "lightgbm:price"])
        elif st == "leakdemo":
            PL.stage_leakdemo(cfg)
        elif st == "evaluate":
            R.evaluate_all(cfg)
            R.method_figures(cfg)
        elif st == "export":
            PL.stage_final_models(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
