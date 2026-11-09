# MarketMood: what we found

All numbers are out-of-sample: averaged over 8 yearly test folds (2016–2023), with every
model trained only on data before its test year. Full tables are in
`docs/RESULTS_TABLES.md` and `reports/tables/`. Figures are in `reports/figures/`.

## The data we ended up with

| | |
|---|---|
| Firms | 30 S&P 500 members from 1 Jan 2012, chosen by news coverage in 2012–2015 |
| Period | 3 Jan 2012 – 29 Dec 2023, 90,540 firm-days |
| News | 1,273,456 FNSPID rows for the 497 start-of-sample members → 1,120,685 after removing duplicates and multi-firm round-ups → 177,792 headlines for our 30 firms |
| Firm-days with news | 69.1% |
| FinBERT | 156,765 unique headlines scored once, cached |
| Reddit | 3,284 posts, only 2.3% of firm-days → too sparse, used for reporting only (plan's fallback) |
| Class balance | 52.0% of next-day returns are positive |

Two data facts shaped the design:

* **FNSPID timestamps are unusable at intraday precision** (99.6% are date-only; the rest
  are not in UTC). Every item is therefore used only from the next trading session after
  its date.
* **News mostly reacts to prices.** The tone of headlines dated D correlates 0.085 with the
  firm's return on day D, 0.040 with day D−1, and only 0.011 with day D+1
  (`figures/news_reaction_profile.png`). Most "news" describes moves that already happened.

## RQ1 — Does sentiment improve next-day direction?

**No.** Direction is close to unpredictable for every model, and adding text changes nothing measurable.

| model | balanced accuracy | AUC |
|---|---|---|
| Majority class | 0.500 | 0.500 |
| Persistence (tomorrow = today) | 0.489 | 0.489 |
| Logistic, price only | 0.509 | 0.515 |
| Logistic, price + FinBERT | 0.509 | 0.515 |
| LightGBM, price only | 0.503 | 0.506 |
| LightGBM, price + FinBERT | 0.504 | 0.506 |
| LightGBM, price + LM lexicon | 0.506 | 0.507 |

* FinBERT vs its price-only twin: +0.001 balanced accuracy, 95% bootstrap interval
  −0.002 to +0.005; Diebold–Mariano p = 0.62. Same picture for logistic regression.
* Of the 6 direction comparisons, one bootstrap interval (LightGBM + LM) just excludes zero,
  but its DM test gives p = 0.48 and every Holm-adjusted p-value (12 tests in total) is 1.0.
  One borderline result among this many comparisons is what chance alone produces.
* Persistence scoring *below* 0.5 shows a small short-term reversal: up days tend to be
  followed by down days. The price-only logistic model picks this up, which is why it is the
  best direction model.

## RQ2 — Does sentiment improve 5-day volatility forecasts?

**Barely, and not significantly.** Volatility is predictable (unlike direction), but price
history already captures it.

| model | QLIKE (lower better) | R² of log vol |
|---|---|---|
| Last week's vol (random walk) | 1.534 | −0.106 |
| HAR, price only | 0.960 | 0.249 |
| HAR + FinBERT | 0.959 | 0.252 |
| LightGBM, price only | 0.907 | 0.294 |
| LightGBM + FinBERT | 0.909 | 0.295 |

* HAR + FinBERT vs HAR: DM p = 0.78. LightGBM + FinBERT vs LightGBM: slightly *worse* on QLIKE, p = 0.61.
* In-sample, more negative news tone **is** associated with higher volatility the next week
  (standardised coefficient −0.008, z = −4.4, controlling for HAR terms). The association is
  real but too small to beat a good price model out of sample.
* The biggest gain in this whole project comes from the learner, not the text: LightGBM on
  price features beats the classic HAR model every year (R² 0.294 vs 0.249).

## RQ3 — Does the value of sentiment depend on policy uncertainty?

**No evidence that it does.**

* Interaction sentiment × EPU: p = 0.12 for volatility, p = 0.96 for direction.
* Within EPU terciles the gains from text are small and change sign (text helps volatility
  slightly in low-EPU periods and hurts slightly in mid/high).
* EPU itself matters: higher policy uncertainty predicts higher volatility (z = 5.6), and the
  2020 spike is visible in every volatility chart.

## Economic check

A long-short strategy (top vs bottom 20% by predicted P(up), traded one day late) turns
over ~135% of the book per day. At 5 bp per unit that costs ~17% a year. Before costs the
FinBERT model's Sharpe ratio is +0.16 and the price-only model's −0.20; after costs both are
strongly negative (−1.3 and −1.6). Not tradable.

## Is the pipeline honest?

* **Shuffled labels:** balanced accuracy falls to 0.496–0.502 (chance) for every model. Nothing leaks.
* **What leakage looks like:** we reran the same models joining news dated D to the return
  *of day D* (a common shortcut). LightGBM + FinBERT jumps from 0.504 to **0.521**, a
  "significant" gain of +0.019 (95% interval +0.014 to +0.024) over price-only. Logistic + FinBERT
  goes from 0.509 to 0.524. The price-only model is unchanged, as it should be. This is
  exactly the kind of result the plan set out to avoid reporting.

## One-paragraph conclusion

Under leakage-controlled, walk-forward evaluation with a fairly tuned price-only
competitor, news sentiment from FNSPID headlines does not improve next-day direction
forecasts for 30 large US stocks, gives at most a negligible and statistically
insignificant improvement in 5-day volatility forecasts, and its value does not depend on
the policy-uncertainty regime. The main reason is visible in the data: headline tone moves
with same-day and previous-day returns, so by the time a headline can be used, the price has
already reacted. When the same models are allowed a one-day look-ahead, a "significant"
sentiment effect appears, which shows how easily such results are produced. A credible
null result was an accepted outcome of the plan.

## Limitations (say these before the examiners do)

1. FNSPID gives dates, not reliable times, so same-day pre-close news is not used. A
   timestamped feed could find a small intraday effect we cannot see.
2. Survivorship: 30 firms that still trade in 2023; delisted 2012 members have no free prices.
   Both model families see the same firms, so the comparison is fair, but levels may be optimistic.
3. Sentiment labels are model outputs (FinBERT), not human annotations.
4. Headlines only, no article bodies (bodies exist only for part of the corpus).
5. Reddit coverage was too thin to test social-media sentiment.
6. Large US firms, daily horizon only.

## Project-II directions

* Scale to more firms and add a timestamped source to test same-day effects properly.
* Earnings-day subsample: the place where text is most likely to matter.
* Temporal fusion / attention model over sequences, now that the baselines are trustworthy.
* Firm-level uncertainty from LM uncertainty words instead of the market-wide EPU.
