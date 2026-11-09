"""MarketMood dashboard.

    uvicorn api.main:app --port 8000          (terminal 1)
    streamlit run dashboard/app.py            (terminal 2)

Every number on screen comes from the API, which reads the pipeline's outputs.
"""
from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("MARKETMOOD_API", "http://127.0.0.1:8000")

INK, MUTED, PAPER = "#1E2430", "#6B6B6B", "#FAF7F2"
SET_COLORS = {"price": "#7A8B99", "attention": "#588B8B", "lm": "#E0A458",
              "finbert": "#C8553D", "all_text": "#2F4858", "baseline": "#B9B2A6"}
SET_LABELS = {"price": "Price only", "attention": "Price + news volume", "lm": "Price + LM lexicon",
              "finbert": "Price + FinBERT", "all_text": "Price + FinBERT + Reddit"}
REGIME_COLORS = ["#9DB4C0", "#5C6B73", "#253237"]

st.set_page_config(page_title="MarketMood", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
  .caption {color: #6B6B6B; font-size: 0.88rem;}
</style>""", unsafe_allow_html=True)


def _theme():
    return {"config": {"background": PAPER, "view": {"stroke": None},
                       "axis": {"labelColor": MUTED, "titleColor": INK, "gridColor": "#E4DED3",
                                "domainColor": INK},
                       "legend": {"labelColor": INK, "titleColor": INK}}}


alt.themes.register("marketmood", _theme)
alt.themes.enable("marketmood")


@st.cache_data(ttl=600)
def get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def df(path: str, **params) -> pd.DataFrame:
    return pd.DataFrame(get(path, **params))


def split_model(m: str):
    learner, fset = m.split(":")
    return learner, fset


try:
    health = get("/health")
except Exception as exc:
    st.error(f"Cannot reach the API at {API}. Start it with `uvicorn api.main:app --port 8000`.\n\n{exc}")
    st.stop()

st.title("MarketMood")
st.markdown(
    f"<div class='caption'>Does news and retail sentiment add anything to price data when "
    f"forecasting US large-cap stocks? {health['tickers']} firms, {health['from']} to {health['to']}, "
    f"walk-forward out-of-sample evaluation.</div>", unsafe_allow_html=True)

def render_firm(s: pd.DataFrame, tk: str) -> None:
    price = alt.Chart(s).mark_line(color=INK, strokeWidth=1.4).encode(
        x=alt.X("date:T", title=None), y=alt.Y("close:Q", title="adjusted close", scale=alt.Scale(zero=False)))
    tone = alt.Chart(s[s.news_n > 0]).mark_bar(width=2).encode(
        x="date:T", y=alt.Y("news_fb_mean:Q", title="FinBERT tone"),
        color=alt.condition("datum.news_fb_mean > 0", alt.value("#588B8B"), alt.value("#C8553D")),
        tooltip=["date:T", "news_n:Q", alt.Tooltip("news_fb_mean:Q", format=".2f")])
    st.altair_chart(alt.vconcat(price.properties(height=200, width=1050),
                                tone.properties(height=120, width=1050)), use_container_width=True)

    probs = s.melt(id_vars="date", value_vars=[c for c in ["p_lightgbm:price", "p_lightgbm:finbert"] if c in s],
                   var_name="model", value_name="p_up").dropna()
    probs["model"] = probs.model.map({"p_lightgbm:price": "Price only", "p_lightgbm:finbert": "Price + FinBERT"})
    st.markdown("**Out-of-sample P(up tomorrow)**")
    st.altair_chart(alt.Chart(probs).mark_line(strokeWidth=1.3).encode(
        x=alt.X("date:T", title=None), y=alt.Y("p_up:Q", scale=alt.Scale(zero=False), title="P(up)"),
        color=alt.Color("model:N", scale=alt.Scale(domain=["Price only", "Price + FinBERT"],
                                                   range=[SET_COLORS["price"], SET_COLORS["finbert"]]),
                        legend=alt.Legend(orient="top", title=None))).properties(height=180),
        use_container_width=True)

    days = s[s.news_n > 0].sort_values("news_n", ascending=False)
    if len(days):
        pick = st.selectbox("Headlines on", days["date"].dt.date.astype(str).tolist()[:60])
        h = df(f"/headlines/{tk}", date=pick)
        if len(h):
            h["dated"] = h["stated"].str.slice(0, 10) if "stated" in h else ""
            st.dataframe(h[["dated", "title", "fb_pos", "fb_neg", "fb_polarity", "lm_tone"]]
                         .style.format({"fb_pos": "{:.2f}", "fb_neg": "{:.2f}", "fb_polarity": "{:+.2f}",
                                        "lm_tone": "{:+.2f}"}),
                         use_container_width=True, hide_index=True)
            st.markdown("<div class='caption'>FNSPID gives a date but no reliable clock time, so an item dated D "
                        "is first used at the close of the next trading session after D. These are the items "
                        "that enter this session's forecast.</div>", unsafe_allow_html=True)


tab_find, tab_firm, tab_epu, tab_data = st.tabs(
    ["Findings", "Firm explorer", "Policy uncertainty", "Data and checks"])

# ------------------------------------------------------------------ findings
with tab_find:
    sd = df("/tables/summary_direction")
    sv = df("/tables/summary_volatility")
    sig = df("/tables/significance")

    def best(table, learner, fset, col):
        r = table[table.model == f"{learner}:{fset}"]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Direction, price only", f"{best(sd, 'lightgbm', 'price', 'bal_acc_mean'):.3f}",
              help="LightGBM balanced accuracy, mean over yearly folds")
    c2.metric("Direction, + FinBERT", f"{best(sd, 'lightgbm', 'finbert', 'bal_acc_mean'):.3f}",
              f"{best(sd, 'lightgbm', 'finbert', 'bal_acc_mean') - best(sd, 'lightgbm', 'price', 'bal_acc_mean'):+.3f}")
    c3.metric("Volatility QLIKE, price only", f"{best(sv, 'lightgbm', 'price', 'qlike_mean'):.3f}")
    c4.metric("Volatility QLIKE, + FinBERT", f"{best(sv, 'lightgbm', 'finbert', 'qlike_mean'):.3f}",
              f"{best(sv, 'lightgbm', 'finbert', 'qlike_mean') - best(sv, 'lightgbm', 'price', 'qlike_mean'):+.3f}",
              delta_color="inverse")

    left, right = st.columns(2)
    with left:
        st.subheader("Next-day direction")
        d = sd.copy()
        d["set"] = d.model.map(lambda m: "baseline" if split_model(m)[0] in ("majority", "persistence")
                               else split_model(m)[1])
        d["lo"], d["hi"] = d.bal_acc_mean - d.bal_acc_std, d.bal_acc_mean + d.bal_acc_std
        base = alt.Chart(d).encode(y=alt.Y("label:N", sort="-x", title=None,
                                           axis=alt.Axis(labelLimit=320)))
        chart = (base.mark_rule(strokeWidth=3, color="#E4DED3").encode(x="lo:Q", x2="hi:Q")
                 + base.mark_circle(size=90, opacity=1).encode(
                     x=alt.X("bal_acc_mean:Q", title="balanced accuracy", scale=alt.Scale(zero=False)),
                     color=alt.Color("set:N", scale=alt.Scale(domain=list(SET_COLORS),
                                                              range=list(SET_COLORS.values())), legend=None),
                     tooltip=["label", alt.Tooltip("bal_acc_mean:Q", format=".4f"),
                              alt.Tooltip("bal_acc_std:Q", format=".4f"), alt.Tooltip("auc_mean:Q", format=".4f")])
                 + alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 3], color=INK).encode(x="x:Q"))
        st.altair_chart(chart.properties(height=380), use_container_width=True)
    with right:
        st.subheader("5-day volatility")
        d = sv.copy()
        d["set"] = d.model.map(lambda m: "baseline" if split_model(m)[0] == "last_vol" else split_model(m)[1])
        chart = alt.Chart(d).mark_bar(height=16).encode(
            y=alt.Y("label:N", sort="x", title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("qlike_mean:Q", title="QLIKE (lower is better)", scale=alt.Scale(zero=False)),
            color=alt.Color("set:N", scale=alt.Scale(domain=list(SET_COLORS), range=list(SET_COLORS.values())),
                            legend=None),
            tooltip=["label", alt.Tooltip("qlike_mean:Q", format=".4f"), alt.Tooltip("r2_logvol_mean:Q", format=".3f")])
        st.altair_chart(chart.properties(height=380), use_container_width=True)

    st.subheader("Is the difference real?")
    st.markdown("<div class='caption'>Each text model against its own price-only twin. Diebold-Mariano on "
                "daily-averaged loss differentials (negative = text model better); Holm-adjusted p-values "
                "across all comparisons; 95% stationary-bootstrap interval for the change in balanced accuracy.</div>",
                unsafe_allow_html=True)
    show = sig[["task", "model", "vs", "loss", "dm_mean_diff", "dm_stat", "dm_p_value", "dm_p_holm",
                "balacc_diff", "ci_low", "ci_high"]].copy()
    st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns if show[c].dtype.kind == "f"}, na_rep="–"),
                 use_container_width=True, hide_index=True)

    bt = df("/tables/backtest")
    st.subheader("Economic check: long-short backtest")
    st.markdown("<div class='caption'>Top vs bottom quintile by predicted P(up), traded one day late, "
                "5 bp cost per unit turnover.</div>", unsafe_allow_html=True)
    st.dataframe(bt.style.format({c: "{:.3f}" for c in bt.columns if bt[c].dtype.kind == "f"}),
                 use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ firm explorer
with tab_firm:
    tickers = get("/tickers")
    a, b, c = st.columns([1, 1, 1])
    tk = a.selectbox("Firm", tickers, index=tickers.index("KO") if "KO" in tickers else 0)
    last = pd.Timestamp(health["to"])
    default_year = 2020 if last >= pd.Timestamp("2020-12-31") else last.year
    start = b.date_input("From", pd.Timestamp(f"{default_year}-01-01"))
    end = c.date_input("To", pd.Timestamp(f"{default_year}-12-31"))
    try:
        s = df(f"/series/{tk}", start=str(start), end=str(end))
        s["date"] = pd.to_datetime(s["date"])
    except Exception:
        s = None
        st.warning("No out-of-sample data for that firm and date range.")
    if s is not None:
        render_firm(s, tk)

# ------------------------------------------------------------------ EPU
with tab_epu:
    rd, rv = df("/tables/regime_direction"), df("/tables/regime_volatility")
    st.markdown("<div class='caption'>Test days are split into low / mid / high policy uncertainty using "
                "cut-points from each fold's training window only.</div>", unsafe_allow_html=True)
    def gains(table, metric, lower_better):
        t = table.pivot(index="regime", columns="model", values=f"{metric}_mean")
        rows = []
        for fset in ("attention", "lm", "finbert", "all_text"):
            m = f"lightgbm:{fset}"
            if m not in t:
                continue
            for reg in ("low", "mid", "high"):
                g = t.loc[reg, "lightgbm:price"] - t.loc[reg, m] if lower_better else t.loc[reg, m] - t.loc[reg, "lightgbm:price"]
                rows.append({"regime": f"{reg} EPU", "set": SET_LABELS[fset], "gain": g})
        return pd.DataFrame(rows)

    order = [SET_LABELS[k] for k in ("attention", "lm", "finbert", "all_text")]
    colors = [SET_COLORS[k] for k in ("attention", "lm", "finbert", "all_text")]
    l, r = st.columns(2)
    for col, table, metric, low, title, ylab in (
            (l, rd, "bal_acc", False, "Direction", "gain in balanced accuracy"),
            (r, rv, "qlike", True, "Volatility", "reduction in QLIKE")):
        with col:
            st.subheader(title)
            g = gains(table, metric, low)
            present = [k for k in ("attention", "lm", "finbert", "all_text") if SET_LABELS[k] in set(g["set"])]
            order = [SET_LABELS[k] for k in present]
            colors = [SET_COLORS[k] for k in present]
            st.altair_chart(alt.Chart(g).mark_bar().encode(
                x=alt.X("regime:N", sort=["low EPU", "mid EPU", "high EPU"], title=None,
                        axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("set:N", sort=order),
                y=alt.Y("gain:Q", title=ylab),
                color=alt.Color("set:N", sort=order, scale=alt.Scale(domain=order, range=colors),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["regime", "set", alt.Tooltip("gain:Q", format="+.4f")]
            ).properties(height=300), use_container_width=True)
    st.markdown("<div class='caption'>Bars show the change versus price-only LightGBM inside each tercile; "
                "above zero means the text features helped.</div>", unsafe_allow_html=True)
    st.subheader("Interaction test: sentiment x EPU")
    it = df("/tables/epu_interaction")
    st.dataframe(it.style.format({c: "{:.4f}" for c in it.columns if it[c].dtype.kind == "f"}),
                 use_container_width=True, hide_index=True)
    st.markdown("<div class='caption'>target ~ price controls + sentiment + EPU + sentiment x EPU, both standardised, "
                "standard errors clustered by date. A significant <b>sent_x_epu</b> would mean the effect of "
                "sentiment depends on the uncertainty regime.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ data
with tab_data:
    cov = df("/tables/coverage_per_firm")
    yr = df("/tables/coverage_per_year")
    l, r = st.columns(2)
    with l:
        st.subheader("News coverage per firm")
        st.altair_chart(alt.Chart(cov).mark_bar(color="#C8553D").encode(
            y=alt.Y("ticker:N", sort="-x", title=None),
            x=alt.X("news_day_share:Q", title="share of days with news", axis=alt.Axis(format="%"))
        ).encode(y=alt.Y("ticker:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False))
        ).properties(height=640), use_container_width=True)
    with r:
        st.subheader("Per year")
        st.dataframe(yr.style.format({c: "{:.3f}" for c in yr.columns if yr[c].dtype.kind == "f"}),
                     use_container_width=True, hide_index=True)
        st.subheader("Honesty check")
        try:
            sh = df("/tables/shuffled_label_control")
            st.markdown("<div class='caption'>Same pipeline, labels randomly permuted inside every training "
                        "window. Performance must fall to chance (0.5); if it did not, something would be "
                        "leaking.</div>", unsafe_allow_html=True)
            st.dataframe(sh.style.format({c: "{:.4f}" for c in sh.columns if sh[c].dtype.kind == "f"}),
                         use_container_width=True, hide_index=True)
        except Exception:
            st.info("Run `python -m marketmood control` to produce the shuffled-label check.")

    st.subheader("What leakage would have looked like")
    st.markdown("<div class='caption'>Same models and folds; only the way news is joined to prices changes. "
                "Joining news dated D to the return of day D manufactures a 'significant' gain.</div>",
                unsafe_allow_html=True)
    try:
        ld = df("/tables/leak_demo")
        ld = ld.astype({c: float for c in ("gain_vs_price", "ci_low", "ci_high") if c in ld})
        st.dataframe(ld.style.format({c: "{:.4f}" for c in ld.columns if ld[c].dtype.kind == "f"}, na_rep="–"),
                     use_container_width=True, hide_index=True)
        rp = df("/tables/news_reaction_profile")
        rp["day"] = rp.lag_k.map(lambda k: f"D{k:+d}" if k else "D")
        st.markdown("**News tone vs the firm's return on surrounding days**")
        st.altair_chart(alt.Chart(rp).mark_bar(size=34).encode(
            x=alt.X("day:N", sort=list(rp.day), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("corr:Q", title="correlation with tone of news dated D"),
            color=alt.condition("datum.lag_k <= 0", alt.value("#C8553D"), alt.value("#7A8B99"))
        ).properties(height=240), use_container_width=True)
    except Exception:
        st.info("Run `python -m marketmood leakdemo` and `evaluate` to produce this section.")
