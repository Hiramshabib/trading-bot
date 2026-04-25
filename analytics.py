"""
Analytics -- historical performance analysis for the trading bot.

Queries all runs stored in the database (live + backfilled), downloads
actual price data, and produces:

  1. Top-pick performance vs SPY -- did recommendations beat the market?
  2. Factor attribution -- which signal (momentum / analyst / fundamentals)
     most strongly predicts 5-day forward returns?
  3. Score trend leaderboard -- which tickers are consistently rising/falling
     in the model's opinion?
  4. Ticker consistency -- how often does each ticker crack the top-10?

Outputs a styled HTML report to reports/analytics_YYYY-MM-DD.html and
prints a summary to the terminal.

Usage:
    python analytics.py
    python analytics.py --top-n 5   # evaluate top-N picks per run (default 5)
    python analytics.py --no-html   # terminal-only output
"""
import argparse
import math
import os
import re
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import config
from storage.database import get_all_scores_with_dates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(val: float | None) -> str:
    if val is None or math.isnan(val):
        return "--"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.1f}%"


def _score(val: float | None) -> str:
    if val is None or math.isnan(val):
        return "--"
    return f"{val:.1f}"


def _arrow(slope: float) -> str:
    if slope > 1.5:
        return "^^"
    if slope > 0.3:
        return "^"
    if slope < -1.5:
        return "vv"
    if slope < -0.3:
        return "v"
    return "->"


def _forward_return(
    prices: pd.DataFrame,   # Close prices, tickers as columns
    ticker: str,
    from_date: pd.Timestamp,
    n_days: int = 5,
) -> float | None:
    """Return n-day forward return for ticker starting at from_date. None if unavailable."""
    if ticker not in prices.columns:
        return None
    col = prices[ticker].dropna()
    future = col[col.index > from_date]
    if len(future) < n_days:
        return None
    p_start = col[col.index <= from_date]
    if p_start.empty:
        return None
    p0 = float(p_start.iloc[-1])
    pn = float(future.iloc[n_days - 1])
    if p0 == 0:
        return None
    return (pn - p0) / p0


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _top_pick_performance(
    scores_df: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """
    For each run, compute the average forward 5-day return of the top-N
    recommendations and compare to SPY over the same window.
    """
    rows = []
    run_dates = scores_df[["run_id", "run_at"]].drop_duplicates().sort_values("run_at")

    for _, rd in run_dates.iterrows():
        run_id   = rd["run_id"]
        run_date = pd.Timestamp(rd["run_at"])
        source   = scores_df.loc[scores_df["run_id"] == run_id, "source"].iloc[0]

        top_picks = (
            scores_df[scores_df["run_id"] == run_id]
            .sort_values("rank")
            .head(top_n)["ticker"]
            .tolist()
        )

        pick_returns = []
        for t in top_picks:
            r = _forward_return(prices, t, run_date, n_days=5)
            if r is not None:
                pick_returns.append(r)

        spy_ret = _forward_return(prices, "SPY", run_date, n_days=5)

        if not pick_returns:
            continue

        avg_pick = float(np.mean(pick_returns))
        alpha    = (avg_pick - spy_ret) if spy_ret is not None else None

        rows.append({
            "date":         run_date.strftime("%Y-%m-%d"),
            "source":       source,
            "top_picks":    ", ".join(top_picks),
            "avg_return":   avg_pick,
            "spy_return":   spy_ret,
            "alpha":        alpha,
            "n_picks":      len(pick_returns),
        })

    return pd.DataFrame(rows)


def _factor_attribution(
    scores_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Pearson correlation between each factor score and actual
    5-day forward return across all (ticker, run) pairs.
    """
    pairs = []
    for _, row in scores_df.iterrows():
        run_date = pd.Timestamp(row["run_at"])
        fwd = _forward_return(prices, row["ticker"], run_date, n_days=5)
        if fwd is None:
            continue
        pairs.append({
            "fwd_5d":            fwd,
            "momentum_score":    row["momentum_score"],
            "analyst_score":     row["analyst_score"],
            "fundamentals_score": row["fundamentals_score"],
            "composite":         row["composite"],
        })

    if not pairs:
        return pd.DataFrame()

    df = pd.DataFrame(pairs).dropna()
    if len(df) < 5:
        return pd.DataFrame()

    factors = ["momentum_score", "analyst_score", "fundamentals_score", "composite"]
    rows = []
    for f in factors:
        corr = df["fwd_5d"].corr(df[f])
        rows.append({"factor": f.replace("_score", "").replace("_", " "), "correlation": corr, "n": len(df)})

    return pd.DataFrame(rows).sort_values("correlation", ascending=False)


def _score_trend_leaderboard(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker that appears in at least 3 runs, compute the linear
    slope of its composite score over time.
    """
    rows = []
    for ticker, grp in scores_df.groupby("ticker"):
        grp = grp.sort_values("run_at")
        if len(grp) < 3:
            continue
        x     = np.arange(len(grp), dtype=float)
        y     = grp["composite"].values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0])
        rows.append({
            "ticker":    ticker,
            "runs":      len(grp),
            "avg_score": float(y.mean()),
            "avg_rank":  float(grp["rank"].mean()),
            "slope":     slope,
            "direction": _arrow(slope),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("slope", ascending=False)
        .reset_index(drop=True)
    )


def _suggest_weights(attr_df: pd.DataFrame, blend: float = 0.50) -> dict[str, float]:
    """
    Compute blended factor weights from attribution correlations.

    blend = fraction of weight to derive from data (1 - blend comes from the
    current config weights as a prior). Lower blend = more conservative.

    Returns a dict suitable for pasting into config.FACTOR_WEIGHTS.
    """
    factor_map = {
        "momentum":     "momentum",
        "analyst":      "analyst",
        "fundamentals": "fundamentals",
    }
    corrs = {}
    for _, row in attr_df.iterrows():
        key = row["factor"].strip().lower().replace(" ", "_")
        if key in factor_map and key != "composite":
            corrs[factor_map[key]] = float(row["correlation"])

    if len(corrs) < 3:
        return dict(config.FACTOR_WEIGHTS)

    vals    = np.array([corrs[k] for k in ("momentum", "analyst", "fundamentals")])
    shifted = vals - vals.min()
    corr_w  = shifted / shifted.sum() if shifted.sum() > 0 else np.ones(3) / 3

    prior   = config.FACTOR_WEIGHTS
    factors = ("momentum", "analyst", "fundamentals")
    blended = {
        f: blend * corr_w[i] + (1 - blend) * prior[f]
        for i, f in enumerate(factors)
    }
    total = sum(blended.values())
    return {k: round(v / total, 4) for k, v in blended.items()}


def _apply_weights_to_config(new_weights: dict[str, float], corrs: dict[str, float], n_pairs: int) -> None:
    """Rewrite the FACTOR_WEIGHTS block in config.py with new_weights."""
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    content     = open(config_path, encoding="utf-8").read()

    corr_comment = (
        f"# Updated {date.today()} via analytics.py --suggest-weights "
        f"({n_pairs} pairs):\n"
        f"#   momentum corr={corrs.get('momentum', 0):+.3f}, "
        f"analyst corr={corrs.get('analyst', 0):+.3f}, "
        f"fundamentals corr={corrs.get('fundamentals', 0):+.3f}\n"
        f"#   Weights blended 50% data-driven / 50% prior.\n"
    )

    new_block = (
        corr_comment
        + "FACTOR_WEIGHTS = {\n"
        + f'    "momentum":     {new_weights["momentum"]},\n'
        + f'    "analyst":      {new_weights["analyst"]},\n'
        + f'    "fundamentals": {new_weights["fundamentals"]},\n'
        + "}"
    )

    # Replace the entire FACTOR_WEIGHTS block (including any preceding comment lines)
    pattern = r"(?:#[^\n]*\n)*FACTOR_WEIGHTS\s*=\s*\{[^}]*\}"
    updated = re.sub(pattern, new_block, content, flags=re.DOTALL)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(updated)


def _ticker_consistency(scores_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """How often does each ticker appear in the top-N across all runs?"""
    total_runs = scores_df["run_id"].nunique()
    top = scores_df[scores_df["rank"] <= top_n]
    counts = top.groupby("ticker").size().reset_index(name="top10_appearances")
    counts["pct_of_runs"] = counts["top10_appearances"] / total_runs
    return counts.sort_values("top10_appearances", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _html_table(df: pd.DataFrame, fmt: dict[str, callable] | None = None) -> str:
    fmt = fmt or {}
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            renderer = fmt.get(col)
            if renderer:
                display = renderer(val)
            elif isinstance(val, float):
                display = f"{val:.2f}"
            else:
                display = str(val) if pd.notna(val) else "--"
            # Color-code returns and alpha
            cls = ""
            if col in ("avg_return", "spy_return", "alpha", "slope") and isinstance(val, float) and not math.isnan(val):
                cls = ' class="pos"' if val > 0 else ' class="neg"'
            cells.append(f"<td{cls}>{display}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
    <table>
      <thead><tr>{headers}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>"""


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 2rem; }
h1   { font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem; }
.subtitle { color: #94a3b8; margin-bottom: 2.5rem; font-size: 0.9rem; }
h2   { font-size: 1.1rem; font-weight: 600; color: #cbd5e1; margin: 2rem 0 0.75rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.4rem; }
p.note { font-size: 0.78rem; color: #64748b; margin-bottom: 0.75rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th    { text-align: left; padding: 0.45rem 0.75rem; background: #1e293b; color: #94a3b8; font-weight: 600; border-bottom: 1px solid #334155; }
td    { padding: 0.4rem 0.75rem; border-bottom: 1px solid #1e293b; }
tr:hover td { background: #1e293b; }
.pos  { color: #4ade80; font-weight: 600; }
.neg  { color: #f87171; font-weight: 600; }
.stat-grid { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem; }
.stat-box  { background: #1e293b; border-radius: 8px; padding: 1rem 1.5rem; min-width: 140px; }
.stat-box .label { font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem; }
.stat-box .value { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }
"""


def _build_html(
    perf_df:   pd.DataFrame,
    attr_df:   pd.DataFrame,
    trend_df:  pd.DataFrame,
    consist_df: pd.DataFrame,
    top_n:     int,
    run_count: int,
    date_range: tuple[str, str],
) -> str:
    # -- Overview stats ---------------------------------------------------
    avg_alpha = perf_df["alpha"].mean() if not perf_df.empty and "alpha" in perf_df else float("nan")
    win_rate  = (
        (perf_df["alpha"] > 0).sum() / len(perf_df)
        if not perf_df.empty and len(perf_df) > 0
        else float("nan")
    )

    stats_html = f"""
    <div class="stat-grid">
      <div class="stat-box"><div class="label">Total Runs</div><div class="value">{run_count}</div></div>
      <div class="stat-box"><div class="label">Date Range</div><div class="value" style="font-size:0.9rem">{date_range[0]} -> {date_range[1]}</div></div>
      <div class="stat-box"><div class="label">Avg Alpha (5d)</div>
        <div class="value {'pos' if avg_alpha > 0 else 'neg'}">{_pct(avg_alpha)}</div></div>
      <div class="stat-box"><div class="label">Win Rate vs SPY</div>
        <div class="value {'pos' if win_rate > 0.5 else 'neg'}">{_pct(win_rate)}</div></div>
    </div>"""

    # -- Performance table ------------------------------------------------
    perf_fmt = {
        "avg_return": _pct,
        "spy_return": _pct,
        "alpha":      _pct,
    }
    perf_section = ""
    if not perf_df.empty:
        perf_display = perf_df.drop(columns=["source"], errors="ignore")
        perf_section = f"""
    <h2>Top-{top_n} Pick Performance vs SPY (5-day forward return)</h2>
    <p class="note">Momentum scores are historically accurate. Analyst & fundamentals scores use today's data as a proxy.</p>
    {_html_table(perf_display, perf_fmt)}"""

    # -- Factor attribution -----------------------------------------------
    attr_section = ""
    if not attr_df.empty:
        attr_fmt = {"correlation": lambda v: f"{v:+.3f}"}
        attr_section = f"""
    <h2>Factor Attribution -- Pearson Correlation with 5-Day Forward Return</h2>
    <p class="note">Higher (more positive) = the factor predicts actual returns better across all (ticker, run) pairs.</p>
    {_html_table(attr_df, attr_fmt)}"""

    # -- Score trend ------------------------------------------------------
    trend_section = ""
    if not trend_df.empty:
        top_rising  = trend_df.head(10)
        top_falling = trend_df.tail(10).sort_values("slope")
        trend_section = f"""
    <h2>Score Trend Leaderboard -- Rising Tickers</h2>
    <p class="note">Slope = score-points per run. ^^ = strong uptrend, vv = strong downtrend.</p>
    {_html_table(top_rising,  {"slope": lambda v: f"{v:+.2f}", "avg_score": lambda v: f"{v:.1f}", "avg_rank": lambda v: f"{v:.1f}"})}
    <h2>Score Trend Leaderboard -- Falling Tickers</h2>
    {_html_table(top_falling, {"slope": lambda v: f"{v:+.2f}", "avg_score": lambda v: f"{v:.1f}", "avg_rank": lambda v: f"{v:.1f}"})}"""

    # -- Consistency ------------------------------------------------------
    consist_section = ""
    if not consist_df.empty:
        consist_fmt = {"pct_of_runs": _pct}
        consist_section = f"""
    <h2>Ticker Consistency -- Top-10 Appearances</h2>
    {_html_table(consist_df.head(20), consist_fmt)}"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trading Bot Analytics -- {date.today()}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>Trading Bot Analytics</h1>
  <p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;·&nbsp; top-{top_n} evaluation window</p>
  {stats_html}
  {perf_section}
  {attr_section}
  {trend_section}
  {consist_section}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Trading bot analytics report")
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of top picks per run to evaluate (default: 5)"
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip HTML report generation"
    )
    parser.add_argument(
        "--suggest-weights", action="store_true",
        help="Compute and display data-driven factor weight suggestions"
    )
    parser.add_argument(
        "--apply-weights", action="store_true",
        help="Like --suggest-weights but also writes the new weights to config.py"
    )
    args = parser.parse_args()

    print("\nLoading historical run data from database...")
    scores_df = get_all_scores_with_dates()

    if scores_df.empty:
        print("No runs found in database. Run backfill.py first.")
        return

    run_count  = scores_df["run_id"].nunique()
    date_range = (
        pd.Timestamp(scores_df["run_at"].min()).strftime("%Y-%m-%d"),
        pd.Timestamp(scores_df["run_at"].max()).strftime("%Y-%m-%d"),
    )
    print(f"  {run_count} runs loaded ({date_range[0]} -> {date_range[1]})")

    # ---- Download price history for all tickers + SPY -------------------
    all_tickers = list(scores_df["ticker"].unique()) + ["SPY"]
    print(f"\nDownloading price data for {len(all_tickers)} tickers...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            all_tickers,
            start=date_range[0],
            # fetch extra days beyond the last run for forward-return windows
            end=(pd.Timestamp(date_range[1]) + timedelta(days=14)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": all_tickers[0]})
    prices = prices.dropna(axis=1, how="all")

    # ---- Run analyses ---------------------------------------------------
    print("Computing performance vs SPY...")
    perf_df = _top_pick_performance(scores_df, prices, args.top_n)

    print("Computing factor attribution...")
    attr_df = _factor_attribution(scores_df, prices)

    print("Computing score trends...")
    trend_df = _score_trend_leaderboard(scores_df)

    print("Computing ticker consistency...")
    consist_df = _ticker_consistency(scores_df, top_n=10)

    # ---- Terminal summary -----------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  ANALYTICS SUMMARY  ({date_range[0]} -> {date_range[1]})")
    print(f"{'=' * 60}")

    if not perf_df.empty:
        avg_alpha = perf_df["alpha"].dropna().mean()
        win_rate  = (perf_df["alpha"].dropna() > 0).mean()
        print(f"\n  Top-{args.top_n} avg alpha (5d):  {_pct(avg_alpha)}")
        print(f"  Win rate vs SPY:       {_pct(win_rate)}")
        print(f"\n  Recent top-pick performance:")
        for _, r in perf_df.tail(5).iterrows():
            print(
                f"    {r['date']}  picks={r['top_picks']:<30s}  "
                f"return={_pct(r['avg_return'])}  spy={_pct(r['spy_return'])}  "
                f"alpha={_pct(r['alpha'])}"
            )

    if not attr_df.empty:
        print(f"\n  Factor attribution (correlation -> 5d return):")
        for _, r in attr_df.iterrows():
            bar_len = max(0, int(abs(r["correlation"]) * 20))
            bar     = ("#" * bar_len).ljust(20)
            print(f"    {r['factor']:<20s}  {r['correlation']:+.3f}  {bar}")

    if not trend_df.empty:
        print(f"\n  Top-5 rising tickers:")
        for _, r in trend_df.head(5).iterrows():
            print(f"    {r['ticker']:6s}  {r['direction']}  slope={r['slope']:+.2f}  avg_score={r['avg_score']:.1f}")
        print(f"\n  Top-5 falling tickers:")
        for _, r in trend_df.tail(5).sort_values("slope").iterrows():
            print(f"    {r['ticker']:6s}  {r['direction']}  slope={r['slope']:+.2f}  avg_score={r['avg_score']:.1f}")

    if not consist_df.empty:
        print(f"\n  Most consistent top-10 tickers:")
        for _, r in consist_df.head(8).iterrows():
            print(f"    {r['ticker']:6s}  {r['top10_appearances']:>3d} runs  ({_pct(r['pct_of_runs'])})")

    print(f"\n{'=' * 60}\n")

    # ---- Weight suggestion ----------------------------------------------
    if (args.suggest_weights or args.apply_weights) and not attr_df.empty:
        corrs = {
            row["factor"].strip().lower().replace(" ", "_"): float(row["correlation"])
            for _, row in attr_df.iterrows()
            if row["factor"].strip().lower() != "composite"
        }
        suggested = _suggest_weights(attr_df)
        n_pairs   = int(attr_df["n"].iloc[0]) if "n" in attr_df.columns else 0

        print("\nFactor weight suggestion (50% data-driven / 50% current prior):")
        print(f"  {'Factor':<15s}  {'Corr':>7s}  {'Current':>8s}  {'Suggested':>10s}  {'Change':>8s}")
        print(f"  {'-'*55}")
        for f in ("momentum", "analyst", "fundamentals"):
            curr = config.FACTOR_WEIGHTS.get(f, 0)
            sugg = suggested.get(f, 0)
            corr = corrs.get(f, float("nan"))
            print(f"  {f:<15s}  {corr:>+7.3f}  {curr:>8.4f}  {sugg:>10.4f}  {sugg-curr:>+8.4f}")

        if args.apply_weights:
            _apply_weights_to_config(suggested, corrs, n_pairs)
            print(f"\n  config.py updated. Restart main.py to use new weights.")
        else:
            print(f"\n  Run with --apply-weights to write these to config.py.")

    # ---- HTML report ----------------------------------------------------
    if not args.no_html:
        html = _build_html(
            perf_df, attr_df, trend_df, consist_df,
            top_n=args.top_n,
            run_count=run_count,
            date_range=date_range,
        )
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        path = os.path.join(config.REPORTS_DIR, f"{date.today()}_analytics.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML report saved: {path}")


if __name__ == "__main__":
    main()
