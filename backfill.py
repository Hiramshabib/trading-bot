"""
Backfill historical runs for the last N trading days.

For each trading day in the window, recomputes momentum scores using
historically-accurate price data sliced to that date. Analyst and
fundamentals scores use today's data as a proxy -- yfinance does not
offer free point-in-time snapshots for those signals, so they are held
constant across all backfilled days.

Usage:
    python backfill.py              # backfill last 30 trading days
    python backfill.py --days 10    # backfill last N trading days
    python backfill.py --dry-run    # preview without writing to DB
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import config
from data.analyst      import fetch_analyst_scores
from data.downside     import compute_downside_alerts
from data.fundamentals import fetch_fundamentals_scores
from data.momentum     import _rsi
from data.universe     import TOP_50_BY_MARKET_CAP, get_universe
from output.html_report import export_html
from scoring.composite import build_composite_scores
from storage.database  import init_db, run_exists_for_date, save_run_at


# ---------------------------------------------------------------------------
# Historical momentum -- same logic as data/momentum.py but sliced to a date
# ---------------------------------------------------------------------------

def _momentum_as_of(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute momentum metrics for every ticker using price history up to
    and including `as_of`. Returns same schema as fetch_momentum_scores().
    """
    c = close[close.index <= as_of]
    v = volume[volume.index <= as_of]

    rows = []
    for ticker in c.columns:
        prices = c[ticker].dropna()
        vols   = v[ticker].dropna() if ticker in v.columns else pd.Series(dtype=float)

        if len(prices) < 6:
            continue

        p_now = prices.iloc[-1]
        p_1d  = prices.iloc[-2]  if len(prices) >= 2  else p_now
        p_5d  = prices.iloc[-6]  if len(prices) >= 6  else prices.iloc[0]
        p_1mo = prices.iloc[-22] if len(prices) >= 22 else prices.iloc[0]

        pct_1d  = (p_now - p_1d)  / p_1d  if p_1d  else 0.0
        pct_5d  = (p_now - p_5d)  / p_5d  if p_5d  else 0.0
        pct_1mo = (p_now - p_1mo) / p_1mo if p_1mo else 0.0

        w = config.MOMENTUM_WEIGHTS
        momentum_raw = (
            pct_1d  * w["1d"] +
            pct_5d  * w["5d"] +
            pct_1mo * w["1mo"]
        )

        rsi_14    = _rsi(prices, 14)
        vol_ratio = 1.0
        if len(vols) >= 10:
            vol_10d = vols.iloc[-10:].mean()
            vol_60d = vols.iloc[-60:].mean() if len(vols) >= 60 else vols.mean()
            if vol_60d and vol_60d > 0:
                vol_ratio = round(float(vol_10d / vol_60d), 2)

        rows.append({
            "ticker":       ticker,
            "momentum_raw": momentum_raw,
            "pct_1d":       pct_1d,
            "pct_5d":       pct_5d,
            "pct_1mo":      pct_1mo,
            "rsi_14":       rsi_14,
            "vol_ratio":    vol_ratio,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical trading bot runs into the database"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of trading days to backfill (default: 30)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be written without touching the database"
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip generating HTML reports (DB only)"
    )
    args = parser.parse_args()

    init_db()

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Backfilling last {args.days} trading days...")

    # ---- Universe --------------------------------------------------------
    tickers = get_universe()
    print(f"Universe: {len(tickers)} tickers\n")

    # ---- Download extended price history ---------------------------------
    # Need: args.days for backfill window
    #     + 22 trading days for 1-month momentum lookback
    #     + 14 trading days for RSI
    #     + safety buffer
    # -> 6 months is comfortably sufficient for any --days <= 90
    print("Downloading price history (6 months)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            period="6mo",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"]
        volume = raw["Volume"]
    else:
        # Single-ticker edge case
        close  = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    close  = close.dropna(axis=1, how="all")
    volume = volume.dropna(axis=1, how="all")

    # ---- Identify target trading days ------------------------------------
    all_days     = list(close.index)
    target_days  = all_days[-args.days:]
    print(
        f"Target window: {target_days[0].date()} -> {target_days[-1].date()} "
        f"({len(target_days)} days)\n"
    )

    # ---- Fetch analyst & fundamentals once (current-data proxy) ----------
    print("Fetching analyst scores (today's data -- used as proxy for all days)...")
    analyst_df = fetch_analyst_scores(tickers)

    print("\nFetching fundamentals scores (today's data -- used as proxy for all days)...")
    fundamentals_df = fetch_fundamentals_scores(tickers)

    # ---- Process each trading day ----------------------------------------
    print()
    inserted = skipped_exists = skipped_empty = 0

    for trade_date in target_days:
        date_str = trade_date.strftime("%Y-%m-%d")

        html_path = os.path.join(config.REPORTS_DIR, f"{date_str}_recommendations.html")
        html_exists = os.path.exists(html_path)

        if run_exists_for_date(date_str) and (args.no_html or html_exists):
            print(f"  {date_str}  already exists -- skipping")
            skipped_exists += 1
            continue

        momentum_df = _momentum_as_of(close, volume, trade_date)

        if momentum_df.empty:
            print(f"  {date_str}  no momentum data -- skipping")
            skipped_empty += 1
            continue

        results = build_composite_scores(momentum_df, analyst_df, fundamentals_df)

        top = results.iloc[0]
        if args.dry_run:
            print(
                f"  {date_str}  [DRY RUN] top={top['ticker']:6s} "
                f"composite={top['composite_score']:.1f}"
            )
            continue

        # Store at market-close time for that date (skip if already in DB)
        if not run_exists_for_date(date_str):
            run_at = f"{date_str}T16:00:00"
            run_id = save_run_at(results, run_at, source="backfill")
            print(
                f"  {date_str}  run #{run_id:>4d} saved  "
                f"top={top['ticker']:6s}  composite={top['composite_score']:.1f}",
                end="",
            )
            inserted += 1
        else:
            print(f"  {date_str}  DB exists, regenerating HTML", end="")

        if not args.no_html and not html_exists:
            det_df, over_df = compute_downside_alerts(results, momentum_df)
            export_html(results, det_df, over_df, run_date=trade_date.date())
            print("  [HTML saved]", end="")

        print()

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'-' * 52}")
    if args.dry_run:
        print(f"  DRY RUN complete -- nothing was written to the database.")
    else:
        print(f"  Inserted : {inserted}")
        print(f"  Skipped  : {skipped_exists} (already existed) + {skipped_empty} (no data)")
    print(f"{'-' * 52}\n")

    if not args.dry_run and inserted > 0:
        if not args.no_html:
            print("Regenerating charts dashboard...")
            from charts import generate as generate_charts
            charts_path = generate_charts()
            print(f"Charts saved: {charts_path}\n")
        print(
            "Run python analytics.py to view performance analysis and "
            "factor attribution reports."
        )


if __name__ == "__main__":
    main()
