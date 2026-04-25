import pandas as pd
from tabulate import tabulate

import config


def print_results(df: pd.DataFrame) -> None:
    top = df.head(config.TOP_N_RESULTS).copy()

    fmt1 = "{:.1f}".format
    top["composite_score"]    = top["composite_score"].map(fmt1)
    top["momentum_score"]     = top["momentum_score"].map(fmt1)
    top["analyst_score"]      = top["analyst_score"].map(fmt1)
    top["fundamentals_score"] = top["fundamentals_score"].map(fmt1)
    top["rsi_14"]             = top["rsi_14"].map(fmt1)
    top["vol_ratio"]          = top["vol_ratio"].map("{:.2f}x".format)

    display_cols = [
        "rank", "ticker", "composite_score",
        "momentum_score", "analyst_score", "fundamentals_score",
        "sector", "rsi_14", "vol_ratio", "notes",
    ]
    headers = [
        "#", "Ticker", "Composite",
        "Momentum", "Analyst", "Fundamentals",
        "Sector", "RSI", "Vol Ratio", "Notes",
    ]

    print("\n" + "=" * 100)
    print(f"  TOP {config.TOP_N_RESULTS} STOCK RECOMMENDATIONS")
    print("=" * 100)
    print(tabulate(
        top[display_cols],
        headers=headers,
        tablefmt="simple",
        showindex=False,
    ))
    print("=" * 100 + "\n")


def print_alerts(det_df: pd.DataFrame, over_df: pd.DataFrame) -> None:
    """Print deterioration and overextension alerts to the terminal."""

    _SEV_LABEL = {"HIGH": "[HIGH]", "MEDIUM": "[MED] ", "LOW":  "[LOW] "}

    has_det  = not det_df.empty
    has_over = not over_df.empty

    if not has_det and not has_over:
        print("  No downside alerts for today's universe.\n")
        return

    print("=" * 100)
    print("  DOWNSIDE ALERTS")
    print("=" * 100)

    if has_det:
        print("  Deterioration signals:\n")
        for _, r in det_df.iterrows():
            sev   = _SEV_LABEL.get(r["severity"], "     ")
            prev  = f"was avg #{r['prev_avg_rank']}" if r["prev_avg_rank"] is not None else "no history"
            chg   = f", dropped {abs(r['rank_change']):.0f}" if r["rank_change"] is not None else ""
            slope = f"  slope {r['score_slope']:+.2f}" if r["score_slope"] is not None else ""
            print(
                f"  {sev}  {r['ticker']:<6s}  rank #{r['current_rank']:<3d} ({prev}{chg})"
                f"  1mo {r['pct_1mo']:+.1f}%  RSI {r['rsi_14']:.1f}{slope}"
                f"  |  {r['flags']}"
            )
        print()

    if has_over:
        print("  Overextension warnings (top picks with RSI > 75):\n")
        for _, r in over_df.iterrows():
            print(
                f"  [WATCH]  {r['ticker']:<6s}  rank #{r['current_rank']:<3d}"
                f"  RSI {r['rsi_14']:.1f}  1mo {r['pct_1mo']:+.1f}%"
                f"  composite {r['composite']:.1f}"
                f"  |  {r['note']}"
            )
        print()

    print("=" * 100 + "\n")
