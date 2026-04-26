"""
Factor 4 — Fundamentals
Returns a DataFrame with columns:
  ticker, fundamentals_raw, forward_pe, eps_growth, revenue_growth, sector, beta
"""
import time

import pandas as pd
import yfinance as yf

import config
from storage.database import save_data_cache, load_data_cache


def fetch_fundamentals_scores(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch fundamental metrics per ticker via yfinance and compute a
    fundamentals_raw score. Higher score = better quality/value.

    Metrics used in scoring:
      - eps_growth:     YoY earnings growth (higher = better)
      - revenue_growth: YoY revenue growth  (higher = better)
      - pe_value:       1 / forwardPE if PE > 0, else 0 (lower P/E = better value)

    Extra columns returned for display only (not in score):
      - sector, beta, forward_pe (raw)
    """
    rows = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        print(f"  Fundamentals {i}/{total}: {ticker}", end="\r")
        try:
            info = yf.Ticker(ticker).info

            eps_growth     = float(info.get("earningsGrowth")  or 0.0)
            revenue_growth = float(info.get("revenueGrowth")   or 0.0)
            forward_pe_raw = info.get("forwardPE")
            sector         = info.get("sector") or "Unknown"
            beta           = float(info.get("beta") or 1.0)

            # Reciprocal of P/E: lower P/E → higher pe_value score.
            # Only meaningful for profitable companies (PE > 0).
            if forward_pe_raw and forward_pe_raw > 0:
                pe_value      = 1.0 / forward_pe_raw
                forward_pe    = round(float(forward_pe_raw), 1)
            else:
                pe_value   = 0.0
                forward_pe = None

            w = config.FUNDAMENTALS_WEIGHTS
            fundamentals_raw = (
                eps_growth     * w["eps_growth"] +
                revenue_growth * w["revenue_growth"] +
                pe_value       * w["pe_value"]
            )

            rows.append({
                "ticker":           ticker,
                "fundamentals_raw": fundamentals_raw,
                "eps_growth":       eps_growth,
                "revenue_growth":   revenue_growth,
                "forward_pe":       forward_pe,
                "sector":           sector,
                "beta":             beta,
            })

        except Exception:
            rows.append({
                "ticker":           ticker,
                "fundamentals_raw": 0.0,
                "eps_growth":       0.0,
                "revenue_growth":   0.0,
                "forward_pe":       None,
                "sector":           "Unknown",
                "beta":             1.0,
            })

        time.sleep(config.REQUEST_DELAY_SECONDS)

    print()
    df = pd.DataFrame(rows)
    if not df.empty and (df["fundamentals_raw"] == 0).all():
        cached = load_data_cache("fundamentals")
        if not cached.empty:
            return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)
    elif not df.empty and not (df["fundamentals_raw"] == 0).all():
        save_data_cache("fundamentals", df)
    return df
