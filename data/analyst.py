"""
Factor 3 — Analyst predictions
Returns a DataFrame with columns: ticker, analyst_raw, upside_pct, buy_ratio
"""
import time

import pandas as pd
import yfinance as yf

import config
from storage.database import save_data_cache, load_data_cache


def fetch_analyst_scores(tickers: list[str]) -> pd.DataFrame:
    rows = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        print(f"  Analyst data {i}/{total}: {ticker}", end="\r")
        try:
            t = yf.Ticker(ticker)

            # Price target upside
            targets = t.analyst_price_targets  # dict: currentPrice, mean, high, low
            upside_pct = 0.0
            if targets and targets.get("mean") and targets.get("current"):
                current = targets["current"]
                mean    = targets["mean"]
                if current and current > 0:
                    upside_pct = (mean - current) / current

            # Buy ratio from recommendations summary
            buy_ratio = 0.0
            recs = t.recommendations_summary
            if recs is not None and not recs.empty:
                latest = recs.iloc[0]
                strong_buy = latest.get("strongBuy", 0) or 0
                buy        = latest.get("buy", 0) or 0
                hold       = latest.get("hold", 0) or 0
                sell       = latest.get("sell", 0) or 0
                strong_sell = latest.get("strongSell", 0) or 0
                total_recs  = strong_buy + buy + hold + sell + strong_sell
                if total_recs > 0:
                    buy_ratio = (strong_buy + buy) / total_recs

            analyst_raw = (upside_pct * 0.60) + (buy_ratio * 0.40)

            rows.append({
                "ticker":      ticker,
                "analyst_raw": analyst_raw,
                "upside_pct":  upside_pct,
                "buy_ratio":   buy_ratio,
            })

        except Exception:
            rows.append({
                "ticker":      ticker,
                "analyst_raw": 0.0,
                "upside_pct":  0.0,
                "buy_ratio":   0.0,
            })

        time.sleep(config.REQUEST_DELAY_SECONDS)

    print()  # newline after progress line
    df = pd.DataFrame(rows)
    if not df.empty and (df["analyst_raw"] == 0).all():
        cached = load_data_cache("analyst")
        if not cached.empty:
            return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)
    elif not df.empty and not (df["analyst_raw"] == 0).all():
        save_data_cache("analyst", df)
    return df
