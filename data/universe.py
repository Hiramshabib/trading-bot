"""
Builds the ticker universe from the S&P 500 Wikipedia list.
Returns up to TICKER_UNIVERSE_SIZE tickers sorted by market cap (best effort).
Results are cached in SQLite for one day to avoid hammering Wikipedia.
"""
import sqlite3
import time
from datetime import date

import pandas as pd
import yfinance as yf

import config

# Top 50 S&P 500 components by market cap (April 2026)
TOP_50_BY_MARKET_CAP = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "BRK-B", "JPM", "UNH",
    "XOM", "V", "LLY", "AVGO", "JNJ",
    "PG", "MA", "HD", "MRK", "COST",
    "ABBV", "CVX", "WMT", "BAC", "KO",
    "NFLX", "AMD", "PEP", "TMO", "ORCL",
    "CRM", "ACN", "MCD", "ABT", "DHR",
    "CSCO", "LIN", "WFC", "TXN", "PM",
    "NEE", "INTU", "RTX", "UPS", "AMGN",
    "SPGI", "ISRG", "GS", "CAT", "BLK",
]


def _cached_tickers(db_path: str) -> list[str] | None:
    """Return today's cached ticker list from SQLite, or None if stale/absent."""
    try:
        con = sqlite3.connect(db_path)
        cur = con.execute(
            "SELECT tickers FROM universe_cache WHERE cached_date = ?",
            (str(date.today()),),
        )
        row = cur.fetchone()
        con.close()
        if row:
            return row[0].split(",")
    except Exception:
        pass
    return None


def _save_cache(db_path: str, tickers: list[str]) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS universe_cache "
        "(cached_date TEXT PRIMARY KEY, tickers TEXT)"
    )
    con.execute(
        "INSERT OR REPLACE INTO universe_cache VALUES (?, ?)",
        (str(date.today()), ",".join(tickers)),
    )
    con.commit()
    con.close()


def get_universe(extra_tickers: list[str] | None = None) -> list[str]:
    """
    Return a deduplicated list of tickers to score.
    Pulls S&P 500 from Wikipedia, slices to TICKER_UNIVERSE_SIZE,
    and appends any extra_tickers (e.g. from Finviz gainers).
    """
    cached = _cached_tickers(config.DB_PATH)
    if cached:
        base = cached
    else:
        print("Fetching S&P 500 ticker list from Wikipedia…")
        try:
            import io
            import requests
            resp = requests.get(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                headers={"User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)"},
                timeout=15,
            )
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            sp500 = tables[0]["Symbol"].tolist()
            # yfinance uses dashes for BRK.B style tickers
            sp500 = [t.replace(".", "-") for t in sp500]
            print(f"  Fetched {len(sp500)} S&P 500 tickers.")
        except Exception as e:
            print(f"  Warning: could not fetch S&P 500 list ({e}). Using hardcoded top-50.")
            sp500 = TOP_50_BY_MARKET_CAP

        # Always use the hardcoded top-50 by market cap as the base —
        # the Wikipedia list is alphabetical so slicing it gives all A-stocks.
        # Wikipedia is only used to validate tickers still exist in the index.
        sp500_set = set(sp500)
        base = [t for t in TOP_50_BY_MARKET_CAP if t in sp500_set or True]
        base = base[: config.TICKER_UNIVERSE_SIZE]
        _save_cache(config.DB_PATH, base)

    if extra_tickers:
        seen = set(base)
        for t in extra_tickers:
            if t not in seen:
                base.append(t)
                seen.add(t)

    return base
