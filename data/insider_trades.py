"""
Factor 1 — Insider & Congress trades
Returns a DataFrame with columns: ticker, insider_raw, openinsider_count, congress_count

Both scrapers are wrapped in try/except — if a site is unreachable, that source
returns zeros and the bot continues normally.
"""
import time
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_openinsider_cluster_buys() -> dict[str, int]:
    """
    Scrape OpenInsider cluster-buy screen and return {ticker: event_count}
    for the past INSIDER_LOOKBACK_DAYS days.
    """
    counts: dict[str, int] = defaultdict(int)
    try:
        # General insider purchases screen (last 30 days, all buys, top 300 rows)
        url = (
            "http://openinsider.com/screener?"
            "s=&o=&pl=&ph=&ll=&lh=&fd=30&fdr=&td=0&tdr=&fdlyl=&fdlyh=&"
            "daysago=&xp=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&"
            "grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&"
            "sortcol=0&cnt=300&action=1"
        )
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", {"class": "tinytable"})
        if table:
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) >= 4:
                    ticker = cells[3].get_text(strip=True)
                    if ticker:
                        counts[ticker.upper()] += 1
    except Exception as e:
        print(f"  Warning: OpenInsider scrape failed ({e})")
    return dict(counts)


def fetch_congress_trades(tickers: list[str]) -> dict[str, int]:
    """
    Fetch Congress purchase events from QuiverQuant's public page.
    Returns {ticker: purchase_count} for trades within INSIDER_LOOKBACK_DAYS.
    """
    counts: dict[str, int] = defaultdict(int)
    cutoff = date.today() - timedelta(days=config.INSIDER_LOOKBACK_DAYS)
    try:
        url = "https://www.quiverquant.com/sources/congresstrading"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table")
        if table:
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                try:
                    ticker     = cells[1].get_text(strip=True).upper()
                    trade_type = cells[3].get_text(strip=True).lower()
                    date_str   = cells[2].get_text(strip=True)
                    trade_date = date.fromisoformat(date_str)
                    if "purchase" in trade_type and trade_date >= cutoff:
                        counts[ticker] += 1
                except Exception:
                    continue
    except Exception as e:
        print(f"  Warning: QuiverQuant scrape failed ({e})")
    return dict(counts)


def fetch_insider_scores(tickers: list[str]) -> pd.DataFrame:
    """
    Combine OpenInsider and Congress signals into a single DataFrame.
    """
    print("  Fetching insider / Congress trade data…")
    oi_counts  = fetch_openinsider_cluster_buys()
    cong_counts = fetch_congress_trades(tickers)

    rows = []
    for ticker in tickers:
        oi   = oi_counts.get(ticker, 0)
        cong = cong_counts.get(ticker, 0)
        insider_raw = (oi * 0.6) + (cong * 0.4)
        rows.append({
            "ticker":            ticker,
            "insider_raw":       insider_raw,
            "openinsider_count": oi,
            "congress_count":    cong,
        })

    return pd.DataFrame(rows)
