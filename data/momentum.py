"""
Factor 2 — Momentum
Returns a DataFrame with columns:
  ticker, momentum_raw, pct_1d, pct_5d, pct_1mo, rsi_14, vol_ratio
Also returns a list of extra tickers surfaced by Finviz top-gainers screener.
"""
import warnings

import pandas as pd
import yfinance as yf

import config
from storage.database import save_data_cache, load_data_cache


def _rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI over the last `period` trading days. Returns 50.0 on failure."""
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]
    if pd.isna(last_gain) or pd.isna(last_loss) or last_loss == 0:
        return 100.0 if last_loss == 0 else 50.0
    return round(100.0 - 100.0 / (1.0 + last_gain / last_loss), 1)


def fetch_momentum_scores(tickers: list[str]) -> pd.DataFrame:
    """
    Batch-download 3-month price/volume history for all tickers and compute:
      - pct_1d, pct_5d, pct_1mo: price return over respective windows
      - momentum_raw: weighted blend of the three returns
      - rsi_14: 14-day Relative Strength Index (display only)
      - vol_ratio: 10-day avg volume / 60-day avg volume (display only)
                   > 1.0 means unusually high recent volume
    """
    if not tickers:
        return pd.DataFrame(columns=[
            "ticker", "momentum_raw", "pct_1d", "pct_5d", "pct_1mo",
            "rsi_14", "vol_ratio",
        ])

    print(f"  Downloading price history for {len(tickers)} tickers…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            period="3mo",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    # yf.download returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"]
        volume = raw["Volume"]
    else:
        close  = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    close  = close.dropna(axis=1, how="all")
    volume = volume.dropna(axis=1, how="all")

    rows = []
    for ticker in close.columns:
        prices = close[ticker].dropna()
        vols   = volume[ticker].dropna() if ticker in volume.columns else pd.Series([], dtype=float)

        if len(prices) < 6:
            continue

        p_now = prices.iloc[-1]
        p_1d  = prices.iloc[-2]  if len(prices) >= 2  else p_now
        p_5d  = prices.iloc[-6]  if len(prices) >= 6  else prices.iloc[0]
        # ~22 trading days ≈ 1 calendar month
        p_1mo = prices.iloc[-22] if len(prices) >= 22 else prices.iloc[0]

        pct_1d  = (p_now - p_1d)  / p_1d  if p_1d  else 0.0
        pct_5d  = (p_now - p_5d)  / p_5d  if p_5d  else 0.0
        pct_1mo = (p_now - p_1mo) / p_1mo if p_1mo else 0.0

        w = config.MOMENTUM_WEIGHTS
        momentum_raw = (pct_1d * w["1d"]) + (pct_5d * w["5d"]) + (pct_1mo * w["1mo"])

        # RSI — neutral 50.0 if insufficient history
        rsi_14 = _rsi(prices, 14)

        # Volume ratio: recent 10-day avg vs 60-day avg
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

    df = pd.DataFrame(rows)
    if df.empty:
        cached = load_data_cache("momentum")
        if not cached.empty:
            return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)
        return df
    save_data_cache("momentum", df)
    return df


def fetch_finviz_gainers() -> list[str]:
    """
    Return a list of top-gaining tickers from Finviz screener.
    Falls back to empty list if the package or network is unavailable.
    """
    try:
        from finvizfinance.screener.overview import Overview
        foverview = Overview()
        foverview.set_filter(filters_dict={"Change": "Up 10%"})
        df = foverview.screener_view(verbose=0)
        if df is not None and "Ticker" in df.columns:
            return df["Ticker"].tolist()
    except Exception as e:
        print(f"  Warning: Finviz gainers unavailable ({e})")
    return []
