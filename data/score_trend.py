"""
Score Trend Signal
Computes the linear slope of each ticker's composite score over the last
SCORE_TREND_LOOKBACK runs. A rising slope means the bot's model has been
consistently upgrading that ticker — a useful momentum-of-opinion signal.

Returns a DataFrame with columns: ticker, score_trend_raw
"""
import numpy as np
import pandas as pd

import config
from data.universe import TOP_50_BY_MARKET_CAP
from storage.database import get_ticker_score_history

_SP500_SET = set(TOP_50_BY_MARKET_CAP)


def fetch_score_trend_scores(tickers: list[str]) -> pd.DataFrame:
    """
    For each ticker, fit a linear trend to composite scores over the last
    SCORE_TREND_LOOKBACK runs. Returns the slope (score-points per run).

    Non-S&P-500 tickers (e.g. Finviz gainers) receive slope=0.0 so that
    short-term penny-stock momentum spikes don't inflate the trend signal.
    Tickers with fewer than 3 data points also receive 0.0.
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "score_trend_raw"])

    history = get_ticker_score_history(tickers, config.SCORE_TREND_LOOKBACK)
    if history.empty:
        return pd.DataFrame(
            [{"ticker": t, "score_trend_raw": 0.0} for t in tickers]
        )

    rows = []
    for ticker in tickers:
        if ticker not in _SP500_SET:
            rows.append({"ticker": ticker, "score_trend_raw": 0.0})
            continue

        t_hist = history[history["ticker"] == ticker].sort_values("run_at")
        if len(t_hist) < 3:
            rows.append({"ticker": ticker, "score_trend_raw": 0.0})
            continue

        x = np.arange(len(t_hist), dtype=float)
        y = t_hist["composite"].values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0])
        rows.append({"ticker": ticker, "score_trend_raw": slope})

    return pd.DataFrame(rows)
