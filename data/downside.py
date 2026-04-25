"""
Downside Alerts
Scans the full scored universe for tickers showing deterioration or overextension.

Two categories of alert are produced:

  DETERIORATION FLAGS (stock is falling or the model is abandoning it)
    PRICE DROP      — down 5%+ over the last month
    OVERSOLD RSI    — RSI-14 below 35 (sustained selling pressure)
    MODEL DOWNGRADE — composite score trending down over recent runs
    RANK COLLAPSE   — was a top-15 pick historically, dropped 8+ ranks

  OVEREXTENSION FLAGS (top pick that may be topping out)
    OVERBOUGHT      — RSI-14 above 75 for a current top-15 pick

Severity is assigned by how many deterioration flags are present:
  HIGH   — 3+ flags
  MEDIUM — 2 flags
  LOW    — 1 flag (only surfaced for notable tickers)
"""
import pandas as pd

from storage.database import get_ticker_score_history

# --- Deterioration thresholds ---
_PRICE_DROP_THRESHOLD = -0.05   # pct_1mo < -5%
_RSI_OVERSOLD         = 35.0
_SLOPE_DOWNGRADE      = -0.5    # score-points per run
_RANK_WAS_GOOD        = 15      # avg historical rank must have been <= this
_RANK_NOW_BAD         = 20      # current rank must be >= this
_RANK_DROP_MIN        = 8       # must have dropped at least this many places

# --- Overextension threshold ---
_RSI_OVERBOUGHT       = 75.0
_OVERBOUGHT_MAX_RANK  = 15      # only flag if currently ranked this high or better

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def compute_downside_alerts(
    results_df:     pd.DataFrame,
    momentum_df:    pd.DataFrame,
    score_trend_df: pd.DataFrame | None = None,
    lookback_runs:  int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan all tickers for downside and overextension signals.

    Parameters
    ----------
    results_df     : full scored universe (all ranks, not just top-N)
    momentum_df    : raw momentum data including pct_1mo
    score_trend_df : optional [ticker, score_trend_raw]
    lookback_runs  : how many past runs to use for rank comparison

    Returns
    -------
    (deterioration_df, overextended_df) — either may be an empty DataFrame
    """
    tickers = results_df["ticker"].tolist()

    rank_history = _avg_rank_history(tickers, lookback_runs)

    mom_map = (
        momentum_df.set_index("ticker")["pct_1mo"].to_dict()
        if not momentum_df.empty and "pct_1mo" in momentum_df.columns
        else {}
    )
    slope_map = (
        score_trend_df.set_index("ticker")["score_trend_raw"].to_dict()
        if score_trend_df is not None and not score_trend_df.empty
        else {}
    )

    det_rows  = []   # deterioration
    over_rows = []   # overextension

    for _, r in results_df.iterrows():
        ticker       = r["ticker"]
        current_rank = int(r["rank"])
        rsi_14       = float(r.get("rsi_14", 50.0) or 50.0)
        pct_1mo      = float(mom_map.get(ticker, 0.0))
        score_slope  = slope_map.get(ticker)
        composite    = float(r.get("composite_score", 0.0))

        avg_rank    = rank_history.get(ticker)
        rank_change = (current_rank - avg_rank) if avg_rank is not None else None

        # --- Deterioration flags ---
        flags = []

        if pct_1mo < _PRICE_DROP_THRESHOLD:
            flags.append("PRICE DROP")

        if rsi_14 < _RSI_OVERSOLD:
            flags.append("OVERSOLD RSI")

        if score_slope is not None and score_slope < _SLOPE_DOWNGRADE:
            flags.append("MODEL DOWNGRADE")

        if (
            avg_rank is not None
            and avg_rank <= _RANK_WAS_GOOD
            and current_rank >= _RANK_NOW_BAD
            and rank_change is not None
            and rank_change <= -_RANK_DROP_MIN
        ):
            flags.append("RANK COLLAPSE")

        if flags:
            n = len(flags)
            if n >= 3:
                severity = "HIGH"
            elif n == 2:
                severity = "MEDIUM"
            else:
                # LOW — skip obscure tickers with a single weak signal
                if current_rank > 25 and (avg_rank is None or avg_rank > 20):
                    flags = []
                else:
                    severity = "LOW"

        if flags:
            det_rows.append({
                "ticker":        ticker,
                "severity":      severity,
                "current_rank":  current_rank,
                "prev_avg_rank": round(avg_rank, 1) if avg_rank is not None else None,
                "rank_change":   round(rank_change, 1) if rank_change is not None else None,
                "pct_1mo":       round(pct_1mo * 100, 1),
                "rsi_14":        rsi_14,
                "score_slope":   round(score_slope, 2) if score_slope is not None else None,
                "flags":         ", ".join(flags),
            })

        # --- Overextension flags ---
        if rsi_14 > _RSI_OVERBOUGHT and current_rank <= _OVERBOUGHT_MAX_RANK:
            over_rows.append({
                "ticker":       ticker,
                "current_rank": current_rank,
                "rsi_14":       rsi_14,
                "pct_1mo":      round(pct_1mo * 100, 1),
                "composite":    round(composite, 1),
                "note":         "RSI overbought - may be near a short-term top",
            })

    det_df = pd.DataFrame()
    if det_rows:
        det_df = pd.DataFrame(det_rows)
        det_df["_ord"] = det_df["severity"].map(_SEVERITY_ORDER)
        det_df = (
            det_df.sort_values(["_ord", "current_rank"])
            .drop(columns=["_ord"])
            .reset_index(drop=True)
        )

    over_df = pd.DataFrame(over_rows) if over_rows else pd.DataFrame()

    return det_df, over_df


def _avg_rank_history(tickers: list[str], n_runs: int) -> dict[str, float]:
    """Return {ticker: avg_rank} over the last n_runs runs (min 3 data points)."""
    hist = get_ticker_score_history(tickers, n_runs)
    if hist.empty or "rank" not in hist.columns:
        return {}
    result = {}
    for ticker, grp in hist.groupby("ticker"):
        if len(grp) >= 3:
            result[ticker] = float(grp["rank"].mean())
    return result
