"""
Merges momentum, analyst, fundamentals (and optionally score_trend) DataFrames
into a single ranked result.
"""
import pandas as pd

import config
from scoring.normalize import min_max_normalize


def build_composite_scores(
    momentum_df:     pd.DataFrame,
    analyst_df:      pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    score_trend_df:  pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    momentum_df     : columns [ticker, momentum_raw, pct_1d, pct_5d, pct_1mo,
                                       rsi_14, vol_ratio]
    analyst_df      : columns [ticker, analyst_raw, upside_pct, buy_ratio]
    fundamentals_df : columns [ticker, fundamentals_raw, eps_growth,
                                       revenue_growth, forward_pe, sector, beta]
    score_trend_df  : optional columns [ticker, score_trend_raw]
                      When provided, adds a 4th factor weighted by
                      config.SCORE_TREND_WEIGHT, scaling down the other
                      three factors proportionally so weights always sum to 1.0.

    Returns
    -------
    DataFrame sorted by composite score descending, with columns:
    rank, ticker, composite_score, momentum_score, analyst_score,
    fundamentals_score, sector, beta, forward_pe, rsi_14, vol_ratio, notes
    """
    def _raw(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["ticker", col])
        return df[["ticker", col]].copy()

    mom  = _raw(momentum_df,     "momentum_raw")
    ana  = _raw(analyst_df,      "analyst_raw")
    fun  = _raw(fundamentals_df, "fundamentals_raw")

    # Outer join — no ticker is dropped if one source has a gap
    merged = (
        mom.merge(ana, on="ticker", how="outer")
           .merge(fun, on="ticker", how="outer")
    )

    use_trend = score_trend_df is not None and not score_trend_df.empty
    if use_trend:
        merged = merged.merge(
            score_trend_df[["ticker", "score_trend_raw"]], on="ticker", how="outer"
        )

    merged = merged.fillna(0)

    # Normalize each factor to [0, 100]
    merged["momentum_score"]      = min_max_normalize(merged["momentum_raw"])
    merged["analyst_score"]       = min_max_normalize(merged["analyst_raw"])
    merged["fundamentals_score"]  = min_max_normalize(merged["fundamentals_raw"])

    w = config.FACTOR_WEIGHTS
    if use_trend:
        merged["score_trend_score"] = min_max_normalize(merged["score_trend_raw"])
        st_w  = config.SCORE_TREND_WEIGHT
        scale = 1.0 - st_w          # proportionally shrink the other three
        merged["composite_score"] = (
            merged["momentum_score"]     * w["momentum"]     * scale +
            merged["analyst_score"]      * w["analyst"]      * scale +
            merged["fundamentals_score"] * w["fundamentals"] * scale +
            merged["score_trend_score"]  * st_w
        )
    else:
        merged["composite_score"] = (
            merged["momentum_score"]     * w["momentum"]     +
            merged["analyst_score"]      * w["analyst"]      +
            merged["fundamentals_score"] * w["fundamentals"]
        )

    merged = merged.sort_values("composite_score", ascending=False).reset_index(drop=True)
    merged.insert(0, "rank", merged.index + 1)

    # Attach display-only columns from source DataFrames
    display_cols = {}
    if not fundamentals_df.empty:
        for col in ("sector", "beta", "forward_pe", "eps_growth", "revenue_growth"):
            if col in fundamentals_df.columns:
                display_cols[col] = fundamentals_df.set_index("ticker")[col]
    if not momentum_df.empty:
        for col in ("rsi_14", "vol_ratio"):
            if col in momentum_df.columns:
                display_cols[col] = momentum_df.set_index("ticker")[col]

    for col, series in display_cols.items():
        merged[col] = merged["ticker"].map(series)

    merged["sector"]     = merged.get("sector",     pd.Series()).fillna("Unknown")
    merged["beta"]       = merged.get("beta",       pd.Series()).fillna(1.0)
    merged["forward_pe"] = merged.get("forward_pe", pd.Series())
    merged["rsi_14"]     = merged.get("rsi_14",     pd.Series()).fillna(50.0)
    merged["vol_ratio"]  = merged.get("vol_ratio",  pd.Series()).fillna(1.0)

    # Build notes string
    def _notes(row: pd.Series) -> str:
        parts = []
        if not momentum_df.empty and "pct_1mo" in momentum_df.columns:
            r = momentum_df[momentum_df["ticker"] == row["ticker"]]
            if not r.empty:
                parts.append(f"{r.iloc[0]['pct_1mo'] * 100:+.1f}% 1mo")
        if not analyst_df.empty and "upside_pct" in analyst_df.columns:
            r = analyst_df[analyst_df["ticker"] == row["ticker"]]
            if not r.empty:
                parts.append(f"{r.iloc[0]['upside_pct'] * 100:+.1f}% analyst upside")
        if not fundamentals_df.empty:
            r = fundamentals_df[fundamentals_df["ticker"] == row["ticker"]]
            if not r.empty:
                eg = r.iloc[0].get("eps_growth", 0) or 0
                rg = r.iloc[0].get("revenue_growth", 0) or 0
                if eg != 0:
                    parts.append(f"EPS {eg * 100:+.0f}%")
                if rg != 0:
                    parts.append(f"Rev {rg * 100:+.0f}%")
        return "; ".join(parts) if parts else "—"

    merged["notes"] = merged.apply(_notes, axis=1)

    cols = [
        "rank", "ticker", "composite_score",
        "momentum_score", "analyst_score", "fundamentals_score",
        "sector", "beta", "forward_pe", "rsi_14", "vol_ratio",
        "notes",
    ]
    return merged[cols]
