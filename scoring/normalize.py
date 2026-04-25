import pandas as pd


def min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Scale a Series to [0, 100] using min-max normalization.
    Returns 50.0 for all values if the range is zero (all identical).
    NaN values are filled with 0 before normalization.
    """
    s = series.fillna(0)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100
