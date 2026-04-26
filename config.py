import os

# --- Universe ---
TICKER_UNIVERSE_SIZE = 50        # top-N S&P 500 tickers to score

# --- Insider / Congress ---
INSIDER_LOOKBACK_DAYS = 30

# --- Momentum sub-weights (must sum to 1.0) ---
MOMENTUM_WEIGHTS = {"1d": 0.20, "5d": 0.30, "1mo": 0.50}

# --- Fundamentals sub-weights (must sum to 1.0) ---
FUNDAMENTALS_WEIGHTS = {
    "eps_growth":     0.40,   # earnings quality
    "revenue_growth": 0.35,   # top-line growth
    "pe_value":       0.25,   # valuation (lower P/E = higher value)
}

# --- Factor weights (must sum to 1.0) ---
# Updated 2026-04-15 via analytics.py --suggest-weights (38 runs, 1200 pairs):
#   momentum corr=-0.025, analyst corr=-0.122, fundamentals corr=+0.078
#   Analyst signal is selecting losers in the current regime (high-analyst picks
#   averaged -0.94% vs -0.07% for low-analyst picks). Fundamentals is the only
#   positive predictor. Weights blended 50% data-driven / 50% prior.
#   Original: momentum=0.40, analyst=0.30, fundamentals=0.30
FACTOR_WEIGHTS = {
    "momentum":     0.35,
    "analyst":      0.15,
    "fundamentals": 0.50,
}

# --- Output ---
TOP_N_RESULTS = 20
REPORTS_DIR = "reports"

# --- Storage ---
DB_PATH = "trading_bot.db"

# --- Rate limiting ---
REQUEST_DELAY_SECONDS = 1.5      # pause between per-ticker API calls

# --- Score trend signal ---
# Minimum number of historical runs required before score_trend activates.
# After running backfill.py this will be satisfied immediately.
SCORE_TREND_LOOKBACK = 10        # number of past runs used to compute slope
SCORE_TREND_WEIGHT   = 0.10      # portion of composite score allocated to trend signal
                                  # the remaining 0.90 is split proportionally among
                                  # momentum / analyst / fundamentals

# --- Optional: Alpha Vantage (leave empty to skip) ---
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")

# --- Optional: Anthropic (leave empty to skip AI summary) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
