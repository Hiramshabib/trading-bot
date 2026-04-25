"""
SQLite persistence: one row per run in `runs`, one row per ticker in `scores`.
"""
import sqlite3
from datetime import datetime

import pandas as pd

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at     TEXT NOT NULL,
    top_ticker TEXT NOT NULL,
    top_score  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER REFERENCES runs(run_id),
    ticker              TEXT NOT NULL,
    rank                INTEGER,
    composite           REAL,
    momentum_score      REAL,
    analyst_score       REAL,
    fundamentals_score  REAL,
    sector              TEXT,
    beta                REAL,
    forward_pe          REAL,
    rsi_14              REAL,
    vol_ratio           REAL,
    notes               TEXT
);
"""


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(config.DB_PATH)


# Columns added after the initial schema — applied via ALTER TABLE if missing.
_MIGRATIONS = [
    "ALTER TABLE scores ADD COLUMN fundamentals_score REAL",
    "ALTER TABLE scores ADD COLUMN sector             TEXT",
    "ALTER TABLE scores ADD COLUMN beta               REAL",
    "ALTER TABLE scores ADD COLUMN forward_pe         REAL",
    "ALTER TABLE scores ADD COLUMN rsi_14             REAL",
    "ALTER TABLE scores ADD COLUMN vol_ratio          REAL",
    # Remove insider_score by dropping is unsupported in SQLite; just ignore it.
    "ALTER TABLE runs ADD COLUMN source TEXT DEFAULT 'live'",
]


def init_db() -> None:
    con = _connect()
    con.executescript(_SCHEMA)
    # Apply any migrations that haven't been run yet (ignore if column already exists)
    for stmt in _MIGRATIONS:
        try:
            con.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    con.commit()
    con.close()


def _insert_scores(con: sqlite3.Connection, run_id: int, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        con.execute(
            "INSERT INTO scores (run_id, ticker, rank, composite, "
            "momentum_score, analyst_score, fundamentals_score, "
            "sector, beta, forward_pe, rsi_14, vol_ratio, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                row["ticker"],
                int(row["rank"]),
                float(row["composite_score"]),
                float(row["momentum_score"]),
                float(row["analyst_score"]),
                float(row["fundamentals_score"]),
                row.get("sector", "Unknown"),
                float(row.get("beta", 1.0) or 1.0),
                float(row["forward_pe"]) if row.get("forward_pe") is not None else None,
                float(row.get("rsi_14", 50.0) or 50.0),
                float(row.get("vol_ratio", 1.0) or 1.0),
                row["notes"],
            ),
        )


def save_run(df: pd.DataFrame) -> int:
    """Insert a live run record and all its scores. Returns the run_id."""
    init_db()
    top = df.iloc[0]
    con = _connect()
    cur = con.execute(
        "INSERT INTO runs (run_at, top_ticker, top_score, source) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), top["ticker"], float(top["composite_score"]), "live"),
    )
    run_id = cur.lastrowid
    _insert_scores(con, run_id, df)
    con.commit()
    con.close()
    return run_id


def save_run_at(df: pd.DataFrame, run_at_iso: str, source: str = "backfill") -> int:
    """Insert a run with a specific timestamp. Returns the run_id."""
    init_db()
    top = df.iloc[0]
    con = _connect()
    cur = con.execute(
        "INSERT INTO runs (run_at, top_ticker, top_score, source) VALUES (?, ?, ?, ?)",
        (run_at_iso, top["ticker"], float(top["composite_score"]), source),
    )
    run_id = cur.lastrowid
    _insert_scores(con, run_id, df)
    con.commit()
    con.close()
    return run_id


def run_exists_for_date(date_str: str) -> bool:
    """Return True if any run (live or backfill) was already saved for the given date (YYYY-MM-DD)."""
    init_db()
    con = _connect()
    cur = con.execute(
        "SELECT COUNT(*) FROM runs WHERE run_at LIKE ?", (f"{date_str}%",)
    )
    count = cur.fetchone()[0]
    con.close()
    return count > 0


def get_run_count() -> int:
    """Return total number of runs stored in the database."""
    init_db()
    con = _connect()
    cur = con.execute("SELECT COUNT(*) FROM runs")
    count = cur.fetchone()[0]
    con.close()
    return count


def get_recent_runs(n: int = 5) -> pd.DataFrame:
    init_db()
    con = _connect()
    df = pd.read_sql_query(
        "SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", con, params=(n,)
    )
    con.close()
    return df


def get_ticker_score_history(tickers: list[str], n_runs: int = 10) -> pd.DataFrame:
    """
    Return the last n_runs composite scores for each ticker in the list,
    sorted oldest-first within each ticker so slopes can be computed correctly.
    Columns: ticker, composite, run_at
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "composite", "run_at"])
    init_db()
    con = _connect()
    placeholders = ",".join("?" * len(tickers))
    # Subquery grabs the most recent n_runs run_ids, then we filter scores to those
    df = pd.read_sql_query(
        f"""
        SELECT s.ticker, s.composite, s.rank, r.run_at
        FROM scores s
        JOIN runs r ON s.run_id = r.run_id
        WHERE s.ticker IN ({placeholders})
          AND r.run_id IN (
              SELECT run_id FROM runs ORDER BY run_at DESC LIMIT ?
          )
        ORDER BY r.run_at ASC
        """,
        con,
        params=(*tickers, n_runs),
    )
    con.close()
    return df


def get_all_scores_with_dates() -> pd.DataFrame:
    """
    Return all scores joined with their run metadata.
    Columns: run_id, run_at, source, ticker, rank, composite,
             momentum_score, analyst_score, fundamentals_score,
             sector, beta, forward_pe, rsi_14, vol_ratio
    """
    init_db()
    con = _connect()
    df = pd.read_sql_query(
        """
        SELECT r.run_id, r.run_at, r.source,
               s.ticker, s.rank, s.composite,
               s.momentum_score, s.analyst_score, s.fundamentals_score,
               s.sector, s.beta, s.forward_pe, s.rsi_14, s.vol_ratio
        FROM scores s
        JOIN runs r ON s.run_id = r.run_id
        ORDER BY r.run_at ASC, s.rank ASC
        """,
        con,
    )
    con.close()
    return df
