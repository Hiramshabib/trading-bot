"""
Charts dashboard -- generates an interactive HTML page with Chart.js charts
drawn directly from the SQLite database.  Charts are broken out per calendar
month; the default view is the current month-to-date (MTD).

Usage:
    python charts.py
    python charts.py --out reports/custom_charts.html
    python charts.py --top-n 10   # tickers shown in trend/factor charts

Outputs: reports/YYYY-MM-DD_charts.html  (opens in any browser, no server needed)

Importable API (called by main.py):
    from charts import generate
    generate(top_n=10)
"""
import argparse
import json
import os
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

import config
from storage.database import get_all_scores_with_dates, init_db


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_all() -> pd.DataFrame:
    init_db()
    df = get_all_scores_with_dates()
    df["run_date"] = pd.to_datetime(df["run_at"], format="ISO8601").dt.date.astype(str)
    df["month"] = df["run_date"].str[:7]   # "YYYY-MM"
    return df


def _top_tickers_by_frequency(df: pd.DataFrame, top_n: int) -> list[str]:
    top10 = df[df["rank"] <= 10]["ticker"]
    counts = Counter(top10)
    return [t for t, _ in counts.most_common(top_n)]


def _trend_data(df: pd.DataFrame, tickers: list[str], dates: list[str]) -> list[dict]:
    sub = df[df["ticker"].isin(tickers)][["ticker", "run_date", "composite"]]
    by_ticker: dict[str, dict[str, float | None]] = {t: {} for t in tickers}
    for _, row in sub.iterrows():
        by_ticker[row["ticker"]][row["run_date"]] = (
            round(row["composite"], 2) if pd.notna(row["composite"]) else None
        )
    datasets = []
    for i, ticker in enumerate(tickers):
        color = _PALETTE[i % len(_PALETTE)]
        datasets.append({
            "label": ticker,
            "data": [by_ticker[ticker].get(d) for d in dates],
            "borderColor": color,
            "backgroundColor": color + "33",
            "tension": 0.3,
            "spanGaps": True,
            "pointRadius": 3,
        })
    return datasets


def _factor_averages(df: pd.DataFrame, tickers: list[str]) -> tuple[list[str], list[dict]]:
    rows = []
    for ticker in tickers:
        sub = df[df["ticker"] == ticker]
        rows.append({
            "ticker":       ticker,
            "momentum":     round(sub["momentum_score"].mean(), 1),
            "analyst":      round(sub["analyst_score"].mean(), 1),
            "fundamentals": round(sub["fundamentals_score"].mean(), 1),
        })
    labels = [r["ticker"] for r in rows]
    datasets = [
        {"label": "Momentum",     "data": [r["momentum"]     for r in rows], "backgroundColor": "#4e9af1"},
        {"label": "Analyst",      "data": [r["analyst"]      for r in rows], "backgroundColor": "#f97316"},
        {"label": "Fundamentals", "data": [r["fundamentals"] for r in rows], "backgroundColor": "#22c55e"},
    ]
    return labels, datasets


def _sector_distribution(df: pd.DataFrame) -> dict:
    latest = df[df["run_date"] == df["run_date"].max()]
    counts = latest["sector"].fillna("Unknown").value_counts()
    return {"labels": counts.index.tolist(), "values": counts.values.tolist()}


def _score_histogram(df: pd.DataFrame, bins: int = 10) -> dict:
    scores = df["composite"].dropna().values
    if len(scores) == 0:
        return {"labels": [], "values": []}
    counts, edges = np.histogram(scores, bins=bins, range=(0, 100))
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges) - 1)]
    return {"labels": labels, "values": counts.tolist()}


def _top_pick_history(df: pd.DataFrame) -> dict:
    top1 = df[df["rank"] == 1].sort_values("run_date")
    return {
        "dates":   top1["run_date"].tolist(),
        "tickers": top1["ticker"].tolist(),
        "scores":  top1["composite"].round(2).tolist(),
    }


def _rsi_snapshot(df: pd.DataFrame) -> dict:
    latest = df[df["run_date"] == df["run_date"].max()].sort_values("rank").head(20)
    rsi_vals = latest["rsi_14"].round(1).tolist()
    colors = [
        "#ef4444" if (v or 0) > 70 else "#22c55e" if (v or 0) < 30 else "#4e9af1"
        for v in rsi_vals
    ]
    return {"tickers": latest["ticker"].tolist(), "rsi": rsi_vals, "colors": colors}


def _rank_stability(df: pd.DataFrame, tickers: list[str]) -> list[dict]:
    rows = []
    for ticker in tickers:
        sub = df[df["ticker"] == ticker]["rank"].dropna()
        rows.append({
            "label": ticker,
            "x": round(sub.mean(), 1),
            "y": round(sub.std(), 1) if len(sub) > 1 else 0.0,
        })
    rows.sort(key=lambda r: r["x"])
    return rows


def _build_month_data(df: pd.DataFrame, month: str, top_n: int) -> dict:
    mdf = df[df["month"] == month]
    dates = sorted(mdf["run_date"].unique().tolist())
    tickers = _top_tickers_by_frequency(mdf, top_n)

    factor_labels, factor_datasets = _factor_averages(mdf, tickers)
    sector = _sector_distribution(mdf)
    top_pick = _top_pick_history(mdf)
    rsi = _rsi_snapshot(mdf)
    stability = _rank_stability(mdf, tickers)

    return {
        "dates":            dates,
        "trend_datasets":   _trend_data(mdf, tickers, dates),
        "factor_labels":    factor_labels,
        "factor_datasets":  factor_datasets,
        "sector":           sector,
        "sector_colors":    _PALETTE[:len(sector["labels"])],
        "histo":            _score_histogram(mdf),
        "top_pick":         top_pick,
        "rsi":              rsi,
        "stability":        stability,
        "run_count":        int(mdf["run_date"].nunique()),
        "total_tickers":    int(mdf["ticker"].nunique()),
        "latest_date":      dates[-1] if dates else "",
        "latest_top_pick":  top_pick["tickers"][-1] if top_pick["tickers"] else "—",
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4e9af1", "#f97316", "#22c55e", "#a855f7", "#f43f5e",
    "#06b6d4", "#eab308", "#10b981", "#ec4899", "#8b5cf6",
    "#3b82f6", "#ef4444", "#84cc16", "#f59e0b", "#14b8a6",
]


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so JSON serialisation succeeds."""
    if isinstance(obj, float):
        import math
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _safe_json(obj) -> str:
    return json.dumps(_sanitize(obj), allow_nan=False)


def _month_label(month_str: str, current_month: str) -> str:
    """E.g. '2026-04' → 'Apr 2026 (MTD)' for current month."""
    dt = date(int(month_str[:4]), int(month_str[5:7]), 1)
    label = dt.strftime("%b %Y")
    if month_str == current_month:
        label += " (MTD)"
    return label


def _build_html(all_data: dict[str, dict], top_n: int, today_str: str, current_month: str) -> str:
    months_newest_first = sorted(all_data.keys(), reverse=True)
    tab_labels = {m: _month_label(m, current_month) for m in months_newest_first}

    # Summary stats come from the current month (or most recent if MTD has no data)
    summary_month = current_month if current_month in all_data else months_newest_first[0]
    s = all_data[summary_month]
    overall_runs = sum(v["run_count"] for v in all_data.values())
    overall_tickers = max(v["total_tickers"] for v in all_data.values())

    tabs_html = "\n".join(
        f'  <button class="tab-btn{" active" if m == months_newest_first[0] else ""}" '
        f'data-month="{m}">{tab_labels[m]}</button>'
        for m in months_newest_first
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot Charts &mdash; {today_str}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:      #0f1117;
    --surface: #1a1d27;
    --border:  #2d3045;
    --text:    #e2e8f0;
    --muted:   #94a3b8;
    --accent:  #4e9af1;
    --active:  #1e3a5f;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    padding: 24px 16px 48px;
  }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 20px; font-size: 0.9rem; }}
  .stats-bar {{
    display: flex; gap: 16px; flex-wrap: wrap;
    margin-bottom: 24px;
  }}
  .stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 18px;
    min-width: 110px;
  }}
  .stat-label {{ color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em; }}
  .stat-value {{ font-size: 1.4rem; font-weight: 700; color: var(--accent); }}
  .tab-bar {{
    display: flex; gap: 8px; flex-wrap: wrap;
    margin-bottom: 24px;
  }}
  .tab-btn {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.85rem;
    cursor: pointer;
    transition: background .15s, color .15s, border-color .15s;
  }}
  .tab-btn:hover {{ border-color: var(--accent); color: var(--text); }}
  .tab-btn.active {{
    background: var(--active);
    border-color: var(--accent);
    color: var(--accent);
    font-weight: 600;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
    gap: 20px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 20px 14px;
  }}
  .card.wide {{ grid-column: 1 / -1; }}
  .card h2 {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 14px;
  }}
  .chart-wrap {{ position: relative; height: 300px; }}
  .chart-wrap.tall {{ height: 380px; }}
  .note {{ margin-top: 8px; font-size: 0.73rem; color: var(--muted); }}
  .rsi-legend {{ display: flex; gap: 16px; margin-top: 8px; font-size: 0.76rem; }}
  .rsi-legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>

<h1>Trading Bot &mdash; Charts Dashboard</h1>
<p class="subtitle">Generated {today_str} &nbsp;|&nbsp; {overall_runs} total runs &nbsp;|&nbsp; {overall_tickers} unique tickers tracked</p>

<div class="stats-bar">
  <div class="stat" id="stat-runs"><div class="stat-label">Runs (Period)</div><div class="stat-value">—</div></div>
  <div class="stat" id="stat-tickers"><div class="stat-label">Tickers</div><div class="stat-value">—</div></div>
  <div class="stat" id="stat-range"><div class="stat-label">Date Range</div><div class="stat-value" style="font-size:0.95rem">—</div></div>
  <div class="stat" id="stat-top"><div class="stat-label">Latest Top Pick</div><div class="stat-value" style="color:#f97316">—</div></div>
</div>

<div class="tab-bar">
{tabs_html}
</div>

<div class="grid">
  <div class="card wide">
    <h2 id="trend-title">Composite Score Trends</h2>
    <div class="chart-wrap tall"><canvas id="trendChart"></canvas></div>
    <p class="note">Only dates where each ticker appeared in the run are plotted. Gaps = absent from universe.</p>
  </div>
  <div class="card">
    <h2>#1 Pick &mdash; Composite Score Per Run</h2>
    <div class="chart-wrap"><canvas id="topPickChart"></canvas></div>
  </div>
  <div class="card">
    <h2 id="factor-title">Avg Factor Scores</h2>
    <div class="chart-wrap"><canvas id="factorChart"></canvas></div>
    <p class="note">Momentum / Analyst / Fundamentals avg across period (each 0&ndash;100).</p>
  </div>
  <div class="card">
    <h2>RSI Snapshot &mdash; Latest Run in Period (Top 20)</h2>
    <div class="chart-wrap"><canvas id="rsiChart"></canvas></div>
    <div class="rsi-legend">
      <span><span class="dot" style="background:#ef4444"></span>Overbought &gt;70</span>
      <span><span class="dot" style="background:#4e9af1"></span>Neutral</span>
      <span><span class="dot" style="background:#22c55e"></span>Oversold &lt;30</span>
    </div>
  </div>
  <div class="card">
    <h2>Score Distribution &mdash; Period</h2>
    <div class="chart-wrap"><canvas id="histoChart"></canvas></div>
  </div>
  <div class="card">
    <h2>Sector Distribution &mdash; Latest Run in Period</h2>
    <div class="chart-wrap"><canvas id="sectorChart"></canvas></div>
  </div>
  <div class="card wide">
    <h2>Rank Stability &mdash; Avg Rank vs Std Dev</h2>
    <div class="chart-wrap"><canvas id="stabilityChart"></canvas></div>
    <p class="note">Best tickers: bottom-left (low avg rank, low variance). Computed across all runs in selected period.</p>
  </div>
</div>

<script>
Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "#2d3045";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

const ALL_DATA = {_safe_json(all_data)};
const PALETTE  = {_safe_json(_PALETTE)};

// ---- Chart instances ----
let trendChart, topPickChart, factorChart, rsiChart, histoChart, sectorChart, stabilityChart;

function makeChart(id, cfg) {{
  const el = document.getElementById(id);
  return new Chart(el, cfg);
}}

function initCharts() {{
  trendChart = makeChart("trendChart", {{
    type: "line",
    data: {{ labels: [], datasets: [] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: "index", intersect: false }},
      plugins: {{ legend: {{ position: "top", labels: {{ boxWidth: 12, padding: 12 }} }} }},
      scales: {{
        x: {{ ticks: {{ maxRotation: 45 }} }},
        y: {{ min: 0, max: 100, title: {{ display: true, text: "Composite Score" }} }},
      }},
    }},
  }});

  topPickChart = makeChart("topPickChart", {{
    type: "line",
    data: {{ labels: [], datasets: [] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            title: (items) => items[0].label,
            label: (item) => ` Score: ${{item.raw}}  (${{item.chart._tickers?.[item.dataIndex] ?? ""}})`,
          }},
        }},
      }},
      scales: {{
        x: {{ ticks: {{ maxRotation: 45 }} }},
        y: {{ min: 0, max: 100, title: {{ display: true, text: "Score" }} }},
      }},
    }},
  }});

  factorChart = makeChart("factorChart", {{
    type: "bar",
    data: {{ labels: [], datasets: [] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: "top" }} }},
      scales: {{
        x: {{ stacked: false }},
        y: {{ min: 0, max: 100, title: {{ display: true, text: "Avg Score" }} }},
      }},
    }},
  }});

  rsiChart = makeChart("rsiChart", {{
    type: "bar",
    data: {{ labels: [], datasets: [{{ label: "RSI-14", data: [], backgroundColor: [] }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ min: 0, max: 100, title: {{ display: true, text: "RSI" }} }} }},
    }},
  }});

  histoChart = makeChart("histoChart", {{
    type: "bar",
    data: {{ labels: [], datasets: [{{ label: "Count", data: [], backgroundColor: "#4e9af1aa", borderColor: "#4e9af1", borderWidth: 1 }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: "Score Range" }} }},
        y: {{ title: {{ display: true, text: "Frequency" }} }},
      }},
    }},
  }});

  sectorChart = makeChart("sectorChart", {{
    type: "doughnut",
    data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderColor: "#1a1d27", borderWidth: 2 }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: "right", labels: {{ boxWidth: 14, padding: 10 }} }} }},
    }},
  }});

  stabilityChart = makeChart("stabilityChart", {{
    type: "bubble",
    data: {{ datasets: [] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: "top", labels: {{ boxWidth: 12, padding: 10 }} }},
        tooltip: {{
          callbacks: {{
            label: (item) => ` ${{item.dataset.label}}  avg rank ${{item.raw.x}}  σ ${{item.raw.y}}`,
          }},
        }},
      }},
      scales: {{
        x: {{ title: {{ display: true, text: "Avg Rank (lower = better)" }} }},
        y: {{ title: {{ display: true, text: "Std Dev of Rank" }} }},
      }},
    }},
  }});
}}

function updateCharts(month) {{
  const d = ALL_DATA[month];
  if (!d) return;

  // Stats bar
  document.querySelector("#stat-runs .stat-value").textContent     = d.run_count;
  document.querySelector("#stat-tickers .stat-value").textContent  = d.total_tickers;
  document.querySelector("#stat-range .stat-value").textContent    = d.dates.length > 0
    ? (d.dates[0] === d.dates[d.dates.length - 1] ? d.dates[0] : d.dates[0] + " → " + d.dates[d.dates.length - 1])
    : "—";
  document.querySelector("#stat-top .stat-value").textContent      = d.latest_top_pick;

  // Titles
  const topNLabel = d.trend_datasets.length + " most frequent tickers";
  document.getElementById("trend-title").textContent  = "Composite Score Trends — " + topNLabel;
  document.getElementById("factor-title").textContent = "Avg Factor Scores — " + topNLabel;

  // 1. Trend
  trendChart.data.labels   = d.dates;
  trendChart.data.datasets = d.trend_datasets;
  trendChart.update();

  // 2. Top-pick
  topPickChart.data.labels              = d.top_pick.dates;
  topPickChart.data.datasets[0]         = {{
    label: "Top Pick Score",
    data: d.top_pick.scores,
    borderColor: "#f97316",
    backgroundColor: "#f9731633",
    tension: 0.3,
    pointRadius: 4,
    fill: true,
  }};
  topPickChart._tickers = d.top_pick.tickers;
  topPickChart.update();

  // 3. Factor
  factorChart.data.labels   = d.factor_labels;
  factorChart.data.datasets = d.factor_datasets;
  factorChart.update();

  // 4. RSI
  rsiChart.data.labels                              = d.rsi.tickers;
  rsiChart.data.datasets[0].data                   = d.rsi.rsi;
  rsiChart.data.datasets[0].backgroundColor        = d.rsi.colors;
  rsiChart.update();

  // 5. Histogram
  histoChart.data.labels            = d.histo.labels;
  histoChart.data.datasets[0].data  = d.histo.values;
  histoChart.update();

  // 6. Sector
  sectorChart.data.labels                        = d.sector.labels;
  sectorChart.data.datasets[0].data              = d.sector.values;
  sectorChart.data.datasets[0].backgroundColor   = d.sector_colors;
  sectorChart.update();

  // 7. Stability (bubble)
  stabilityChart.data.datasets = d.stability.map((pt, i) => ({{
    label: pt.label,
    data: [{{ x: pt.x, y: pt.y, r: 8 }}],
    backgroundColor: PALETTE[i % PALETTE.length] + "bb",
    borderColor:     PALETTE[i % PALETTE.length],
  }}));
  stabilityChart.update();
}}

// ---- Tab switching ----
document.querySelectorAll(".tab-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    updateCharts(btn.dataset.month);
  }});
}});

// ---- Init ----
initCharts();
const firstTab = document.querySelector(".tab-btn.active") || document.querySelector(".tab-btn");
if (firstTab) updateCharts(firstTab.dataset.month);
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(top_n: int = 10, out_path: str | None = None) -> str:
    """
    Build the charts HTML from the database and write it to disk.
    Returns the output file path. Called by main.py after each run.
    """
    out_path = out_path or os.path.join(
        config.REPORTS_DIR, f"{date.today()}_charts.html"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    df = _load_all()
    if df.empty:
        print("Charts: no data in database, skipping.")
        return out_path

    today_str = str(date.today())
    current_month = today_str[:7]

    months = sorted(df["month"].unique().tolist())
    all_data: dict[str, dict] = {}
    for m in months:
        all_data[m] = _build_month_data(df, m, top_n)

    html = _build_html(all_data, top_n, today_str, current_month)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate monthly charts HTML from the trading bot database.")
    parser.add_argument("--out",   default=None, help="Output path (default: reports/YYYY-MM-DD_charts.html)")
    parser.add_argument("--top-n", type=int, default=10, help="Tickers in trend/factor charts per month (default: 10)")
    args = parser.parse_args()

    print("Loading data from database...")
    df = _load_all()
    if df.empty:
        print("No data found. Run main.py or backfill.py first.")
        return

    print(f"  {df['run_date'].nunique()} runs across {df['month'].nunique()} month(s), {df['ticker'].nunique()} tickers")

    out_path = generate(top_n=args.top_n, out_path=args.out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
