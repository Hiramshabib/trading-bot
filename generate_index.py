#!/usr/bin/env python3
"""
Generate index.html for the trading-bot-reports GitHub Pages site.
Run from the trading-bot-reports repo root directory:
  python /path/to/trading-bot/generate_index.py
"""
import glob
import os
import re
from datetime import datetime, timezone


REPORTS_DIR = "reports"
OUTPUT_FILE = "index.html"


def scan_reports():
    rec_files = sorted(
        glob.glob(os.path.join(REPORTS_DIR, "*_recommendations.html")),
        reverse=True,
    )
    chart_map = {
        m.group(1): f
        for f in glob.glob(os.path.join(REPORTS_DIR, "*_charts.html"))
        for m in [re.match(r".+[/\\](\d{4}-\d{2}-\d{2})_charts\.html", f)]
        if m
    }

    entries = []
    for f in rec_files:
        m = re.match(r".+[/\\](\d{4}-\d{2}-\d{2})_recommendations\.html", f)
        if not m:
            continue
        date_str = m.group(1)
        try:
            label = datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %d, %Y")
        except ValueError:
            label = date_str
        chart_f = chart_map.get(date_str)
        entries.append({
            "date": date_str,
            "label": label,
            "rec": f"reports/{os.path.basename(f)}",
            "chart": f"reports/{os.path.basename(chart_f)}" if chart_f else None,
        })
    return entries


def build_html(entries):
    if not entries:
        return "<html><body><p>No reports yet.</p></body></html>"

    latest = entries[0]

    items = []
    for i, e in enumerate(entries):
        active = ' class="active"' if i == 0 else ""
        chart_btn = (
            f'<a href="{e["chart"]}" target="report-frame" '
            f'class="chart-link" title="Charts">&#9642;</a>'
            if e["chart"] else ""
        )
        items.append(
            f'<li{active}>'
            f'<a href="{e["rec"]}" target="report-frame" '
            f'onclick="setActive(this)">{e["label"]}</a>'
            f'{chart_btn}</li>'
        )

    sidebar = "\n      ".join(items)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cache_bust = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trading Bot Dashboard</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ display: flex; height: 100vh; font-family: Arial, sans-serif; background: #0d1117; }}

    #sidebar {{
      width: 210px; min-width: 210px;
      background: #161b22; color: #c9d1d9;
      display: flex; flex-direction: column; overflow: hidden;
      border-right: 1px solid #30363d;
    }}
    #sidebar-header {{
      padding: 1rem 0.9rem 0.75rem;
      border-bottom: 1px solid #30363d;
    }}
    #sidebar-header h1 {{ font-size: 0.95rem; color: #58a6ff; font-weight: 600; }}
    #sidebar-header p  {{ font-size: 0.68rem; color: #8b949e; margin-top: 0.2rem; }}
    #sidebar-header .updated {{ font-size: 0.65rem; color: #484f58; margin-top: 0.4rem; }}

    #report-list {{ list-style: none; overflow-y: auto; flex: 1; }}
    #report-list li {{
      display: flex; align-items: center;
      border-bottom: 1px solid #21262d;
    }}
    #report-list li a {{
      display: block; padding: 0.6rem 0.9rem; color: #8b949e;
      text-decoration: none; font-size: 0.82rem; flex: 1;
      transition: background 0.1s;
    }}
    #report-list li a:hover      {{ background: #1f2937; color: #e6edf3; }}
    #report-list li.active a     {{ background: #1f2937; color: #58a6ff; font-weight: 600; }}
    .chart-link {{
      padding: 0.5rem 0.5rem 0.5rem 0.2rem;
      color: #484f58; text-decoration: none; font-size: 0.75rem;
    }}
    .chart-link:hover {{ color: #58a6ff; }}

    #main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
    iframe {{ flex: 1; border: none; width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id="sidebar">
    <div id="sidebar-header">
      <h1>Trading Bot</h1>
      <p>Daily Stock Recommendations</p>
      <p class="updated">Updated: {updated}</p>
    </div>
    <ul id="report-list">
      {sidebar}
    </ul>
  </div>
  <div id="main">
    <iframe name="report-frame" id="report-frame" src="{latest['rec']}?v={cache_bust}"></iframe>
  </div>
  <script>
    function setActive(link) {{
      document.querySelectorAll('#report-list li').forEach(li => li.classList.remove('active'));
      link.closest('li').classList.add('active');
    }}
  </script>
</body>
</html>"""


if __name__ == "__main__":
    entries = scan_reports()
    html = build_html(entries)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated {OUTPUT_FILE} with {len(entries)} report(s).")
