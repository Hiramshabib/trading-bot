"""
Trading Bot — main entry point
Usage:
  python main.py                        # full run (50 tickers)
  python main.py --top-n 10             # score only top 10
  python main.py --tickers AAPL MSFT    # custom ticker list
  python main.py --no-html              # skip HTML report
  python main.py --no-publish           # skip pushing to trading-bot-reports
"""
import argparse
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import config
from data.universe     import get_universe
from data.momentum     import fetch_momentum_scores, fetch_finviz_gainers
from data.analyst      import fetch_analyst_scores
from data.fundamentals import fetch_fundamentals_scores
from data.score_trend  import fetch_score_trend_scores
from data.downside     import compute_downside_alerts
from scoring.composite import build_composite_scores
from storage.database  import save_run, get_run_count
from output.terminal   import print_results, print_alerts
from output.csv_export import export_csv
from output.html_report import export_html
from charts import generate as generate_charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock recommendation bot")
    parser.add_argument("--top-n",   type=int,  default=config.TOP_N_RESULTS,
                        help="Number of results to display")
    parser.add_argument("--tickers", nargs="+",
                        help="Override the ticker universe with specific symbols")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip generating the HTML report")
    parser.add_argument("--no-publish", action="store_true",
                        help="Skip pushing reports to trading-bot-reports")
    return parser.parse_args()


def publish_reports(run_date: date) -> None:
    bot_dir = Path(__file__).parent.resolve()
    reports_repo = bot_dir.parent / "trading-bot-reports"
    if not (reports_repo / ".git").exists():
        print(f"Publish skipped: trading-bot-reports repo not found at {reports_repo}")
        return

    date_str = run_date.strftime("%Y-%m-%d")
    dest = reports_repo / "reports"
    dest.mkdir(exist_ok=True)

    copied = []
    for src in bot_dir.glob(f"reports/{date_str}*.html"):
        shutil.copy2(src, dest / src.name)
        copied.append(src.name)

    if not copied:
        print("Publish skipped: no HTML reports found for today.")
        return

    subprocess.run(
        [sys.executable, str(bot_dir / "generate_index.py")],
        cwd=str(reports_repo), check=True,
    )

    subprocess.run(["git", "config", "user.email", "bot@trading-bot"], cwd=str(reports_repo), check=True)
    subprocess.run(["git", "config", "user.name", "Trading Bot"],       cwd=str(reports_repo), check=True)
    subprocess.run(["git", "add", "reports/", "index.html"],            cwd=str(reports_repo), check=True)

    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(reports_repo))
    if diff.returncode != 0:
        subprocess.run(["git", "commit", "-m", f"Daily run {date_str}"], cwd=str(reports_repo), check=True)
        subprocess.run(["git", "push", "origin", "master"],              cwd=str(reports_repo), check=True)
        print(f"Published {len(copied)} report(s) to trading-bot-reports.")
    else:
        print("Publish skipped: no changes to commit.")


def main() -> None:
    args = parse_args()
    config.TOP_N_RESULTS = args.top_n

    print("\n[1/5] Building ticker universe…")
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"  Using custom list: {tickers}")
    else:
        from data.universe import TOP_50_BY_MARKET_CAP
        sp500_set = set(TOP_50_BY_MARKET_CAP)
        finviz_extras = [t for t in fetch_finviz_gainers() if t in sp500_set][:10]
        tickers = get_universe(extra_tickers=finviz_extras)
        print(f"  Universe: {len(tickers)} tickers ({len(finviz_extras)} Finviz extras)")

    print("\n[2/5] Fetching momentum data…")
    momentum_df = fetch_momentum_scores(tickers)

    print("\n[3/5] Fetching analyst data…")
    analyst_df = fetch_analyst_scores(tickers)

    print("\n[4/5] Fetching fundamentals data…")
    fundamentals_df = fetch_fundamentals_scores(tickers)

    print("\n[5/5] Scoring and ranking…")
    score_trend_df = None
    if get_run_count() >= config.SCORE_TREND_LOOKBACK:
        print("  Score trend signal active (enough historical runs found)")
        score_trend_df = fetch_score_trend_scores(tickers)
    results = build_composite_scores(momentum_df, analyst_df, fundamentals_df, score_trend_df)

    print_results(results)

    print("\n[Downside alerts]")
    det_df, over_df = compute_downside_alerts(results, momentum_df, score_trend_df)
    print_alerts(det_df, over_df)

    csv_path = export_csv(results)
    print(f"CSV saved:  {csv_path}")

    if not args.no_html:
        html_path = export_html(results, det_df, over_df)
        print(f"HTML saved: {html_path}")

    run_id = save_run(results)
    print(f"Run #{run_id} saved to {config.DB_PATH}")

    if not args.no_html:
        charts_path = generate_charts()
        print(f"Charts saved: {charts_path}\n")

    if not args.no_html and not args.no_publish:
        from datetime import date as date_cls
        publish_reports(date_cls.today())


if __name__ == "__main__":
    main()
