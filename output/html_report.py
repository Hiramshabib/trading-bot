import os
from datetime import date

import pandas as pd
from jinja2 import Environment, FileSystemLoader

import config


def export_html(
    df: pd.DataFrame,
    det_df: pd.DataFrame | None = None,
    over_df: pd.DataFrame | None = None,
    run_date: date | None = None,
) -> str:
    if run_date is None:
        run_date = date.today()

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report.html.j2")

    top = df.head(config.TOP_N_RESULTS)
    rows = top.to_dict(orient="records")

    det_alerts  = det_df.to_dict(orient="records")  if det_df  is not None and not det_df.empty  else []
    over_alerts = over_df.to_dict(orient="records") if over_df is not None and not over_df.empty else []

    html = template.render(
        run_date=str(run_date),
        total=len(df),
        rows=rows,
        det_alerts=det_alerts,
        over_alerts=over_alerts,
    )

    path = os.path.join(config.REPORTS_DIR, f"{run_date}_recommendations.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
