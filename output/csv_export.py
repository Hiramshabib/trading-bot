import os
from datetime import date

import pandas as pd

import config


def export_csv(df: pd.DataFrame) -> str:
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    path = os.path.join(config.REPORTS_DIR, f"{date.today()}_recommendations.csv")
    df.to_csv(path, index=False)
    return path
