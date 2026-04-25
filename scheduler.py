"""
Runs main.py once every day at 07:00 local time.
Start with: python scheduler.py
Stop with:  Ctrl+C

Alternatively, set up a Windows Task Scheduler job pointing at:
  python C:\Users\Hiram\Documents\trading-bot\main.py
"""
import subprocess
import sys
import time

import schedule


def run_bot() -> None:
    print("Scheduler: starting bot run…")
    result = subprocess.run([sys.executable, "main.py"])
    if result.returncode != 0:
        print(f"Scheduler: bot exited with code {result.returncode}")
    else:
        print("Scheduler: run complete.")


if __name__ == "__main__":
    schedule.every().day.at("07:00").do(run_bot)
    print("Scheduler started — bot will run daily at 07:00. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)
