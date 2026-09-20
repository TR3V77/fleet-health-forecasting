"""
Single entry point that runs the full pipeline end to end:
  generate data -> load into PostgreSQL -> forecast -> train categorizer

This is the "automation" piece: run this one script (or schedule it, e.g.
via Windows Task Scheduler / cron) to refresh everything from scratch.
"""
import subprocess
import sys
from pathlib import Path

STEPS = ["generate_data.py", "build_db.py", "forecast.py", "categorize.py"]
SRC_DIR = Path(__file__).resolve().parent


def main():
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        result = subprocess.run([sys.executable, str(SRC_DIR / step)])
        if result.returncode != 0:
            print(f"Step {step} failed, stopping pipeline.")
            sys.exit(result.returncode)
    print("\nPipeline complete. Data, database, forecasts, and model are up to date.")


if __name__ == "__main__":
    main()
