"""Loads devices.csv and tickets.csv into a PostgreSQL database for SQL analysis.

Connects using PG* environment variables (falls back to the docker-compose
defaults: host localhost, port 5432, db/user/password all "fleet"). Start the
database with `docker compose up -d` before running this.
"""
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PG_HOST = os.environ.get("PGHOST", "localhost")
PG_PORT = os.environ.get("PGPORT", "5432")
PG_DB = os.environ.get("PGDATABASE", "fleet")
PG_USER = os.environ.get("PGUSER", "fleet")
PG_PASSWORD = os.environ.get("PGPASSWORD", "fleet")


def main():
    devices = pd.read_csv(DATA_DIR / "devices.csv")
    tickets = pd.read_csv(DATA_DIR / "tickets.csv")

    url = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    engine = create_engine(url)

    devices.to_sql("devices", engine, if_exists="replace", index=False)
    tickets.to_sql("tickets", engine, if_exists="replace", index=False)

    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tickets_device ON tickets(device_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tickets_date ON tickets(ticket_date)"))

    print(f"Loaded {len(devices)} devices and {len(tickets)} tickets -> postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}")


if __name__ == "__main__":
    main()
