"""
Forecasts next-quarter support ticket volume using Holt-Winters exponential
smoothing on the monthly ticket time series (overall, and per region).

Outputs (written to outputs/):
  - forecast_overall.csv
  - forecast_by_region.csv
  - monthly_history.png (chart: history + forecast)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
FORECAST_MONTHS = 3


def ends_mid_month(tickets: pd.DataFrame) -> bool:
    last_day = tickets["ticket_date"].max()
    return last_day != last_day + pd.offsets.MonthEnd(0)


def load_monthly_series(tickets: pd.DataFrame, region: str | None = None) -> pd.Series:
    df = tickets if region is None else tickets[tickets["region"] == region]
    monthly = df.set_index("ticket_date").resample("MS").size()
    # A partial final month would look like a demand drop and bias the fit, so exclude it.
    if ends_mid_month(tickets):
        monthly = monthly.iloc[:-1]
    return monthly


def fit_and_forecast(series: pd.Series, periods: int = FORECAST_MONTHS) -> pd.Series:
    # Trim leading zero-ish ramp-up months so the model isn't skewed by the
    # fleet's early, sparsely-populated period.
    series = series[series.index >= series[series > 0].index.min()]
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=12 if len(series) >= 24 else None,
        initialization_method="estimated",
    )
    fit = model.fit()
    forecast = fit.forecast(periods)
    return forecast


def main():
    tickets = pd.read_csv(DATA_DIR / "tickets.csv", parse_dates=["ticket_date"])

    # When the partial month is dropped, forecast one extra month so it is still covered.
    horizon = FORECAST_MONTHS + (1 if ends_mid_month(tickets) else 0)

    overall = load_monthly_series(tickets)
    overall_forecast = fit_and_forecast(overall, horizon)

    overall_out = pd.DataFrame(
        {
            "month": list(overall.index) + list(overall_forecast.index),
            "ticket_count": list(overall.values) + [None] * len(overall_forecast),
            "forecast": [None] * len(overall) + list(overall_forecast.values),
        }
    )
    overall_out.to_csv(OUTPUT_DIR / "forecast_overall.csv", index=False)

    region_rows = []
    for region in sorted(tickets["region"].unique()):
        series = load_monthly_series(tickets, region=region)
        try:
            fc = fit_and_forecast(series, horizon)
        except Exception as exc:  # small regions can be too short/sparse to fit seasonally
            print(f"  (skipping seasonal fit for {region}: {exc}; using simple trend)")
            model = ExponentialSmoothing(series, trend="add", initialization_method="estimated")
            fc = model.fit().forecast(horizon)
        for month, value in fc.items():
            region_rows.append({"region": region, "month": month, "forecast": round(value, 1)})
    pd.DataFrame(region_rows).to_csv(OUTPUT_DIR / "forecast_by_region.csv", index=False)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(overall.index, overall.values, label="Actual monthly tickets", marker="o")
    # Start the dashed line at the last actual so the two series read as one continuous line.
    forecast_line = pd.concat([overall.iloc[[-1]], overall_forecast])
    ax.plot(forecast_line.index, forecast_line.values, label="Forecast", marker="o", linestyle="--", color="tab:orange")
    ax.set_title("Fleet Support Tickets: Monthly Volume & Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Ticket count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_history.png", dpi=150)

    print("Overall forecast:")
    print(overall_forecast.round(1))
    print(f"\nWrote forecast_overall.csv, forecast_by_region.csv, monthly_history.png -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
