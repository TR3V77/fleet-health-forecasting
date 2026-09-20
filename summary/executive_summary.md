# Fleet Health & Support Forecasting: Executive Summary

**Scope:** 3,000 devices (laptops and printers) across 4 regions and 5,505 support tickets, Sep 2023 to Sep 14, 2026.
The data is synthetic; the patterns below were built into it and then recovered by the analysis.

## Key Findings

1. **One purchase cohort is driving a battery defect.** EliteBook 840 units bought in Q1 2025 show
   **1.21 battery tickets per device**, versus 0.17-0.44 for the other purchase quarters (2.8x the
   next-highest, up to 7x the lowest). 41 of that cohort's 46 battery tickets fall 9-14 months after purchase. A spike
   isolated to one purchase window, rather than a slope across cohorts, is the signature of a bad component
   batch and not general wear-out (`src/queries.sql`, query 4). 2026 purchases are excluded from the
   comparison because they are too new to have reached the failure window.
   **Recommendation:** contact this cohort proactively for battery replacement before failure.

2. **Ticket volume is seasonal and forecastable.** Measured against trend, August runs 1.35x, September
   1.22x and January 1.27x, with a summer lull (Jun-Jul) at 0.75x. A Holt-Winters model forecasts
   **332 tickets in September, 317 in October, 312 in November and 335 in December 2026**. On a 6-month
   holdout it missed by 3.9% on average, versus 16.7% for a last-value baseline.
   **Recommendation:** staff for the Aug-Sep and January peaks specifically, not a flat average.

3. **Severity, not category, should drive SLA design.** Critical tickets average **69.4 hours** to resolve,
   High 34.3, Medium 13.5 and Low 4.6. Categories look different overall (Hardware Failure 37.3h vs. Other
   5.3h) only because Hardware Failure is mostly High or Critical severity; within a severity level,
   resolution time is roughly flat across categories.
   **Recommendation:** route and set SLAs by severity, with a distinct fast-path for Critical.

4. **Ticket rate varies more by product than by region.** Laptops average 2.03 tickets per device vs. 1.49
   for printers (ProBook 450 highest at 2.24, Color LaserJet M555 lowest at 1.31), while regions only span
   1.78-1.96.
   **Recommendation:** use per-model rates as an input to hardware reliability targets and vendor reviews.

## Ticket auto-tagging
A TF-IDF and logistic regression classifier predicts a ticket's category from its description alone:
**88% accuracy (macro F1 0.89)** on a 25% held-out split. The ticket text is templated synthetic data, with
12% of descriptions made deliberately vague and 4% of tickets mislabeled, so this shows the method works,
not what real-world accuracy would be.

## How this was produced
- **Data:** synthetic fleet and ticket log generated in Python (`src/generate_data.py`), seeded so every run
  reproduces the same data.
- **SQL:** cohort, trend and warranty-expiry queries against PostgreSQL (`src/queries.sql`).
- **Forecasting:** Holt-Winters exponential smoothing on monthly ticket counts, overall and by region
  (`src/forecast.py`). The partial final month (September 2026, data through Sep 14) is excluded from the fit.
- **Visualization:** Power BI report (`powerbi/`) covering fleet overview, trend and forecast, and regional drill-down.
