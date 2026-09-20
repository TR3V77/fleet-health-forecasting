# Fleet Health & Support Forecasting

An analytics project on a problem enterprise IT support teams face: given a fleet of deployed devices and
their support ticket history, what is driving support load, and what is coming next?

The data is synthetic: 3,000 laptops and printers across 4 regions and about 3 years of support tickets, with
a battery defect deliberately hidden in one laptop model's purchase cohort for the analysis to find.

Work in progress; built up in stages.

## Setup
Requires Python 3.10+ and Docker.
```powershell
docker compose up -d       # PostgreSQL on localhost:5432 (database, user and password: fleet)
pip install -r requirements.txt
```
