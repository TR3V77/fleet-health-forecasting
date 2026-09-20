"""
Generates a synthetic-but-realistic device fleet + support ticket dataset.

Two CSVs are written to ../data/:
  - devices.csv : one row per device in the fleet
  - tickets.csv : one row per support ticket raised against a device

A deliberate "bad batch" is injected (one laptop model, one purchase quarter)
so downstream analysis/forecasting has a real signal to find and explain.
"""
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TODAY = date(2026, 9, 14)
FLEET_START = TODAY - timedelta(days=3 * 365)  # devices purchased over the last 3 years
N_DEVICES = 3000

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
REGION_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

MODELS = [
    ("EliteBook 840", "Laptop", 0.9),
    ("EliteBook 650", "Laptop", 1.0),
    ("ProBook 450", "Laptop", 1.1),
    ("ZBook Firefly 14", "Laptop", 0.85),
    ("LaserJet Pro M404", "Printer", 0.7),
    ("OfficeJet Pro 9015", "Printer", 0.75),
    ("Color LaserJet Enterprise M555", "Printer", 0.65),
]
MODEL_NAMES = [m[0] for m in MODELS]
MODEL_TYPE = {m[0]: m[1] for m in MODELS}
MODEL_BASE_RATE = {m[0]: m[2] for m in MODELS}  # relative baseline ticket rate

WARRANTY_MONTHS_CHOICES = [12, 24, 36]
WARRANTY_MONTHS_WEIGHTS = [0.5, 0.35, 0.15]

# The "bad batch": EliteBook 840 units purchased in this quarter develop a
# battery defect that shows up ~9-14 months after purchase.
BAD_BATCH_MODEL = "EliteBook 840"
BAD_BATCH_START = date(2025, 1, 1)
BAD_BATCH_END = date(2025, 3, 31)

CATEGORIES = [
    "Hardware Failure",
    "Battery",
    "Software/OS",
    "Network/Connectivity",
    "Print Quality",
    "Driver/Firmware",
    "Other",
]

# Ticket description templates per category, used later to demo a text classifier.
DESCRIPTION_TEMPLATES = {
    "Hardware Failure": [
        "Device won't power on after {event}.",
        "Screen shows physical damage and flickers intermittently.",
        "Unit overheats and shuts down under normal load.",
        "Keyboard/trackpad stopped responding after {event}.",
    ],
    "Battery": [
        "Battery drains from full to empty in under {n} minutes.",
        "Device will not hold a charge, battery health reports critical.",
        "Battery swelling observed, requesting immediate replacement.",
        "Charger connected but battery percentage is not increasing.",
    ],
    "Software/OS": [
        "OS fails to boot, stuck on repair screen after {event}.",
        "Frequent blue screen crashes since the latest update.",
        "Application suite freezes randomly during normal use.",
        "System performance degraded significantly after {event}.",
    ],
    "Network/Connectivity": [
        "Cannot connect to corporate Wi-Fi after {event}.",
        "VPN drops repeatedly, disconnecting every few minutes.",
        "Ethernet port not recognized by the operating system.",
        "Bluetooth peripherals fail to pair with the device.",
    ],
    "Print Quality": [
        "Printouts show streaking and faded toner across the page.",
        "Paper jams on nearly every print job since {event}.",
        "Color calibration is off, prints appear washed out.",
        "Duplex printing misaligns pages consistently.",
    ],
    "Driver/Firmware": [
        "Firmware update failed and device now shows an error code.",
        "Display driver crashes when connecting an external monitor.",
        "Print spooler driver conflict causes repeated job failures.",
        "Device firmware reports version mismatch after {event}.",
    ],
    "Other": [
        "User requesting accessory replacement (cable, dock, tray).",
        "General usability question routed to hardware support.",
        "Cosmetic issue with casing, no functional impact reported.",
        "Asset tag missing, requesting re-registration in inventory.",
    ],
}

SEVERITY_BY_CATEGORY = {
    "Hardware Failure": ["High", "Critical", "High", "Medium"],
    "Battery": ["Medium", "High", "Medium"],
    "Software/OS": ["Medium", "Low", "High"],
    "Network/Connectivity": ["Low", "Medium"],
    "Print Quality": ["Low", "Medium"],
    "Driver/Firmware": ["Medium", "Low"],
    "Other": ["Low"],
}

GENERIC_DESCRIPTIONS = [
    "Device is not working properly, please assist.",
    "Having an issue since yesterday, need help.",
    "Something is wrong with my device, urgent.",
    "Problem reported by user, details unclear.",
    "Not behaving as expected, requesting support.",
    "Issue persists after restart, please advise.",
]
VAGUE_RATE = 0.12  # share of tickets with a non-specific description
MISLABEL_RATE = 0.04  # share of tickets filed under the wrong category

EVENT_FILLERS = ["the last update", "a recent firmware push", "reimaging", "a dock swap", "a BIOS update"]


def add_text_noise(category: str, description: str) -> tuple[str, str]:
    """Real ticket text is often vague and sometimes misfiled; simulate both."""
    if random.random() < VAGUE_RATE:
        description = random.choice(GENERIC_DESCRIPTIONS)
    if random.random() < MISLABEL_RATE:
        category = random.choice([c for c in CATEGORIES if c != category])
    return category, description


def random_date(start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta_days, 0)))


def month_seasonality(d: date) -> float:
    """Relative multiplier for ticket likelihood by month (back-to-school + post-holiday spikes)."""
    m = d.month
    if m in (8, 9):
        return 1.35  # back-to-school deployment churn
    if m == 1:
        return 1.25  # post-holiday return-to-office issues
    if m in (6, 7):
        return 0.85  # summer lull
    return 1.0


def draw_ticket_day(purchase_date: date, active_days: int) -> int:
    """Sample days-since-purchase, thinning by month so seasonal peaks and lulls appear."""
    peak = 1.35
    while True:
        days = random.randint(1, active_days)
        if random.random() < month_seasonality(purchase_date + timedelta(days=days)) / peak:
            return days


def build_devices() -> pd.DataFrame:
    rows = []
    for i in range(1, N_DEVICES + 1):
        model = random.choices(MODEL_NAMES, weights=[m[2] for m in MODELS], k=1)[0]
        region = random.choices(REGIONS, weights=REGION_WEIGHTS, k=1)[0]
        purchase_date = random_date(FLEET_START, TODAY - timedelta(days=30))
        warranty_months = random.choices(WARRANTY_MONTHS_CHOICES, weights=WARRANTY_MONTHS_WEIGHTS, k=1)[0]
        warranty_end = purchase_date + timedelta(days=warranty_months * 30)
        rows.append(
            {
                "device_id": f"DEV-{i:05d}",
                "model": model,
                "device_type": MODEL_TYPE[model],
                "region": region,
                "purchase_date": purchase_date.isoformat(),
                "warranty_months": warranty_months,
                "warranty_end_date": warranty_end.isoformat(),
            }
        )
    return pd.DataFrame(rows)


def is_bad_batch_unit(model: str, purchase_date: date) -> bool:
    return model == BAD_BATCH_MODEL and BAD_BATCH_START <= purchase_date <= BAD_BATCH_END


def build_tickets(devices: pd.DataFrame) -> pd.DataFrame:
    ticket_rows = []
    ticket_counter = 1

    for _, dev in devices.iterrows():
        purchase_date = date.fromisoformat(dev["purchase_date"])
        model = dev["model"]
        base_rate = MODEL_BASE_RATE[model]
        bad_batch = is_bad_batch_unit(model, purchase_date)

        active_days = (TODAY - purchase_date).days
        if active_days <= 0:
            continue

        # Expected number of tickets over the device's lifetime so far.
        expected_tickets = base_rate * (active_days / 365.0) * 1.4
        n_tickets = np.random.poisson(lam=max(expected_tickets, 0.05))

        for _ in range(n_tickets):
            days_since_purchase = draw_ticket_day(purchase_date, active_days)
            ticket_date = purchase_date + timedelta(days=days_since_purchase)

            if bad_batch and 270 <= days_since_purchase <= 420:
                # Defect window: heavily weighted toward Battery tickets.
                category = random.choices(
                    CATEGORIES,
                    weights=[5, 55, 10, 10, 5, 10, 5],
                    k=1,
                )[0]
            elif dev["device_type"] == "Printer":
                category = random.choices(
                    CATEGORIES,
                    weights=[10, 0, 5, 10, 45, 20, 10],
                    k=1,
                )[0]
            else:
                category = random.choices(
                    CATEGORIES,
                    weights=[20, 15, 25, 20, 0, 15, 5],
                    k=1,
                )[0]

            severity = random.choice(SEVERITY_BY_CATEGORY[category])
            template = random.choice(DESCRIPTION_TEMPLATES[category])
            description = template.format(event=random.choice(EVENT_FILLERS), n=random.choice([15, 20, 30, 45]))
            logged_category, description = add_text_noise(category, description)

            base_resolution_hours = {"Low": 4, "Medium": 12, "High": 30, "Critical": 60}[severity]
            resolution_hours = max(1, np.random.lognormal(mean=np.log(base_resolution_hours), sigma=0.5))

            ticket_rows.append(
                {
                    "ticket_id": f"TCK-{ticket_counter:06d}",
                    "device_id": dev["device_id"],
                    "model": model,
                    "region": dev["region"],
                    "ticket_date": ticket_date.isoformat(),
                    "category": logged_category,
                    "severity": severity,
                    "description": description,
                    "resolution_hours": round(resolution_hours, 1),
                    "in_warranty": ticket_date <= date.fromisoformat(dev["warranty_end_date"]),
                }
            )
            ticket_counter += 1

        # Bad-batch units get an additional, near-guaranteed battery failure
        # in the defect window on top of their normal ticket volume, so the
        # defect produces an unambiguous spike rather than a subtle skew.
        if bad_batch and active_days >= 270 and random.random() < 0.8:
            window_end = min(active_days, 420)
            days_since_purchase = random.randint(270, window_end)
            ticket_date = purchase_date + timedelta(days=days_since_purchase)
            severity = random.choices(["High", "Critical"], weights=[0.6, 0.4], k=1)[0]
            template = random.choice(DESCRIPTION_TEMPLATES["Battery"])
            description = template.format(event=random.choice(EVENT_FILLERS), n=random.choice([15, 20, 30]))
            base_resolution_hours = {"High": 30, "Critical": 60}[severity]
            resolution_hours = max(1, np.random.lognormal(mean=np.log(base_resolution_hours), sigma=0.5))
            ticket_rows.append(
                {
                    "ticket_id": f"TCK-{ticket_counter:06d}",
                    "device_id": dev["device_id"],
                    "model": model,
                    "region": dev["region"],
                    "ticket_date": ticket_date.isoformat(),
                    "category": "Battery",
                    "severity": severity,
                    "description": description,
                    "resolution_hours": round(resolution_hours, 1),
                    "in_warranty": ticket_date <= date.fromisoformat(dev["warranty_end_date"]),
                }
            )
            ticket_counter += 1

    tickets = pd.DataFrame(ticket_rows)
    return tickets.sort_values("ticket_date").reset_index(drop=True)


def main():
    devices = build_devices()
    tickets = build_tickets(devices)

    devices_path = DATA_DIR / "devices.csv"
    tickets_path = DATA_DIR / "tickets.csv"
    devices.to_csv(devices_path, index=False)
    tickets.to_csv(tickets_path, index=False)

    print(f"Wrote {len(devices)} devices -> {devices_path}")
    print(f"Wrote {len(tickets)} tickets -> {tickets_path}")


if __name__ == "__main__":
    main()
