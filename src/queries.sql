-- Exploratory SQL queries against the "fleet" PostgreSQL database.
-- Run with: psql -h localhost -U fleet -d fleet -f src/queries.sql
-- or paste individual queries into any Postgres client (DBeaver, pgAdmin, etc.)

-- 1. Monthly ticket volume overall (the final month, 2026-09, is partial: data ends Sep 14)
SELECT
    substr(ticket_date, 1, 7) AS month,
    COUNT(*) AS ticket_count
FROM tickets
GROUP BY month
ORDER BY month;

-- 2. Monthly ticket volume by category
SELECT
    substr(ticket_date, 1, 7) AS month,
    category,
    COUNT(*) AS ticket_count
FROM tickets
GROUP BY month, category
ORDER BY month, ticket_count DESC;

-- 3. Top failing models (tickets per device, to normalize for fleet size)
SELECT
    t.model,
    COUNT(*) AS total_tickets,
    COUNT(DISTINCT t.device_id) AS distinct_devices_affected,
    (SELECT COUNT(*) FROM devices d WHERE d.model = t.model) AS fleet_size,
    ROUND(1.0 * COUNT(*) / (SELECT COUNT(*) FROM devices d WHERE d.model = t.model), 3) AS tickets_per_device
FROM tickets t
GROUP BY t.model
ORDER BY tickets_per_device DESC;

-- 4. The "bad batch" signal: Battery tickets per device on EliteBook 840, by purchase quarter
-- (normalized per device in the cohort, since older cohorts have more time to accumulate tickets)
SELECT
    q.purchase_quarter,
    q.fleet_size,
    COALESCE(b.battery_tickets, 0) AS battery_tickets,
    ROUND(COALESCE(b.battery_tickets, 0) * 1.0 / q.fleet_size, 3) AS battery_tickets_per_device
FROM (
    SELECT
        EXTRACT(YEAR FROM purchase_date::date) || '-Q' || EXTRACT(QUARTER FROM purchase_date::date) AS purchase_quarter,
        COUNT(*) AS fleet_size
    FROM devices
    WHERE model = 'EliteBook 840'
    GROUP BY purchase_quarter
) q
LEFT JOIN (
    SELECT
        EXTRACT(YEAR FROM d.purchase_date::date) || '-Q' || EXTRACT(QUARTER FROM d.purchase_date::date) AS purchase_quarter,
        COUNT(*) AS battery_tickets
    FROM tickets t
    JOIN devices d ON d.device_id = t.device_id
    WHERE t.model = 'EliteBook 840' AND t.category = 'Battery'
    GROUP BY purchase_quarter
) b ON b.purchase_quarter = q.purchase_quarter
ORDER BY q.purchase_quarter;

-- 5. Average resolution time by severity and warranty status
SELECT
    severity,
    in_warranty,
    ROUND(AVG(resolution_hours)::numeric, 1) AS avg_resolution_hours,
    COUNT(*) AS n_tickets
FROM tickets
GROUP BY severity, in_warranty
ORDER BY severity, in_warranty;

-- 6. Ticket volume by region and category (for the regional drill-down in Power BI)
SELECT
    region,
    category,
    COUNT(*) AS ticket_count
FROM tickets
GROUP BY region, category
ORDER BY region, ticket_count DESC;

-- 7. Devices approaching warranty expiry in the next 90 days (proactive outreach list)
SELECT
    device_id, model, region, warranty_end_date
FROM devices
WHERE warranty_end_date::date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '90 days'
ORDER BY warranty_end_date;
