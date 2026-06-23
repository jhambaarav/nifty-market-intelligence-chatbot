-- =============================================
-- PROJECT 6: NIFTY MARKET INTELLIGENCE CHATBOT
-- queries.sql — Core SQL views and queries
-- =============================================

-- TABLE: nifty_ohlcv
-- Raw Nifty 50 daily OHLCV data from GOOGLEFINANCE
-- date | open | high | low | close | volume

-- TABLE: india_vix
-- Daily India VIX closing values
-- date | close

-- TABLE: sectoral
-- Daily OHLCV for Bank Nifty, IT, Auto, FMCG
-- date | open | high | low | close | sector

-- TABLE: weekly_sentiment
-- FinBERT sentiment scores per week
-- week | sentiment | score | weekly_return | vix

-- =============================================
-- VIEWS
-- =============================================

-- Weekly Nifty aggregation
CREATE VIEW IF NOT EXISTS weekly_nifty AS
SELECT
    strftime('%Y-W%W', date) AS week,
    MIN(date) AS week_start,
    ROUND(AVG(close), 2) AS avg_close,
    ROUND((MAX(close) - MIN(close)) / MIN(close) * 100, 2) AS weekly_range_pct,
    ROUND((julianday(MAX(date)) - julianday(MIN(date))), 0) AS trading_days
FROM nifty_ohlcv
GROUP BY week;

-- VIX regime classification
CREATE VIEW IF NOT EXISTS vix_regime AS
SELECT
    date,
    close AS vix,
    CASE
        WHEN close < 13 THEN 'LOW'
        WHEN close BETWEEN 13 AND 20 THEN 'MEDIUM'
        ELSE 'HIGH'
    END AS regime
FROM india_vix;

-- Sector weekly aggregation
CREATE VIEW IF NOT EXISTS sector_weekly AS
SELECT
    strftime('%Y-W%W', date) AS week,
    sector,
    ROUND(AVG(close), 2) AS avg_close,
    ROUND((MAX(close) - MIN(close)) / MIN(close) * 100, 2) AS weekly_range_pct
FROM sectoral
GROUP BY week, sector;

-- =============================================
-- ANALYTICAL QUERIES
-- =============================================

-- 1. Weekly Nifty returns
SELECT
    week,
    week_start,
    avg_close,
    ROUND((avg_close - LAG(avg_close) OVER (ORDER BY week)) /
          LAG(avg_close) OVER (ORDER BY week) * 100, 2) AS weekly_return_pct
FROM weekly_nifty
ORDER BY week;

-- 2. VIX regime distribution
SELECT
    regime,
    COUNT(*) as days,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM vix_regime), 1) as pct
FROM vix_regime
GROUP BY regime
ORDER BY days DESC;

-- 3. Most volatile sector per week
SELECT
    week,
    sector,
    weekly_range_pct
FROM sector_weekly s1
WHERE weekly_range_pct = (
    SELECT MAX(weekly_range_pct)
    FROM sector_weekly s2
    WHERE s2.week = s1.week
)
ORDER BY week;

-- 4. Nifty performance during HIGH VIX weeks
SELECT
    n.week,
    n.avg_close,
    n.weekly_range_pct,
    v.regime
FROM weekly_nifty n
JOIN (
    SELECT strftime('%Y-W%W', date) as week, regime
    FROM vix_regime
    GROUP BY week
    HAVING COUNT(*) = MAX(CASE WHEN regime = 'HIGH' THEN 1 ELSE 0 END)
) v ON n.week = v.week
WHERE v.regime = 'HIGH'
ORDER BY n.week;

-- 5. Monthly Nifty summary
SELECT
    strftime('%Y-%m', date) AS month,
    ROUND(AVG(close), 2) AS avg_close,
    ROUND(MAX(close), 2) AS monthly_high,
    ROUND(MIN(close), 2) AS monthly_low,
    ROUND((MAX(close) - MIN(close)) / MIN(close) * 100, 2) AS monthly_range_pct
FROM nifty_ohlcv
GROUP BY month
ORDER BY month;

-- 6. Sector leadership count per year
SELECT
    strftime('%Y', date) AS year,
    s.sector,
    COUNT(*) AS weeks_led
FROM sectoral s
JOIN (
    SELECT week, MAX(weekly_range_pct) as max_range
    FROM sector_weekly
    GROUP BY week
) top ON strftime('%Y-W%W', s.date) = top.week
JOIN sector_weekly sw ON sw.week = strftime('%Y-W%W', s.date)
    AND sw.sector = s.sector
    AND sw.weekly_range_pct = top.max_range
GROUP BY year, s.sector
ORDER BY year, weeks_led DESC;

-- 7. Nifty 52-week high and low by year
SELECT
    strftime('%Y', date) AS year,
    ROUND(MAX(close), 2) AS yearly_high,
    ROUND(MIN(close), 2) AS yearly_low,
    ROUND((MAX(close) - MIN(close)) / MIN(close) * 100, 2) AS yearly_range_pct
FROM nifty_ohlcv
GROUP BY year
ORDER BY year;

-- 8. FinBERT sentiment distribution
SELECT
    sentiment,
    COUNT(*) as weeks,
    ROUND(AVG(weekly_return), 2) as avg_return,
    ROUND(AVG(vix), 2) as avg_vix
FROM weekly_sentiment
GROUP BY sentiment
ORDER BY weeks DESC;
