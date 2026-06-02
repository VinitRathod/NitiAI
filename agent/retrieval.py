# agent/retrieval.py
import duckdb
import pandas as pd
from typing import Optional

# ── Bootstrap ──────────────────────────────────────────────────────────────
# Call setup_db() once at app startup, passing your existing DataFrames.
_con: Optional[duckdb.DuckDBPyConnection] = None

def setup_db(
    df_sales: pd.DataFrame,
    df_stores: pd.DataFrame,
    df_oil: pd.DataFrame,
    df_holidays: pd.DataFrame,
    df_forecast: pd.DataFrame,
):
    """Register DataFrames as DuckDB in-memory views."""
    global _con
    _con = duckdb.connect(database=":memory:")
    _con.register("sales", df_sales)
    _con.register("stores", df_stores)
    _con.register("oil", df_oil)
    _con.register("holidays", df_holidays)
    _con.register("forecast", df_forecast)

def _q(sql: str) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""
    return _con.execute(sql).df()

# ── Query helpers ───────────────────────────────────────────────────────────

def get_sales_by_city(city: str, date_start: str, date_end: str) -> pd.DataFrame:
    return _q(f"""
        SELECT s.date, SUM(s.sales) AS total_sales, COUNT(DISTINCT s.store_nbr) AS stores
        FROM sales s
        JOIN stores st ON s.store_nbr = st.store_nbr
        WHERE st.city = '{city}'
          AND s.date BETWEEN '{date_start}' AND '{date_end}'
        GROUP BY s.date
        ORDER BY s.date
    """)

def get_promo_counts(city: str, date_start: str, date_end: str) -> pd.DataFrame:
    return _q(f"""
        SELECT s.date, SUM(s.onpromotion) AS promo_count
        FROM sales s
        JOIN stores st ON s.store_nbr = st.store_nbr
        WHERE st.city = '{city}'
          AND s.date BETWEEN '{date_start}' AND '{date_end}'
        GROUP BY s.date
        ORDER BY s.date
    """)

def get_oil_prices(date_start: str, date_end: str) -> pd.DataFrame:
    return _q(f"""
        SELECT date, dcoilwtico AS oil_price
        FROM oil
        WHERE date BETWEEN '{date_start}' AND '{date_end}'
        ORDER BY date
    """)

def get_holidays(city: str, date_start: str, date_end: str) -> pd.DataFrame:
    return _q(f"""
        SELECT date, description, type, locale_name, locale
        FROM holidays
        WHERE (locale_name = '{city}' OR locale = 'National')
          AND date BETWEEN '{date_start}' AND '{date_end}'
    """)

def get_forecast_vs_actual(city: str, date_start: str, date_end: str) -> pd.DataFrame:
    return _q(f"""
        SELECT f.date,
               f.predicted_total_sales AS forecast,
               SUM(s.sales) AS actual
        FROM forecast f
        LEFT JOIN sales s ON f.date = s.date
        LEFT JOIN stores st ON s.store_nbr = st.store_nbr
          AND st.city = '{city}'
        WHERE f.date BETWEEN '{date_start}' AND '{date_end}'
        GROUP BY f.date, f.predicted_total_sales
        ORDER BY f.date
    """)

def get_underperforming_stores(
    city: str, date_start: str, date_end: str, threshold: float = -0.10
) -> pd.DataFrame:
    """Returns stores whose actual sales are > threshold% below their period average."""
    return _q(f"""
        WITH store_totals AS (
            SELECT s.store_nbr,
                   SUM(s.sales) AS total_sales
            FROM sales s
            JOIN stores st ON s.store_nbr = st.store_nbr
            WHERE st.city = '{city}'
              AND s.date BETWEEN '{date_start}' AND '{date_end}'
            GROUP BY s.store_nbr
        ),
        city_avg AS (
            SELECT AVG(total_sales) AS avg_sales FROM store_totals
        )
        SELECT st.store_nbr,
               st.type,
               st.cluster,
               t.total_sales,
               ROUND((t.total_sales - c.avg_sales) / c.avg_sales * 100, 1) AS pct_vs_avg
        FROM store_totals t
        CROSS JOIN city_avg c
        JOIN stores st ON t.store_nbr = st.store_nbr
        WHERE (t.total_sales - c.avg_sales) / c.avg_sales < {threshold}
        ORDER BY pct_vs_avg ASC
    """)