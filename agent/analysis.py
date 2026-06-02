# agent/analysis.py
import pandas as pd
import numpy as np
from . import retrieval

def analyze(intent: dict) -> dict:
    """
    Run all relevant analyses for the given intent.
    Returns a structured findings dict consumed by response.py.
    """
    city = intent["cities"][0] if intent["cities"] else "Quito"
    date_start = intent["date_start"] or "2017-07-01"
    date_end   = intent["date_end"]   or "2017-08-15"
    compare_city = intent["cities"][1] if len(intent.get("cities", [])) > 1 else None

    findings = {
        "city": city,
        "date_start": date_start,
        "date_end": date_end,
        "intent_type": intent["intent_type"],
    }

    # ── Sales trend ─────────────────────────────────────────────────────────
    sales = retrieval.get_sales_by_city(city, date_start, date_end)
    findings["sales_data"] = sales.to_dict("records")
    findings["total_sales"] = round(sales["total_sales"].sum(), 0)
    findings["avg_daily_sales"] = round(sales["total_sales"].mean(), 0)

    # ── Forecast vs actual (residuals) ──────────────────────────────────────
    fva = retrieval.get_forecast_vs_actual(city, date_start, date_end)
    if not fva.empty and "forecast" in fva.columns:
        fva["residual_pct"] = (fva["actual"] - fva["forecast"]) / fva["forecast"] * 100
        findings["mean_residual_pct"] = round(fva["residual_pct"].mean(), 1)
        findings["forecast_data"] = fva.to_dict("records")
    else:
        findings["mean_residual_pct"] = None
        findings["forecast_data"] = []

    # ── Holiday impact ──────────────────────────────────────────────────────
    holidays = retrieval.get_holidays(city, date_start, date_end)
    findings["holiday_count"] = len(holidays)
    findings["holidays"] = holidays.to_dict("records") if not holidays.empty else []

    # ── Promotion delta ─────────────────────────────────────────────────────
    # Compare current period promos to previous same-length period
    period_days = (pd.to_datetime(date_end) - pd.to_datetime(date_start)).days
    prev_start  = (pd.to_datetime(date_start) - pd.Timedelta(days=period_days)).strftime("%Y-%m-%d")
    prev_end    = (pd.to_datetime(date_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    promos_curr = retrieval.get_promo_counts(city, date_start, date_end)
    promos_prev = retrieval.get_promo_counts(city, prev_start, prev_end)

    curr_promos = promos_curr["promo_count"].sum() if not promos_curr.empty else 0
    prev_promos = promos_prev["promo_count"].sum() if not promos_prev.empty else 0
    findings["promo_current"]  = int(curr_promos)
    findings["promo_previous"] = int(prev_promos)
    findings["promo_delta_pct"] = (
        round((curr_promos - prev_promos) / prev_promos * 100, 1)
        if prev_promos > 0 else None
    )

    # ── Oil price trend ─────────────────────────────────────────────────────
    oil = retrieval.get_oil_prices(date_start, date_end)
    if not oil.empty:
        findings["oil_mean"]  = round(oil["oil_price"].mean(), 2)
        findings["oil_start"] = round(oil["oil_price"].iloc[0], 2)
        findings["oil_end"]   = round(oil["oil_price"].iloc[-1], 2)
        findings["oil_delta_pct"] = round(
            (findings["oil_end"] - findings["oil_start"]) / findings["oil_start"] * 100, 1
        )
    else:
        findings["oil_mean"] = findings["oil_start"] = findings["oil_end"] = findings["oil_delta_pct"] = None

    # ── Underperforming stores ───────────────────────────────────────────────
    bad_stores = retrieval.get_underperforming_stores(city, date_start, date_end)
    findings["underperforming_stores"] = bad_stores.to_dict("records") if not bad_stores.empty else []

    # ── Optional: comparison city ────────────────────────────────────────────
    if compare_city:
        comp_sales = retrieval.get_sales_by_city(compare_city, date_start, date_end)
        findings["compare_city"] = compare_city
        findings["compare_total_sales"] = round(comp_sales["total_sales"].sum(), 0)
        findings["compare_avg_daily"]   = round(comp_sales["total_sales"].mean(), 0)

    return findings