# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_sales_df():
    """Minimal sales DataFrame matching NitiAI schema."""
    dates = pd.date_range("2017-07-01", periods=31, freq="D")
    rows = []
    for d in dates:
        for store in [1, 2, 3]:
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "store_nbr": store,
                "family": "GROCERY I",
                "sales": np.random.uniform(800, 1200),
                "onpromotion": np.random.randint(0, 10),
            })
    return pd.DataFrame(rows)

@pytest.fixture
def sample_stores_df():
    return pd.DataFrame([
        {"store_nbr": 1, "city": "Quito",     "type": "A", "cluster": 1},
        {"store_nbr": 2, "city": "Quito",     "type": "B", "cluster": 2},
        {"store_nbr": 3, "city": "Guayaquil", "type": "A", "cluster": 1},
    ])

@pytest.fixture
def sample_oil_df():
    dates = pd.date_range("2017-07-01", periods=31, freq="D")
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "dcoilwtico": np.linspace(45.0, 48.0, 31),
    })

@pytest.fixture
def sample_holidays_df():
    return pd.DataFrame([
        {"date": "2017-07-24", "description": "Fundacion de Guayaquil",
         "type": "Holiday", "locale": "Regional", "locale_name": "Guayaquil"},
        {"date": "2017-08-10", "description": "Independencia de Quito",
         "type": "Holiday", "locale": "Regional", "locale_name": "Quito"},
    ])

@pytest.fixture
def sample_forecast_df():
    dates = pd.date_range("2017-07-01", periods=31, freq="D")
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "predicted_total_sales": np.random.uniform(2400, 3200, 31),
    })

@pytest.fixture
def registered_db(sample_sales_df, sample_stores_df, sample_oil_df,
                   sample_holidays_df, sample_forecast_df):
    """Set up the DuckDB in-memory database with all sample data."""
    from agent import retrieval
    retrieval.setup_db(
        sample_sales_df, sample_stores_df, sample_oil_df,
        sample_holidays_df, sample_forecast_df,
    )
    return retrieval

@pytest.fixture
def valid_intent():
    return {
        "intent_type": "sales_drop",
        "cities": ["Quito"],
        "families": [],
        "stores": [],
        "date_start": "2017-07-01",
        "date_end": "2017-07-31",
        "metric": "sales",
        "compare_to": "previous_period",
        "raw_question": "Why did sales drop in Quito in July 2017?",
    }