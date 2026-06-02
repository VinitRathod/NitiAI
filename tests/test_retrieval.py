# tests/test_retrieval.py
import pytest

class TestRetrieval:

    def test_get_sales_by_city_returns_dataframe(self, registered_db):
        df = registered_db.get_sales_by_city("Quito", "2017-07-01", "2017-07-31")
        assert not df.empty
        assert "total_sales" in df.columns
        assert "date" in df.columns

    def test_get_sales_excludes_other_cities(self, registered_db):
        df = registered_db.get_sales_by_city("Quito", "2017-07-01", "2017-07-31")
        # Only stores 1 and 2 are in Quito — max 2 stores per day
        assert df["stores"].max() <= 2

    def test_get_oil_prices_in_range(self, registered_db):
        df = registered_db.get_oil_prices("2017-07-01", "2017-07-31")
        assert not df.empty
        assert "oil_price" in df.columns
        assert df["oil_price"].between(40, 60).all()

    def test_get_holidays_filters_by_city(self, registered_db):
        df = registered_db.get_holidays("Quito", "2017-07-01", "2017-08-31")
        # Only the Quito holiday should appear, not Guayaquil's
        assert len(df) == 1
        assert "Quito" in df["locale_name"].values[0]

    def test_get_promo_counts_returns_daily_totals(self, registered_db):
        df = registered_db.get_promo_counts("Quito", "2017-07-01", "2017-07-31")
        assert not df.empty
        assert "promo_count" in df.columns
        assert (df["promo_count"] >= 0).all()