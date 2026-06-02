# tests/test_analysis.py
import pytest
from unittest.mock import patch

class TestAnalysis:

    def test_analyze_returns_required_keys(self, registered_db, valid_intent):
        from agent.analysis import analyze
        findings = analyze(valid_intent)

        required = [
            "city", "date_start", "date_end", "intent_type",
            "total_sales", "avg_daily_sales",
            "holiday_count", "promo_current", "promo_previous",
        ]
        for key in required:
            assert key in findings, f"Missing key: {key}"

    def test_total_sales_is_positive(self, registered_db, valid_intent):
        from agent.analysis import analyze
        findings = analyze(valid_intent)
        assert findings["total_sales"] > 0

    def test_promo_delta_is_float_or_none(self, registered_db, valid_intent):
        from agent.analysis import analyze
        findings = analyze(valid_intent)
        delta = findings["promo_delta_pct"]
        assert delta is None or isinstance(delta, float)

    def test_oil_fields_populated(self, registered_db, valid_intent):
        from agent.analysis import analyze
        findings = analyze(valid_intent)
        assert findings["oil_mean"] is not None
        assert 40 < findings["oil_mean"] < 60

    def test_comparison_city_included_when_two_cities(self, registered_db):
        from agent.analysis import analyze
        intent = {
            "intent_type": "comparison",
            "cities": ["Quito", "Guayaquil"],
            "families": [], "stores": [],
            "date_start": "2017-07-01", "date_end": "2017-07-31",
            "metric": "sales", "compare_to": "another_city",
            "raw_question": "Compare Quito vs Guayaquil",
        }
        findings = analyze(intent)
        assert "compare_city" in findings
        assert findings["compare_city"] == "Guayaquil"
        assert "compare_total_sales" in findings

    def test_underperforming_stores_is_list(self, registered_db, valid_intent):
        from agent.analysis import analyze
        findings = analyze(valid_intent)
        assert isinstance(findings["underperforming_stores"], list)