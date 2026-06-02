# tests/test_integration.py
import json
import pprint
import pytest
from unittest.mock import patch

def make_ollama_response(payload):
    return {"message": {"content": json.dumps(payload) if isinstance(payload, dict) else payload}}

MOCK_INTENT = {
    "intent_type": "sales_drop",
    "cities": ["Quito"],
    "families": [],
    "stores": [],
    "date_start": "2017-07-01",
    "date_end": "2017-07-31",
    "metric": "sales",
    "compare_to": "previous_period",
}

class TestIntentToAnalysis:

    def test_intent_output_feeds_analysis(self, registered_db):
        """Intent dict from extract_intent() must work directly in analyze()."""
        from agent.intent import extract_intent
        from agent.analysis import analyze

        with patch("agent.intent.ollama.chat", return_value=make_ollama_response(MOCK_INTENT)):
            intent = extract_intent("Why did sales drop in Quito?")

        findings = analyze(intent)
        # print(findings)  # For debugging if test fails
        assert findings["city"] == "Quito"
        assert findings["total_sales"] > 0

    def test_analysis_output_feeds_response(self, registered_db):
        """Findings dict from analyze() must not crash generate_response()."""
        from agent.analysis import analyze
        from agent.response import generate_response

        findings = analyze(MOCK_INTENT)

        mock_llm_reply = {"message": {"content": "**What happened**\nSales dropped 10%.\n**Why it happened**\n- Promotions fell 30%.\n**What to do**\n1. Run a promo.\n2. Check stores.\n3. Review oil prices."}}

        with patch("agent.response.ollama.chat", return_value=mock_llm_reply):
            answer = generate_response("Why did sales drop?", findings)

        assert isinstance(answer, str)
        assert len(answer) > 20

class TestFullPipelineIntegration:

    def test_run_returns_tuple(self, registered_db):
        """agent.run() must return (str, dict)."""
        from agent.agent import run

        with patch("agent.intent.extract_intent", return_value=MOCK_INTENT):
        # with patch("agent.intent.ollama.chat", return_value=make_ollama_response(MOCK_INTENT)):
            with patch("agent.response.ollama.chat", return_value={
                "message": {"content": "**What happened**\nDropped.\n**Why it happened**\n- Promos down.\n**What to do**\n1. Act.\n2. Act.\n3. Act."}
            }):
                answer, findings = run("Why did sales drop in Quito in July 2017?")

        assert isinstance(answer, str)
        assert isinstance(findings, dict)

    def test_findings_dict_has_chart_data(self, registered_db):
        """Streamlit chart rendering depends on forecast_data key existing."""
        from agent.agent import run

        with patch("agent.intent.extract_intent", return_value=MOCK_INTENT):
            with patch("agent.response.ollama.chat", return_value={
                "message": {"content": "Some response."}
            }):
                _, findings = run("Why did sales drop?")

        assert "forecast_data" in findings
        assert isinstance(findings["forecast_data"], list)

    def test_agent_handles_unknown_city_gracefully(self, registered_db):
        """Agent must not crash when given a city not in the data."""
        from agent.agent import run
        bad_intent = {**MOCK_INTENT, "cities": ["Atlantis"]}

        with patch("agent.intent.extract_intent", return_value=bad_intent):
            with patch("agent.response.ollama.chat", return_value={
                "message": {"content": "No data found."}
            }):
                answer, findings = run("Sales in Atlantis?")

        assert isinstance(answer, str)
        assert findings["total_sales"] == 0 or isinstance(findings["total_sales"], float)

    def test_comparison_intent_includes_both_city_totals(self, registered_db):
        """When two cities are given, findings must have both totals for the chart."""
        from agent.agent import run
        compare_intent = {
            "intent_type": "comparison",
            "cities": ["Quito", "Guayaquil"],
            "families": [], "stores": [],
            "date_start": "2017-07-01",
            "date_end": "2017-07-31",
            "metric": "sales",
            "compare_to": "another_city",
            "raw_question": "Compare Quito vs Guayaquil",
        }

        with patch("agent.intent.extract_intent", return_value=compare_intent):
            with patch("agent.response.ollama.chat", return_value={
                "message": {"content": "Comparison response."}
            }):
                _, findings = run("Compare Quito vs Guayaquil")
        # with patch("agent.intent.ollama.chat", return_value=make_ollama_response(compare_intent)):
        #     with patch("agent.response.ollama.chat", return_value={
        #         "message": {"content": "Comparison response."}
        #     }):
        #         _, findings = run("Compare Quito vs Guayaquil")

        assert findings["intent_type"] == "comparison", \
            f"intent_type was '{findings['intent_type']}' — mock not applied"
        assert "compare_city" in findings, \
            f"compare_city missing. Keys: {list(findings.keys())}"
        assert "compare_total_sales" in findings
        assert findings["compare_city"] == "Guayaquil"