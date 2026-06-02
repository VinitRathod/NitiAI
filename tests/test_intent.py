# tests/test_intent.py
import json
import pytest
from unittest.mock import patch, MagicMock

def make_ollama_response(payload: dict) -> dict:
    """Helper: build a fake ollama.chat() return value."""
    return {"message": {"content": json.dumps(payload)}}

class TestExtractIntent:

    def test_basic_sales_drop_quito(self):
        payload = {
            "intent_type": "sales_drop",
            "cities": ["Quito"],
            "families": [],
            "stores": [],
            "date_start": "2017-07-01",
            "date_end": "2017-07-31",
            "metric": "sales",
            "compare_to": "previous_period",
        }
        with patch("agent.intent.ollama.chat", return_value=make_ollama_response(payload)):
            from agent.intent import extract_intent
            result = extract_intent("Why did sales drop in Quito in July 2017?")

        assert result["intent_type"] == "sales_drop"
        assert "Quito" in result["cities"]
        assert result["date_start"] == "2017-07-01"
        assert result["raw_question"] == "Why did sales drop in Quito in July 2017?"

    def test_comparison_two_cities(self):
        payload = {
            "intent_type": "comparison",
            "cities": ["Quito", "Guayaquil"],
            "families": [],
            "stores": [],
            "date_start": "2017-07-01",
            "date_end": "2017-07-31",
            "metric": "sales",
            "compare_to": "another_city",
        }
        with patch("agent.intent.ollama.chat", return_value=make_ollama_response(payload)):
            from agent.intent import extract_intent
            result = extract_intent("Compare Quito vs Guayaquil in July 2017")

        assert result["intent_type"] == "comparison"
        assert len(result["cities"]) == 2
        assert "Guayaquil" in result["cities"]

    def test_raw_question_always_preserved(self):
        payload = {"intent_type": "general", "cities": [], "families": [],
                   "stores": [], "date_start": None, "date_end": None,
                   "metric": "sales", "compare_to": None}
        with patch("agent.intent.ollama.chat", return_value=make_ollama_response(payload)):
            from agent.intent import extract_intent
            q = "How are sales doing?"
            result = extract_intent(q)
        assert result["raw_question"] == q

    def test_broken_json_returns_safe_defaults(self):
        """If Ollama returns invalid JSON, the agent must not crash."""
        bad_response = {"message": {"content": "Sorry, I cannot answer that."}}
        with patch("agent.intent.ollama.chat", return_value=bad_response):
            from agent.intent import extract_intent
            result = extract_intent("gibberish query")

        assert "intent_type" in result
        assert result["intent_type"] == "general"
        assert isinstance(result["cities"], list)

    def test_product_family_extracted(self):
        payload = {
            "intent_type": "sales_drop", "cities": ["Quito"],
            "families": ["GROCERY I", "BEVERAGES"],
            "stores": [], "date_start": "2017-08-01", "date_end": "2017-08-15",
            "metric": "sales", "compare_to": None,
        }
        with patch("agent.intent.ollama.chat", return_value=make_ollama_response(payload)):
            from agent.intent import extract_intent
            result = extract_intent("Why did GROCERY I and BEVERAGES drop in Quito?")
        assert "GROCERY I" in result["families"]

    def test_date_fields_are_strings_or_none(self):
        payload = {"intent_type": "general", "cities": [], "families": [],
                   "stores": [], "date_start": None, "date_end": None,
                   "metric": "sales", "compare_to": None}
        with patch("agent.intent.ollama.chat", return_value=make_ollama_response(payload)):
            from agent.intent import extract_intent
            result = extract_intent("What is the overall trend?")
        assert result["date_start"] is None or isinstance(result["date_start"], str)
        assert result["date_end"]   is None or isinstance(result["date_end"], str)

    def test_ollama_called_once_per_question(self):
        payload = {"intent_type": "general", "cities": [], "families": [],
                   "stores": [], "date_start": None, "date_end": None,
                   "metric": "sales", "compare_to": None}
        with patch("agent.intent.ollama.chat", return_value=make_ollama_response(payload)) as mock_chat:
            from agent.intent import extract_intent
            extract_intent("test question")
        mock_chat.assert_called_once()