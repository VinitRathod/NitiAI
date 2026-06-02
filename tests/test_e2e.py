# tests/test_e2e.py
import pytest

# Mark every test in this file as e2e — run separately with: pytest -m e2e
pytestmark = pytest.mark.e2e

def ollama_is_running():
    """Check if Ollama server is up before running e2e tests."""
    try:
        import ollama
        ollama.chat(model="llama3.1", messages=[{"role": "user", "content": "hi"}])
        return True
    except Exception:
        return False

skip_if_no_ollama = pytest.mark.skipif(
    not ollama_is_running(),
    reason="Ollama not running — start with `ollama serve`"
)

class TestEndToEnd:

    @skip_if_no_ollama
    def test_sales_drop_question_returns_structured_response(self, registered_db):
        from agent.agent import run
        answer, findings = run("Why did sales drop in Quito in July 2017?")

        assert isinstance(answer, str)
        assert len(answer) > 50
        assert "What happened" in answer or "sales" in answer.lower()

    @skip_if_no_ollama
    def test_comparison_question_includes_two_cities(self, registered_db):
        from agent.agent import run
        answer, findings = run("Compare Quito vs Guayaquil in July 2017")

        assert "Quito" in answer or "Guayaquil" in answer
        assert "compare_total_sales" in findings

    @skip_if_no_ollama
    def test_response_mentions_no_invented_cities(self, registered_db):
        """LLM must not hallucinate city names not in the question."""
        from agent.agent import run
        answer, _ = run("How are sales in Quito doing?")

        invented_cities = ["Lima", "Bogota", "Buenos Aires", "Santiago"]
        for city in invented_cities:
            assert city not in answer, f"LLM hallucinated city: {city}"