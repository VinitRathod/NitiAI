# agent/agent.py
from . import intent as intent_module
from . import analysis as analysis_module
from . import response as response_module

def run(question: str) -> tuple[str, dict]:
    """
    Run the full agent pipeline.
    Returns (natural_language_response, findings_dict).
    The findings dict is used by Streamlit to render charts.
    """
    # 1. Parse intent
    intent = intent_module.extract_intent(question)

    # 2. Retrieve + analyse
    findings = analysis_module.analyze(intent)

    # 3. Generate response
    answer = response_module.generate_response(question, findings)

    return answer, findings