# agent/response.py
import json
import os
import ollama
from .config import OLLAMA_MODEL as MODEL

# MODEL = "llama3.1"

SYSTEM_PROMPT = """You are NitiAI, an intelligent retail analytics advisor for Ecuadorian grocery stores.
You speak like a smart, direct business consultant — not a data scientist.

Structure your response exactly like this (use these bold headers):

**What happened**
2 sentences max. State the key finding with specific numbers.

**Why it happened**
Only include a bullet if the finding shows a change greater than 5 percentage or a holiday is present/absent.
- Holiday impact: ...
- Promotion activity: ...
- Oil price effect: ...
- Store-level issues: ...

**What to do**
Exactly 3 action items. Be specific: name cities, store numbers, product families, and timing.

Rules you must follow:
- Never invent numbers not present in the findings JSON.
- Never mention a factor unless the data shows it is significant.
- Total response must be under 250 words.
- Do not use code blocks, JSON, or markdown tables.
- Do not add any text before "**What happened**".
"""

def generate_response(question: str, findings: dict) -> str:
    """Generate a natural language business response from analysis findings."""
    # import json
    findings_str = json.dumps(findings, indent=2, default=str)

    user_message = (
        f"User question: {question}\n\n"
        f"Analysis findings:\n{findings_str}\n\n"
        "Now write the NitiAI response following the exact format in your instructions."
    )

    response = ollama.chat(
        model=MODEL,
        options={"temperature": 0.2, "num_predict": 500},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    return response["message"]["content"]