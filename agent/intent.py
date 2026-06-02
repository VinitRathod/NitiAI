# agent/intent.py
import json
import ollama
from datetime import datetime
from .config import OLLAMA_MODEL as MODEL

# MODEL = "llama3.1"

SYSTEM_PROMPT = """You are an intent extraction engine for a retail analytics system.
Extract structured information from the user's question.

Return ONLY a valid JSON object — no explanation, no markdown, no extra text.
Use exactly these fields:

{
  "intent_type": "sales_drop" | "sales_spike" | "comparison" | "forecast" | "general",
  "cities": ['Quito', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra', 'Santo Domingo', 'Guaranda', 'Puyo', 'Ambato', 'Guayaquil', 'Salinas', 'Daule', 'Babahoyo', 'Quevedo', 'Playas', 'Libertad', 'Cuenca', 'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen'] or [],
  "states": ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura', 'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza', 'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja', 'El Oro', 'Esmeraldas', 'Manabi'] or [],
  "families": ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY', 'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'] or [],
  "stores": [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,  3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,  5, 50, 51, 52, 53, 54,  6, 7,  8,  9] or [],
  "date_start": "YYYY-MM-DD" or null,
  "date_end": "YYYY-MM-DD" or null,
  "metric": one of ["sales", "units", "promo_count", "oil_price"] or "sales",
  "compare_to": one of ["previous_period", "forecast", "another_city"] or null,
  "raw_question": the original question verbatim
}

Dataset date range: 2013-01-01 to 2017-08-15.
Today's date for resolving relative dates: """ + datetime.today().strftime("%Y-%m-%d") + """
Clamp all dates to the dataset range above.
"""

def extract_intent(question: str) -> dict:
    """Parse a natural language question into a structured intent dict."""
    response = ollama.chat(
        model=MODEL,
        format="json",
        options={"temperature": 0},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
    )

    raw = response["message"]["content"]

    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: safe defaults so the agent never crashes
        intent = {
            "intent_type": "general",
            "cities": [],
            "families": [],
            "stores": [],
            "date_start": "2013-01-01",
            "date_end": "2017-08-15",
            "metric": "sales",
            "compare_to": None,
        }

    intent["raw_question"] = question
    return intent