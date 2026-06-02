# NitiAI — AI Analytics Dashboard, Forecasting & Decision Agent

End-to-end data science system for sales forecasting, interactive KPI 
monitoring, and natural language business insights.

## Features
- Automated data pipeline processing 3M rows with 20+ engineered features
- Four forecasting models benchmarked; XGBoost achieves MAPE 4.78% (48% improvement)
- Interactive Streamlit dashboard with holiday lift, heatmaps, and forecast tab
- Retail Decision Agent (Llama 3.1 via Ollama) — answers business questions in natural language
- Testing Pyramid: 24 pytest tests (unit + integration + e2e), 85%+ coverage

## Tech Stack
Python · Pandas · XGBoost · scikit-learn · Streamlit · Plotly · 
Llama 3.1 · Ollama · DuckDB · pytest · pytest-cov

## Run the dashboard
```bash
streamlit run dashboard.py
```

## Run tests
```bash
pytest tests/ -v --cov=agent --cov-report=term-missing
```
