# Dynamic Pricing & Markdown Optimization — Streamlit Demo

Interactive demo for the AIBE subject project. Runs the trained hybrid pricing
optimizer (LightGBM demand model + econometric price elasticity) on Walmart M5 data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io → New app → select the repo, `app.py`.
3. Deploy — no secrets or setup needed; all data ships in `data/`.

## What's inside

- `app.py` — the demo (decision cards, price simulator, model performance, about)
- `pricing.py` — hybrid optimizer replicating the Kaggle notebook (v6)
- `data/` — trained model (`lgbm_demand_full.txt`), 2,000 decision cards with
  feature rows (`cards_panel.parquet`), metrics, elasticities, plots

Full pipeline notebook:
https://www.kaggle.com/code/aadityasunilprabhu/dynamic-pricing-and-markdown-optimization-m5
