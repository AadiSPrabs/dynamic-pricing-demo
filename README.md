# Dynamic Pricing & Markdown Optimization — Streamlit Demo

Interactive demo for the AIBE subject project. Runs the trained hybrid pricing
optimizer (LightGBM demand model + econometric price elasticity) on Walmart M5 data.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo.
2. Go to share.streamlit.io → New app → select the repo, `app.py`.
3. Deploy — no secrets or setup needed; all data ships in `data/`.

## What's inside

- `app.py` — the demo: Home, Recommendations, Price Simulator, How It Works, About
- `pricing.py` — hybrid optimizer replicating the Kaggle notebook (v6)
- `data/` — trained model chunks (`model/`), 2,000 decision cards with feature
  rows (`cards/`), metrics, elasticities, plots (`plots/`). Large files are split
  into <100KB chunks and reassembled at runtime by `pricing.py`.

Full pipeline notebook:
https://www.kaggle.com/code/aadityasunilprabhu/dynamic-pricing-and-markdown-optimization-m5
