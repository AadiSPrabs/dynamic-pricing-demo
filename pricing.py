"""Pricing engine for the Streamlit demo.
Replicates the v6 Kaggle notebook's hybrid optimizer exactly:
  - LightGBM predicts demand LEVELS and MARKDOWN responses
  - dept-level econometric elasticity for price HIKES (the ML blind spot)
  - 5% materiality hurdle: a change must beat holding, else hold/wait
"""
import base64
import glob
import io
import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

DATA = os.path.join(os.path.dirname(__file__), "data")

FEATURES = ["sell_price", "prev_price", "markdown", "drop_depth", "price_vs_roll8",
            "lag_1", "lag_2", "lag_4", "lag_8", "roll_4", "roll_8",
            "month", "event_Sporting", "event_Cultural", "event_National",
            "event_Religious", "snap", "dept_id", "cat_id", "store_id"]
CATS = ["dept_id", "cat_id", "store_id"]
CANDIDATES = [1.00, 1.05, 0.95, 0.90, 0.85, 0.80, 0.70]

UNIT_COST_RATIO = 0.62
MIN_MARGIN = 0.10
MAX_DISCOUNT = 0.30
MIN_UPLIFT = 0.05

_model = None
_elas_map = None
_cat_levels = None


def _read_parts(pattern):
    """Reassemble chunked files (data/model/part*.txt, data/cards/part*.b64)."""
    paths = sorted(glob.glob(os.path.join(DATA, pattern)))
    assert paths, f"no chunks found for {pattern}"
    return "".join(open(p, encoding="utf-8").read() for p in paths)


def load_model():
    global _model
    if _model is None:
        model_str = _read_parts(os.path.join("model", "part*.txt"))
        _model = lgb.Booster(model_str=model_str)
    return _model


def elas_map():
    global _elas_map
    if _elas_map is None:
        e = pd.read_csv(os.path.join(DATA, "elasticity_by_dept.csv"))
        _elas_map = dict(zip(e["dept_id"].astype(str), e["elasticity"]))
    return _elas_map


def cat_levels():
    """Category levels matching the training panel's encoding."""
    global _cat_levels
    if _cat_levels is None:
        cards = load_cards()
        _cat_levels = {c: pd.Categorical(cards[c].astype(str)).categories for c in CATS}
    return _cat_levels


def load_cards():
    raw = base64.b64decode(_read_parts(os.path.join("cards", "part*.b64")))
    df = pd.read_parquet(io.BytesIO(raw))
    return df


def load_plot(name):
    """Plot PNG bytes for st.image (name like 'pred_vs_actual.png')."""
    raw = base64.b64decode(open(os.path.join(DATA, "plots", name + ".b64"),
                                encoding="utf-8").read())
    return raw


def _prep(df):
    """Build the model feature frame with the training-time categorical encoding."""
    X = df[FEATURES].copy()
    levels = cat_levels()
    for c in CATS:
        X[c] = pd.Categorical(X[c].astype(str), categories=levels[c])
    return X


def predict_units(X):
    return np.maximum(load_model().predict(_prep(X)), 0.0)


def demand_at_price(row, price):
    """Hybrid demand for one product-week row at a candidate price."""
    r = row.copy()
    base_price = float(row["sell_price"])
    mult = price / base_price if base_price else 1.0
    X = pd.DataFrame([{f: row[f] for f in FEATURES}])
    if mult < 1.0:
        X["sell_price"] = price
        X["prev_price"] = base_price
        X["markdown"] = 1
        X["drop_depth"] = 1 - mult
        pp = row["prev_price"]
        X["price_vs_roll8"] = price / pp if pd.notna(pp) and pp else 1.0
        return float(predict_units(X)[0])
    # baseline at current price (LightGBM's strength: volume levels)
    X["sell_price"] = base_price
    X["prev_price"] = base_price
    X["markdown"] = 0
    X["drop_depth"] = 0.0
    X["price_vs_roll8"] = 1.0
    base_dem = float(predict_units(X)[0])
    if mult > 1.0:
        e = elas_map().get(str(row["dept_id"]), -0.5)
        return base_dem * (mult ** e)
    return base_dem


def optimise(row):
    """Full hybrid optimizer for one row -> action dict (mirrors the notebook)."""
    base_price = float(row["sell_price"])
    cost = base_price * UNIT_COST_RATIO
    min_price = cost * (1 + MIN_MARGIN)
    results = []
    for mult in CANDIDATES:
        price = base_price * mult
        if mult < 1 - MAX_DISCOUNT - 1e-9 or price < min_price - 1e-9:
            continue
        dem = demand_at_price(row, price)
        profit = (price - cost) * dem
        results.append({"mult": mult, "price": price, "demand": dem, "profit": profit})
    if not results:
        return {"action": "wait_reassess", "mult": 1.0, "price": base_price,
                "demand": 0.0, "profit": 0.0, "reason": "no candidate satisfies constraints"}
    hold = next(r for r in results if r["mult"] == 1.0)
    best = max(results, key=lambda r: r["profit"])
    hurdle = hold["profit"] * (1 + MIN_UPLIFT)
    if best["mult"] == 1.0:
        action, reason = "protect_margin", "holding beats every priced alternative"
    elif best["profit"] > hurdle:
        if best["mult"] > 1.0:
            action = "test_increase"
            reason = f"higher price beats hold by {best['profit']/hold['profit']-1:.0%} (dept elasticity)"
        else:
            action = "apply_markdown"
            reason = f"{1-best['mult']:.0%} off beats hold by {best['profit']/hold['profit']-1:.0%}"
    elif best["mult"] < 1.0:
        best, action = hold, "wait_reassess"
        reason = "a markdown may be needed later; evidence too weak to act now"
    else:
        best, action = hold, "protect_margin"
        reason = "no priced alternative beats holding convincingly"
    return {"action": action, "mult": best["mult"], "price": best["price"],
            "demand": best["demand"], "profit": best["profit"], "reason": reason}


ACTION_LABELS = {
    "protect_margin": "Protect margin",
    "test_increase": "Test price increase",
    "apply_markdown": "Apply markdown",
    "wait_reassess": "Wait & reassess",
}

ACTION_DESCRIPTIONS = {
    "protect_margin": "Keep the current price. The numbers say a change isn't justified — don't touch it.",
    "test_increase": "Raise the price a little. Demand looks inelastic enough to hold.",
    "apply_markdown": "Cut the price. Expected volume gain outweighs the margin given up.",
    "wait_reassess": "Don't act yet. A markdown may be needed later — check again soon.",
}


def metrics():
    with open(os.path.join(DATA, "metrics.json")) as f:
        return json.load(f)
