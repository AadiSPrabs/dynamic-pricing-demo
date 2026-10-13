"""Dynamic Pricing and Markdown Optimization System — Streamlit demo.
AIBE subject project (solo) by Aadi. Walmart M5 data.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

import pricing as P

st.set_page_config(
    page_title="Dynamic Pricing & Markdown Optimization",
    page_icon="🏷️",
    layout="wide",
)

DATA = os.path.join(os.path.dirname(__file__), "data")

ACTION_COLORS = {
    "protect_margin": "#3b82f6",
    "test_increase": "#16a34a",
    "apply_markdown": "#ea580c",
    "wait_reassess": "#6b7280",
}


@st.cache_data
def get_cards():
    df = P.load_cards()
    df["category"] = df["cat_id"]
    df["label"] = df["item_id"] + " @ " + df["store_id"] + " (wk " + df["wm_yr_wk"].astype(str) + ")"
    return df


@st.cache_data
def get_metrics():
    return P.metrics()


def action_badge(action):
    label = P.ACTION_LABELS[action]
    color = ACTION_COLORS[action]
    return (f'<span style="background:{color};color:white;padding:4px 14px;'
            f'border-radius:20px;font-weight:600;">{label}</span>')


def render_card(row, res):
    disc = (1 - res["mult"]) if res["mult"] < 1 else 0
    price_line = (f"${row['sell_price']:.2f} → **${res['price']:.2f}**"
                  + (f" ({disc:.0%} markdown)" if disc else " (hold)"))
    st.markdown(f"### {row['item_id']} @ {row['store_id']} &nbsp; {action_badge(res['action'])}",
                unsafe_allow_html=True)
    st.caption(f"Week {int(row['wm_yr_wk'])} · {row['dept_id']} · {row['cat_id']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", price_line)
    c2.metric("Expected demand", f"{res['demand']:.1f} units/week")
    c3.metric("Expected profit", f"${res['profit']:.2f}/week")
    st.markdown(f"**Why this action:** {res['reason']}.")
    st.markdown(P.ACTION_DESCRIPTIONS[res["action"]])
    with st.expander("What could go wrong"):
        st.write("Demand may not respond as modelled — markdowns here are inferred from "
                 "price drops (not labelled promotions), and inventory/cost are declared "
                 "assumptions, not observed data. Treat the profit figure as directional, "
                 "not a guarantee.")
    st.info("**Next review:** re-evaluate after 2 weeks, or sooner if sales deviate.")


# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("🏷️ Pricing Optimizer")
st.sidebar.caption("AIBE project · Walmart M5")
page = st.sidebar.radio("Go to", ["Decision Cards", "Price Simulator",
                                  "Model Performance", "About"])

cards = get_cards()
metrics = get_metrics()

# ------------------------------------------------------- 1. decision cards ---
if page == "Decision Cards":
    st.title("Pricing Decision Cards")
    st.write("Every product-week gets one recommended action. "
             "Filter the list, pick a product, and see the full card.")

    f1, f2, f3 = st.columns(3)
    with f1:
        actions = st.multiselect("Action",
                                 options=list(P.ACTION_LABELS),
                                 format_func=lambda a: P.ACTION_LABELS[a],
                                 default=list(P.ACTION_LABELS))
    with f2:
        cats = st.multiselect("Category", options=sorted(cards["category"].unique()),
                              default=sorted(cards["category"].unique()))
    with f3:
        q = st.text_input("Search item id", "")

    view = cards[cards["action"].isin(actions) & cards["category"].isin(cats)]
    if q:
        view = view[view["item_id"].str.contains(q.upper(), na=False)]
    view = view.reset_index(drop=True)

    mix = view["action"].value_counts()
    mcols = st.columns(4)
    for i, a in enumerate(P.ACTION_LABELS):
        mcols[i].metric(P.ACTION_LABELS[a], f"{int(mix.get(a, 0)):,}")

    st.dataframe(
        view[["item_id", "store_id", "category", "sell_price", "action",
              "price", "demand", "profit"]].rename(columns={
                  "item_id": "Item", "store_id": "Store", "category": "Category",
                  "sell_price": "Current $", "action": "Action",
                  "price": "Recommended $", "demand": "Exp. demand/wk",
                  "profit": "Exp. profit/wk"}),
        width="stretch", height=320,
    )

    if len(view):
        pick = st.selectbox("Show card for", view["label"].tolist())
        row = view[view["label"] == pick].iloc[0]
        with st.spinner("Running optimizer..."):
            res = P.optimise(row)
        st.divider()
        render_card(row, res)
    else:
        st.warning("No products match the filters.")

# ------------------------------------------------------ 2. price simulator ---
elif page == "Price Simulator":
    st.title("Price Simulator")
    st.write("Pick a product and move the price slider. The hybrid demand model "
             "(LightGBM for markdowns, econometric elasticity for hikes) predicts "
             "demand and profit at each price.")

    pick = st.selectbox("Product", cards["label"].tolist(), key="sim_pick")
    row = cards[cards["label"] == pick].iloc[0]
    base = float(row["sell_price"])
    cost = base * P.UNIT_COST_RATIO

    with st.spinner("Running optimizer..."):
        res = P.optimise(row)

    st.markdown(f"**{row['item_id']}** @ {row['store_id']} · {row['dept_id']} &nbsp; "
                f"{action_badge(res['action'])}", unsafe_allow_html=True)
    st.caption(f"Current price ${base:.2f} · unit cost ${cost:.2f} · "
               f"recommended ${res['price']:.2f} ({P.ACTION_LABELS[res['action']]})")

    mult = st.slider("Price multiplier", 0.70, 1.10, 1.00, 0.01,
                     help="1.00 = current price")
    price = base * mult
    dem = P.demand_at_price(row, price)
    profit = (price - cost) * dem
    k1, k2, k3 = st.columns(3)
    k1.metric("Price", f"${price:.2f}")
    k2.metric("Predicted demand", f"{dem:.1f} units/week")
    k3.metric("Expected profit", f"${profit:.2f}/week",
              delta=f"{profit - res['profit']:+.2f} vs recommended")

    grid = np.linspace(0.70 * base, 1.10 * base, 41)
    ds = np.array([P.demand_at_price(row, p) for p in grid])
    ps = (grid - cost) * ds
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=ds, name="Demand (units/wk)", yaxis="y1",
                             line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(x=grid, y=ps, name="Profit ($/wk)", yaxis="y2",
                             line=dict(color="#16a34a")))
    for x, name, col, dash in [(base, "current", "#6b7280", "dot"),
                               (res["price"], "recommended", "#ea580c", "dash")]:
        fig.add_vline(x=x, line_dash=dash, line_color=col,
                      annotation_text=name, annotation_position="top")
    fig.update_layout(
        xaxis_title="Price ($)",
        yaxis=dict(title="Demand (units/week)", side="left"),
        yaxis2=dict(title="Profit ($/week)", side="right", overlaying="y"),
        legend=dict(orientation="h", y=1.08), height=420, margin=dict(t=40),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Demand left axis · profit right axis. The model responds strongly to "
               "markdowns (learned from 33.6k price-drop events); hike responses use "
               "the department econometric elasticity.")

# ----------------------------------------------------- 3. model performance ---
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Demand model trained on all 30,490 item-store series (Walmart M5).")

    st.subheader("Forecast accuracy")
    perf = pd.DataFrame([
        {"Model": "Naive (last week)", "MAE": metrics["naive (lag_1)"]["mae"],
         "RMSE": metrics["naive (lag_1)"]["rmse"]},
        {"Model": "Linear regression", "MAE": metrics["linear regression"]["mae"],
         "RMSE": metrics["linear regression"]["rmse"]},
        {"Model": "LightGBM (validation)", "MAE": metrics["LightGBM (val)"]["mae"],
         "RMSE": metrics["LightGBM (val)"]["rmse"]},
        {"Model": "LightGBM (unseen test weeks)",
         "MAE": metrics["LightGBM (TEST - unseen weeks)"]["mae"],
         "RMSE": metrics["LightGBM (TEST - unseen weeks)"]["rmse"]},
    ])
    st.dataframe(perf.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}"}),
                 width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Predicted vs actual")
        st.image(P.load_plot("pred_vs_actual.png"))
    with c2:
        st.subheader("What the model relies on")
        st.image(P.load_plot("feat_importance.png"))
    st.caption("Sales history (lags, rolling means) dominates; price features matter "
               "less — one reason hike responses use the econometric model instead.")

    st.subheader("Price elasticity by department")
    elas = pd.read_csv(os.path.join(DATA, "elasticity_by_dept.csv"))
    st.dataframe(elas.rename(columns={"dept_id": "Department", "elasticity": "Elasticity",
                                      "r2": "R²", "n": "Observations"})
                 .style.format({"Elasticity": "{:.3f}", "R²": "{:.3f}"}),
                 width="stretch", hide_index=True)
    st.caption("All departments are inelastic (|elasticity| < 1): a 5% price rise costs "
               "only 1–4% of volume — the economic basis for most increase recommendations.")

    st.subheader("Policy backtest (model-based)")
    pol = metrics["policy"]
    p1, p2, p3 = st.columns(3)
    p1.metric("Optimizer profit/week", f"${pol['optimizer_profit_per_week']:,.2f}")
    p2.metric("Hold-price profit/week", f"${pol['hold_profit_per_week']:,.2f}")
    p3.metric("Uplift", f"+{pol['uplift']:.1%}")
    st.caption("Directional comparison under the demand model — not a causal claim. "
               "Real markdown outcomes need a live experiment.")

    with st.expander("Declared synthetic assumptions"):
        asm = pd.read_json(os.path.join(DATA, "business_assumptions.json"), typ="series")
        st.write(asm.to_dict())
        st.caption("M5 has no inventory or cost fields — these placeholders stand in, "
                   "and every downstream result is labelled accordingly.")

# ------------------------------------------------------------------ 4. about ---
else:
    st.title("About this system")
    st.write("""
**Dynamic Pricing and Markdown Optimization System** — a pricing decision-support
prototype for small and medium retailers, built on Walmart M5 data.

Instead of another sales dashboard, it answers one question per product per week:
**what should I do about the price?** — protect the margin, test an increase,
apply a markdown, or wait and reassess. Each answer comes as an explainable
**Pricing Decision Card**: the action, the economics, why, what could go wrong,
and when to review next.
""")
    st.subheader("Pipeline")
    st.write("Data validation → exploratory analysis → weekly feature panel → "
             "demand baselines → global LightGBM demand model → department price "
             "elasticity → hybrid optimizer → decision cards → model-based backtest.")
    st.subheader("The hybrid price response")
    st.write("""
The demand model learns markdown responses well (33,600 observed price-drop events)
but its hike response is ~zero — 99% of weeks have no price change, so it never
learned what a price rise does. Recommending hikes off that blind spot would be
dishonest. The optimizer therefore uses LightGBM for demand *levels* and
*markdown* responses, and the department econometric elasticity for *hikes*.
A 5% materiality hurdle means a change must beat holding still — otherwise the
answer is deliberately to do nothing.
""")
    st.subheader("Honesty notes")
    st.write("""
- Inventory and product cost are **declared synthetic assumptions**, not observed data.
- Markdown events are **inferred** from price drops, not labelled promotions.
- The +20.7% uplift is a **model-based** directional comparison, not a real-world result.
- Hike recommendations rest on econometric estimates from sparse hike data.
""")
    st.subheader("Links")
    st.write("- Kaggle notebook (full pipeline): "
             "https://www.kaggle.com/code/aadityasunilprabhu/dynamic-pricing-and-markdown-optimization-m5")
    st.write("- Dataset: Walmart M5 Forecasting Accuracy "
             "(https://www.kaggle.com/competitions/m5-forecasting-accuracy)")
    st.caption("Solo project · AI for Business Analytics (AIBE) · October 2026")
