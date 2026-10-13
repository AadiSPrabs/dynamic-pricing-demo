"""Dynamic Pricing and Markdown Optimization System — Streamlit demo.
AIBE subject project (solo) by Aadi. Walmart M5 data.

Written for non-technical users: plain language, guided steps, glossary.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

import pricing as P

st.set_page_config(
    page_title="Smart Pricing Demo",
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

# Plain-English one-liners for each recommendation, shown to end users.
ACTION_PLAIN = {
    "protect_margin": "Keep the price as it is",
    "test_increase": "Raise the price a little",
    "apply_markdown": "Cut the price",
    "wait_reassess": "Hold on, decide later",
}

PAGES = ["Home", "Recommendations", "Price Simulator", "How It Works", "About"]


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
    label = ACTION_PLAIN[action]
    color = ACTION_COLORS[action]
    return (f'<span style="background:{color};color:white;padding:6px 16px;'
            f'border-radius:20px;font-weight:600;font-size:1.05em;">{label}</span>')


def glossary():
    with st.expander("📖 Words you'll see around here"):
        st.markdown("""
- **Markdown** — a price cut meant to sell more units quickly (for example, to clear stock).
- **Margin** — the slice of the price you keep after covering the product's cost. Bigger margin = more profit per sale.
- **Price elasticity** — how strongly shoppers react to a price change. If demand is *inelastic*, a small price rise barely dents sales, so raising the price can pay off.
- **Expected demand / profit** — the model's best guess from past sales patterns. An estimate, not a promise.
- **Hold** — do nothing; keep the current price this week.
""")


# ------------------------------------------------------------------ sidebar ---
st.sidebar.title("🏷️ Pricing Optimizer")
st.sidebar.caption("AIBE project · Walmart M5")

if "page" not in st.session_state:
    st.session_state.page = "Home"
page = st.sidebar.radio("Go to", PAGES,
                        index=PAGES.index(st.session_state.page))
st.session_state.page = page

st.sidebar.divider()
st.sidebar.caption("💡 New here? Start at **Home** — it explains everything in plain words.")

cards = get_cards()
metrics = get_metrics()
mix_all = cards["action"].value_counts()

# --------------------------------------------------------------------- home ---
if page == "Home":
    st.title("🏷️ Smart Pricing for Retail")
    st.write("""
Imagine you run a shop with thousands of products. Every week you face the same
question for each one: **should I raise the price, cut it, or leave it alone?**

This app answers that question for **2,000 real products**, using their actual
weekly sales history from Walmart. For each product it suggests one clear action,
shows you the reasoning, and lets you test your own prices to see what would happen.
""")

    st.subheader("How to use this app")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("### 1. 📋 Browse suggestions")
        st.write("See what the app recommends for every product — raise, cut, or hold — "
                 "and why. Filter by product type or search for an item.")
    with s2:
        st.markdown("### 2. 🔬 Try your own price")
        st.write("Pick any product, drag the price slider, and watch the predicted "
                 "sales and profit move. Compare your price against the recommendation.")
    with s3:
        st.markdown("### 3. 📊 Check the fine print")
        st.write("See how accurate the predictions are, what the model pays attention "
                 "to, and where its limits are — stated honestly.")

    st.subheader("The big picture")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products analyzed", f"{len(cards):,}")
    c2.metric("Suggestions to raise price", f"{int(mix_all.get('test_increase', 0)):,}")
    c3.metric("Suggestions to cut price", f"{int(mix_all.get('apply_markdown', 0)):,}")
    c4.metric("Model-estimated uplift", f"+{metrics['policy']['uplift']:.0%}",
              help="How much more profit the suggestions would make vs keeping all "
                   "prices unchanged, according to the model. Directional, not a real-world result.")

    st.info("**In one line:** most products should hold their price, about two-thirds get a "
            "small increase suggestion, and about one-third get a markdown suggestion.")

    glossary()

    st.subheader("Where the numbers come from")
    st.write("""
- **Sales data:** Walmart M5 — daily sales and prices for 3,049 products across 10 stores.
- **The brain:** a demand-forecasting model trained on all 30,490 product-store histories,
  plus a classic economics formula for how shoppers react to price rises.
- **Honest limits:** product costs are assumed (Walmart never published them), markdowns are
  spotted from price drops rather than labelled promotions, and the profit uplift is what the
  *model* expects — not a measured real-world result.
""")

# ---------------------------------------------------------- recommendations ---
elif page == "Recommendations":
    st.title("📋 What should I do with each price?")
    st.write("Every product below gets **one clear suggestion**. Use the filters to narrow "
             "things down, then click a product to see the full reasoning.")

    f1, f2, f3 = st.columns(3)
    with f1:
        actions = st.multiselect("Suggestion type",
                                 options=list(ACTION_PLAIN),
                                 format_func=lambda a: ACTION_PLAIN[a],
                                 default=list(ACTION_PLAIN))
    with f2:
        cats = st.multiselect("Product category", options=sorted(cards["category"].unique()),
                              default=sorted(cards["category"].unique()))
    with f3:
        q = st.text_input("Search product code", "",
                          help="Type part of a product code, e.g. FOODS_1")

    view = cards[cards["action"].isin(actions) & cards["category"].isin(cats)]
    if q:
        view = view[view["item_id"].str.contains(q.upper(), na=False)]
    view = view.reset_index(drop=True)

    st.subheader("Suggestion breakdown")
    mix = view["action"].value_counts()
    mcols = st.columns(4)
    for i, a in enumerate(ACTION_PLAIN):
        mcols[i].metric(ACTION_PLAIN[a], f"{int(mix.get(a, 0)):,}")

    st.dataframe(
        view[["item_id", "store_id", "category", "sell_price", "action",
              "price", "demand", "profit"]].rename(columns={
                  "item_id": "Product", "store_id": "Store", "category": "Category",
                  "sell_price": "Current price ($)", "action": "Suggestion",
                  "price": "Suggested price ($)", "demand": "Expected sales/week",
                  "profit": "Expected profit/week ($)"}),
        width="stretch", height=320,
    )
    st.caption("Expected sales and profit are the model's estimates from past patterns — "
               "a guide, not a guarantee.")

    if len(view):
        st.subheader("The full story for one product")
        pick = st.selectbox("Pick a product", view["label"].tolist())
        row = view[view["label"] == pick].iloc[0]
        with st.spinner("Crunching the numbers..."):
            res = P.optimise(row)
        st.divider()

        st.markdown(f"### {row['item_id']} @ {row['store_id']} &nbsp; {action_badge(res['action'])}",
                    unsafe_allow_html=True)
        st.caption(f"Week {int(row['wm_yr_wk'])} · {row['dept_id']} · {row['cat_id']}")

        disc = (1 - res["mult"]) if res["mult"] < 1 else 0
        if res["action"] == "test_increase":
            delta_txt = f"+{res['mult'] - 1:.0%}"
        elif disc:
            delta_txt = f"-{disc:.0%}"
        else:
            delta_txt = None
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current price", f"${row['sell_price']:.2f}")
        c2.metric("Suggested price", f"${res['price']:.2f}", delta=delta_txt)
        c3.metric("Expected sales", f"{res['demand']:.0f} units/week")
        c4.metric("Expected profit", f"${res['profit']:.2f}/week")

        st.markdown(f"**Why this suggestion?** {res['reason']}.")
        st.markdown(P.ACTION_DESCRIPTIONS[res["action"]])

        with st.expander("⚠️ What could go wrong"):
            st.write("Sales may not react the way the model expects. Markdowns here are "
                     "spotted from price drops (not labelled promotions), and product costs "
                     "are assumed, not real Walmart data. Treat the profit figure as a "
                     "directional guide, not a promise.")
        st.info("**When to look again:** re-check in 2 weeks, or sooner if sales move oddly.")

        if st.button("🔬 Test my own price for this product"):
            st.session_state.sim_pick = pick
            st.session_state.page = "Price Simulator"
            st.rerun()
    else:
        st.warning("No products match those filters — try widening them.")

    glossary()

# --------------------------------------------------------------- simulator ---
elif page == "Price Simulator":
    st.title("🔬 What if I change the price?")
    st.write("Pick a product, set the price **you** have in mind, and see the predicted "
             "sales and profit — then compare against the app's suggestion.")

    st.subheader("Step 1 — Pick a product")
    labels = cards["label"].tolist()
    default_ix = labels.index(st.session_state.sim_pick) if st.session_state.get("sim_pick") in labels else 0
    pick = st.selectbox("Product", labels, index=default_ix, key="sim_pick_box")
    row = cards[cards["label"] == pick].iloc[0]
    base = float(row["sell_price"])
    cost = base * P.UNIT_COST_RATIO

    with st.spinner("Crunching the numbers..."):
        res = P.optimise(row)
    hold_dem = P.demand_at_price(row, base)
    hold_profit = (base - cost) * hold_dem

    st.markdown(f"**{row['item_id']}** @ {row['store_id']} · {row['dept_id']} &nbsp; "
                f"{action_badge(res['action'])}", unsafe_allow_html=True)
    st.caption(f"The app suggests **\\${res['price']:.2f}** ({ACTION_PLAIN[res['action']].lower()}). "
               f"Current price \\${base:.2f} · assumed unit cost \\${cost:.2f}.")

    st.subheader("Step 2 — Set your price")
    price = st.slider("Your price ($)", float(round(0.70 * base, 2)),
                      float(round(1.10 * base, 2)), float(round(base, 2)), 0.01,
                      help="Drag between 30% off and 10% above the current price.")
    dem = P.demand_at_price(row, price)
    profit = (price - cost) * dem

    st.subheader("Step 3 — See what happens")
    k1, k2, k3 = st.columns(3)
    k1.metric("Your price", f"${price:.2f}")
    k2.metric("Predicted sales", f"{dem:.0f} units/week")
    k3.metric("Predicted profit", f"${profit:.2f}/week")

    d_rec = profit - res["profit"]
    d_hold = profit - hold_profit
    if abs(d_rec) < 0.5:
        st.success(f"🤝 About the same as the app's suggestion (\\${res['price']:.2f}) — "
                   f"either price works.")
    elif d_rec > 0:
        st.success(f"👍 At **\\${price:.2f}** you'd make about **\\${profit:.2f}/week** — "
                   f"roughly **\\${d_rec:.2f} more** than the app's suggestion.")
    else:
        st.warning(f"👎 At **\\${price:.2f}** you'd make about **\\${profit:.2f}/week** — "
                   f"roughly **\\${-d_rec:.2f} less** than the app's suggestion of "
                   f"\\${res['price']:.2f}.")
    st.caption(f"For reference, simply holding at \\${base:.2f} would make about "
               f"\\${hold_profit:.2f}/week.")

    grid = np.linspace(0.70 * base, 1.10 * base, 41)
    ds = np.array([P.demand_at_price(row, p) for p in grid])
    ps = (grid - cost) * ds
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=ds, name="Sales (units/week)", yaxis="y1",
                             line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(x=grid, y=ps, name="Profit ($/week)", yaxis="y2",
                             line=dict(color="#16a34a")))
    for x, name, col, dash in [(base, "current price", "#6b7280", "dot"),
                               (res["price"], "suggestion", "#ea580c", "dash"),
                               (price, "your price", "#7c3aed", "solid")]:
        fig.add_vline(x=x, line_dash=dash, line_color=col,
                      annotation_text=name, annotation_position="top")
    fig.update_layout(
        xaxis_title="Price ($)",
        yaxis=dict(title="Sales (units/week)", side="left"),
        yaxis2=dict(title="Profit ($/week)", side="right", overlaying="y"),
        legend=dict(orientation="h", y=1.08), height=420, margin=dict(t=40),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("**How to read this:** the blue line is predicted weekly sales (left axis), "
               "the green line is predicted weekly profit (right axis). Profit usually "
               "peaks somewhere in the middle — too cheap and you earn little per sale, "
               "too pricey and shoppers walk away.")

    with st.expander("⚙️ What's under the hood here"):
        st.write("Sales predictions come from a demand model trained on 30,490 product-store "
                 "sales histories. It learned markdown reactions from ~33,600 real price-drop "
                 "events; for price *rises* it uses a classic economics formula, because the "
                 "model rarely saw rises in training. Unit cost is assumed at 62% of price "
                 "(Walmart never published real costs).")

    glossary()

# -------------------------------------------------------------- how it works ---
elif page == "How It Works":
    st.title("📊 How good are the predictions?")
    st.write("We tested the demand model on weeks it had **never seen before** — that's the "
             "honest test. Here's how it did, in plain words.")

    st.subheader("Prediction accuracy")
    st.write("The table compares three approaches. Lower error = better.")
    perf = pd.DataFrame([
        {"Method": "Just repeat last week", "Average miss (units/week)": metrics["naive (lag_1)"]["mae"]},
        {"Method": "Simple trend line", "Average miss (units/week)": metrics["linear regression"]["mae"]},
        {"Method": "Our model (on new data)", "Average miss (units/week)": metrics["LightGBM (TEST - unseen weeks)"]["mae"]},
    ])
    st.dataframe(perf.style.format({"Average miss (units/week)": "{:.1f}"}),
                 width="stretch", hide_index=True)
    st.caption("**In plain words:** on weeks it never saw, our model's sales guess misses by "
               f"about **{metrics['LightGBM (TEST - unseen weeks)']['mae']:.1f} units a week** "
               "on average — better than just repeating last week.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Predicted vs actual sales")
        st.image(P.load_plot("pred_vs_actual.png"))
        st.caption("Each dot is a product-week. Dots hugging the diagonal = good predictions.")
    with c2:
        st.subheader("What the model pays attention to")
        st.image(P.load_plot("feat_importance.png"))
        st.caption("Recent sales history matters most; price itself matters less — which is "
                   "why price *rises* use the economics formula instead.")

    st.subheader("How shoppers react to price rises")
    st.write("We measured this per department with a classic economics method. All departments "
             "are *inelastic* — shoppers don't flee over small rises, which is why most "
             "suggestions are small increases.")
    elas = pd.read_csv(os.path.join(DATA, "elasticity_by_dept.csv"))
    st.dataframe(elas.rename(columns={"dept_id": "Department", "elasticity": "Reaction strength",
                                      "r2": "Fit quality", "n": "Weeks studied"})
                 .style.format({"Reaction strength": "{:.2f}", "Fit quality": "{:.2f}"}),
                 width="stretch", hide_index=True)
    st.caption("Reaction strength −0.50 means a 10% price rise costs about 5% of sales.")

    st.subheader("Did the suggestions actually help? (simulated)")
    pol = metrics["policy"]
    p1, p2, p3 = st.columns(3)
    p1.metric("With suggestions", f"${pol['optimizer_profit_per_week']:,.0f}/week")
    p2.metric("Keeping all prices", f"${pol['hold_profit_per_week']:,.0f}/week")
    p3.metric("Improvement", f"+{pol['uplift']:.0%}")
    st.caption("This compares the two strategies *inside the model* — a directional check, "
               "not a real-world experiment. Real proof would need a live store trial.")

    with st.expander("📦 Model performance and limitations (the fine print)"):
        st.write(f"""
- **Validation error:** MAE {metrics['LightGBM (val)']['mae']:.4f}, RMSE {metrics['LightGBM (val)']['rmse']:.4f}
- **Unseen-weeks error:** MAE {metrics['LightGBM (TEST - unseen weeks)']['mae']:.4f}, RMSE {metrics['LightGBM (TEST - unseen weeks)']['rmse']:.4f}
- These measure **demand-estimation error**, not pricing-policy accuracy.
- **Product costs are assumed** (62% of price) — Walmart never published them.
- **Markdowns are inferred** from price drops, not labelled promotions.
- The **+20.7% uplift is model-based** — directional, not a causal real-world result.
- Price-rise suggestions rest on econometric estimates from sparse rise data.
""")

    glossary()

# -------------------------------------------------------------------- about ---
else:
    st.title("ℹ️ About this app")
    st.write("""
**Smart Pricing Demo** — a pricing decision-support prototype for small and medium
retailers, built on real Walmart sales data.

Instead of another sales dashboard, it answers one question per product per week:
**what should I do about the price?** Each answer comes as an explainable card:
the suggestion, the reasoning, what could go wrong, and when to look again.
""")
    st.subheader("How it's built")
    st.write("Sales data → demand forecasting model → shopper-reaction economics → "
             "suggestion engine → these pages. The full pipeline (code and math) is on Kaggle:")
    st.write("https://www.kaggle.com/code/aadityasunilprabhu/dynamic-pricing-and-markdown-optimization-m5")
    st.subheader("The honest bits")
    st.write("""
- Product costs are **assumed**, not real Walmart data.
- Markdowns are **spotted from price drops**, not labelled promotions.
- The +20.7% uplift is what the **model expects** — not a measured store result.
""")
    st.caption("Solo project · AI for Business Analytics (AIBE) · October 2026 · "
               "Data: Walmart M5 Forecasting Accuracy")
