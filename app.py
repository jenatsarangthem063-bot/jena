


import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import logic

# ---------------------------------------------------------------------------
# Page config & light styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy | Factory Reallocation & Shipping Optimization",
    page_icon="🏭",
    layout="wide",
)

PRIMARY = "#7A2E8E"      # plum
ACCENT = "#F2A20C"       # amber (candy accent)
RISK = "#D64545"
GOOD = "#2E9E6B"
MUTED = "#6B6B76"

st.markdown(f"""
<style>
    .metric-card {{
        background: #ffffff10;
        border: 1px solid #ffffff22;
        border-radius: 10px;
        padding: 14px 18px;
    }}
    .risk-badge {{
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:0.78rem; font-weight:600;
    }}
    .risk-high {{ background:{RISK}22; color:{RISK}; }}
    .risk-normal {{ background:{GOOD}22; color:{GOOD}; }}
    h1, h2, h3 {{ letter-spacing: -0.01em; }}
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "Cleaned_Nassau_Data.csv")


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and engineering data...")
def get_data():
    return logic.load_and_engineer(DATA_PATH)


@st.cache_resource(show_spinner="Training predictive models...")
def get_models(df):
    return logic.train_models(df)


@st.cache_data(show_spinner="Clustering routes...")
def get_clusters(df):
    return logic.cluster_routes(df)


@st.cache_data(show_spinner="Generating recommendations...")
def get_batch_recs(df, _model, _encoders, priority_weight):
    return logic.batch_recommendations(df, _model, _encoders, priority_weight)


df = get_data()
model_bundle = get_models(df)
best_model = model_bundle["best_model"]
encoders = model_bundle["encoders"]
clusters_df = get_clusters(df)

# ---------------------------------------------------------------------------
# Sidebar — global controls
# ---------------------------------------------------------------------------
st.sidebar.title("🍬 Nassau Candy")
st.sidebar.caption("Factory Reallocation & Shipping Optimization")

products = sorted(df["Product Name"].unique())
regions = sorted(df["Region"].unique())
ship_modes = sorted(df["Ship Mode"].unique())

st.sidebar.markdown("### Global filters")
sel_product = st.sidebar.selectbox("Product", products, index=products.index("Wonka Bar - Milk Chocolate") if "Wonka Bar - Milk Chocolate" in products else 0)
sel_region = st.sidebar.selectbox("Region (optional)", ["All regions"] + regions)
sel_ship_mode = st.sidebar.selectbox("Ship Mode (optional)", ["All ship modes"] + ship_modes)

st.sidebar.markdown("### Optimization priority")
priority_weight = st.sidebar.slider(
    "Speed  ⟵────────⟶  Profit", 0.0, 1.0, 0.5, 0.05,
    help="0 = optimize purely for profit impact · 1 = optimize purely for lead-time reduction",
)
st.sidebar.caption(f"Weighting: **{priority_weight*100:.0f}% speed** / **{(1-priority_weight)*100:.0f}% profit**")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Assumption: incremental logistics cost of "
    f"${logic.SHIPPING_RATE_PER_UNIT_MILE:.4f} per unit-mile is used to translate "
    "factory–customer distance into a shipping-cost delta. Manufacturing cost per "
    "unit is held constant across factories for the same product."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Factory Reallocation & Shipping Optimization")
st.caption(
    "Decision-intelligence system for Nassau Candy Distributor — predicts shipping "
    "outcomes under different factory configurations and recommends product "
    "reassignments that balance shipping efficiency and profitability."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders analyzed", f"{df.shape[0]:,}")
c2.metric("Products", f"{df['Product Name'].nunique()}")
c3.metric("Factories", f"{df['Factory'].nunique()}")
c4.metric("Best lead-time model", model_bundle["best_model_name"])

tabs = st.tabs([
    "📊 Overview",
    "🤖 Predictive Model",
    "🧭 Route & Product Clustering",
    "🏭 Factory Optimization Simulator",
    "🔀 What-If Scenario Analysis",
    "✅ Recommendation Dashboard",
    "⚠️ Risk & Impact Panel",
])

# =============================================================================
# TAB 1 — Overview / EDA
# =============================================================================
with tabs[0]:
    st.subheader("Descriptive overview")

    colA, colB = st.columns(2)
    with colA:
        sales_by_region = df.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
        fig = px.bar(sales_by_region, x="Region", y="Sales", color="Region",
                     title="Total sales by region", color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with colB:
        profit_by_factory = df.groupby("Factory", as_index=False)["Gross Profit"].sum().sort_values("Gross Profit", ascending=False)
        fig = px.bar(profit_by_factory, x="Factory", y="Gross Profit", color="Factory",
                     title="Total gross profit by factory", color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    colC, colD = st.columns(2)
    with colC:
        fig = px.histogram(df, x="Lead Time", nbins=40, title="Lead time distribution (days)",
                            color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, width='stretch')
    with colD:
        margin_by_div = df.groupby("Division", as_index=False)["Profit Margin"].mean()
        fig = px.bar(margin_by_div, x="Division", y="Profit Margin", color="Division",
                     title="Average profit margin (%) by division", color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    lt_std = df["Lead Time"].std()
    lt_range = df.groupby("Factory")["Lead Time"].mean()
    st.info(
        f"**EDA insight:** average lead time varies only mildly across factories "
        f"({lt_range.min():.0f}–{lt_range.max():.0f} days) relative to overall "
        f"variability (σ ≈ {lt_std:.0f} days). Ship Mode, Region, and Factory alone "
        "explain relatively little of the spread in this dataset — see the model "
        "evaluation tab for how this affects predictive accuracy."
    )

    with st.expander("Show cleaned data sample"):
        st.dataframe(df.head(50), width='stretch')

# =============================================================================
# TAB 2 — Predictive Model
# =============================================================================
with tabs[1]:
    st.subheader("Lead-time prediction: model comparison")
    st.caption("Predict expected shipping lead time from Product/Division, origin factory, "
               "destination region, ship mode, and distance.")

    metrics_df = model_bundle["metrics"]
    st.dataframe(
        metrics_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.4f}"})
        .highlight_max(subset=["R2"], color=f"{GOOD}33")
        .highlight_min(subset=["RMSE", "MAE"], color=f"{GOOD}33"),
        width='stretch',
    )
    st.success(f"Selected model: **{model_bundle['best_model_name']}** (highest R² / lowest error on held-out data).")

    if metrics_df["R2"].max() < 0.1:
        st.warning(
            "R² is close to zero (or negative) for all candidate models. This means "
            "Division, Factory, Region, Ship Mode, and Distance explain very little of "
            "the variation in Lead Time in this dataset — lead times behave close to "
            "random noise around the overall mean. Recommendations later in this app "
            "still use the best available model, but should be weighted primarily on "
            "**profit impact**, which is grounded directly in Sales/Cost, rather than on "
            "the lead-time prediction alone."
        )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(model_bundle["test_results"], x="Actual", y="Predicted",
                          title="Actual vs. predicted lead time (test set)",
                          color_discrete_sequence=[PRIMARY], opacity=0.5)
        lo = model_bundle["test_results"][["Actual", "Predicted"]].min().min()
        hi = model_bundle["test_results"][["Actual", "Predicted"]].max().max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=MUTED, dash="dash"))
        st.plotly_chart(fig, width='stretch')
    with col2:
        if model_bundle["importances"] is not None:
            fig = px.bar(model_bundle["importances"], x="Importance", y="Feature", orientation="h",
                         title=f"Feature importance ({model_bundle['best_model_name']})",
                         color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')
        else:
            coefs = pd.DataFrame({
                "Feature": ["Division", "Factory", "Region", "Ship Mode", "Distance", "Units"],
                "Coefficient": best_model.coef_,
            }).sort_values("Coefficient", key=abs, ascending=False)
            fig = px.bar(coefs, x="Coefficient", y="Feature", orientation="h",
                         title="Linear regression coefficients", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, width='stretch')

# =============================================================================
# TAB 3 — Route & Product Clustering
# =============================================================================
with tabs[2]:
    st.subheader("Route & product clustering")
    st.caption("Region × Product combinations clustered by average lead time and average profit margin.")

    fig = px.scatter(
        clusters_df, x="Avg_Lead_Time", y="Avg_Profit_Margin",
        color="Cluster Label", size="Total_Units", hover_data=["Region", "Product Name", "Orders"],
        title="Route clusters: lead time vs. profit margin",
        color_discrete_map={
            "Efficient": GOOD, "Slow but Profitable": ACCENT,
            "Fast but Thin-Margin": "#3B82C4", "Congested / High-Risk": RISK,
        },
    )
    fig.update_layout(xaxis_title="Avg. lead time (days)", yaxis_title="Avg. profit margin (%)")
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Consistently slow / congested combinations")
    congested = clusters_df[clusters_df["Cluster Label"] == "Congested / High-Risk"].sort_values(
        "Avg_Lead_Time", ascending=False
    )
    st.dataframe(
        congested[["Region", "Product Name", "Avg_Lead_Time", "Avg_Profit_Margin", "Total_Units", "Orders"]]
        .rename(columns={"Avg_Lead_Time": "Avg Lead Time (days)", "Avg_Profit_Margin": "Avg Profit Margin (%)"}),
        width='stretch',
    )

# =============================================================================
# TAB 4 — Factory Optimization Simulator
# =============================================================================
with tabs[3]:
    st.subheader(f"Factory optimization simulator — {sel_product}")
    st.caption("Predicted performance for this product across every factory, using the global filters in the sidebar.")

    subset = df[df["Product Name"] == sel_product]
    if sel_region != "All regions":
        subset = subset[subset["Region"] == sel_region]
    if sel_ship_mode != "All ship modes":
        subset = subset[subset["Ship Mode"] == sel_ship_mode]

    if subset.empty:
        st.warning("No orders match this product + filter combination. Try a broader filter.")
    else:
        sim, meta = logic.simulate_reassignment(subset, best_model, encoders, priority_weight)

        st.markdown(
            f"Current factory: **{meta['current_factory']}** · Division: **{meta['division']}** · "
            f"Orders in scope: **{meta['order_count']:,}** · Total units: **{meta['total_units']:,.0f}**"
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(sim, x="Factory", y="Predicted Lead Time (days)", color="Is Current",
                         title="Predicted lead time by factory",
                         color_discrete_map={True: ACCENT, False: PRIMARY})
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig = px.bar(sim, x="Factory", y="Profit Impact ($)", color="Is Current",
                         title="Simulated profit impact vs. current assignment",
                         color_discrete_map={True: ACCENT, False: PRIMARY})
            st.plotly_chart(fig, width='stretch')

        st.dataframe(
            sim[["Factory", "Is Current", "Distance (mi)", "Predicted Lead Time (days)",
                 "Lead Time Reduction (%)", "Profit Impact ($)", "Profit Impact (%)",
                 "Score", "Confidence Score", "Risk Flag"]],
            width='stretch',
        )

        top = sim.iloc[0]
        if top["Is Current"]:
            st.success("The current factory is already the top-ranked option under this priority weighting.")
        else:
            st.success(
                f"Recommended reassignment: **{meta['current_factory']} → {top['Factory']}** "
                f"({top['Lead Time Reduction (%)']:+.1f}% lead time, {top['Profit Impact (%)']:+.1f}% profit)."
            )

# =============================================================================
# TAB 5 — What-If Scenario Analysis
# =============================================================================
with tabs[4]:
    st.subheader("What-if: current vs. recommended assignment")

    subset = df[df["Product Name"] == sel_product]
    if sel_region != "All regions":
        subset = subset[subset["Region"] == sel_region]
    if sel_ship_mode != "All ship modes":
        subset = subset[subset["Ship Mode"] == sel_ship_mode]

    if subset.empty:
        st.warning("No orders match this product + filter combination. Try a broader filter.")
    else:
        sim, meta = logic.simulate_reassignment(subset, best_model, encoders, priority_weight)
        current = sim[sim["Is Current"]].iloc[0]
        recommended = sim.iloc[0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### Current: {current['Factory']}")
            st.metric("Predicted lead time", f"{current['Predicted Lead Time (days)']:.1f} days")
            st.metric("Simulated profit", f"${current['Simulated Profit ($)']:,.2f}")
        with c2:
            st.markdown(f"### Recommended: {recommended['Factory']}")
            st.metric("Predicted lead time", f"{recommended['Predicted Lead Time (days)']:.1f} days",
                       delta=f"{recommended['Predicted Lead Time (days)'] - current['Predicted Lead Time (days)']:.1f} days")
            st.metric("Simulated profit", f"${recommended['Simulated Profit ($)']:,.2f}",
                       delta=f"${recommended['Profit Impact ($)']:,.2f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Lead Time (days)", x=["Current", "Recommended"],
                              y=[current["Predicted Lead Time (days)"], recommended["Predicted Lead Time (days)"]],
                              marker_color=[MUTED, ACCENT]))
        fig.update_layout(title="Lead-time comparison")
        st.plotly_chart(fig, width='stretch')

        if recommended["Factory"] == current["Factory"]:
            st.info("No reassignment is recommended for this product/filter combination at the current priority weighting.")
        else:
            st.markdown(
                f"Switching **{sel_product}** from **{current['Factory']}** to **{recommended['Factory']}** is "
                f"projected to change lead time by **{recommended['Lead Time Reduction (%)']:+.1f}%** and profit by "
                f"**{recommended['Profit Impact (%)']:+.1f}%** (**${recommended['Profit Impact ($)']:,.2f}**), "
                f"based on {meta['order_count']:,} historical orders in scope."
            )

# =============================================================================
# TAB 6 — Recommendation Dashboard
# =============================================================================
with tabs[5]:
    st.subheader("Ranked factory reassignment recommendations")
    st.caption("Every product evaluated across its full customer base at the current priority weighting.")

    batch = get_batch_recs(df, best_model, encoders, priority_weight)

    only_changes = st.checkbox("Show only products with a recommended reassignment", value=False)
    view = batch[batch["Changed?"] == "Reassign"] if only_changes else batch

    def style_risk(v):
        return f"color: {RISK}; font-weight:600;" if v == "High Risk" else f"color: {GOOD};"

    st.dataframe(
        view.style.map(style_risk, subset=["Risk Flag"]),
        width='stretch',
    )

    st.download_button(
        "Download recommendations as CSV",
        data=batch.to_csv(index=False).encode("utf-8"),
        file_name="nassau_factory_recommendations.csv",
        mime="text/csv",
    )

    n_changed = (batch["Changed?"] == "Reassign").sum()
    st.markdown(f"**{n_changed} of {batch.shape[0]} products** have a recommended reassignment under the current priority weighting.")

# =============================================================================
# TAB 7 — Risk & Impact Panel
# =============================================================================
with tabs[6]:
    st.subheader("Risk & impact panel")

    batch = get_batch_recs(df, best_model, encoders, priority_weight)
    high_risk = batch[batch["Risk Flag"] == "High Risk"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Products flagged high-risk", int(high_risk.shape[0]))
    c2.metric("Total profit impact of all recs", f"${batch['Profit Impact ($)'].sum():,.2f}")
    c3.metric("Products recommended to move", int((batch["Changed?"] == "Reassign").sum()))

    if high_risk.empty:
        st.success("No reassignment recommendations currently cross the high-risk profit-loss threshold.")
    else:
        st.error(f"{high_risk.shape[0]} recommendation(s) carry a projected profit loss beyond the risk threshold:")
        st.dataframe(
            high_risk[["Product Name", "Current Factory", "Recommended Factory",
                       "Profit Impact ($)", "Profit Impact (%)", "Confidence Score"]],
            width='stretch',
        )

    fig = px.bar(
        batch.sort_values("Profit Impact ($)"), x="Profit Impact ($)", y="Product Name", orientation="h",
        color="Risk Flag", title="Profit impact by product (all recommendations)",
        color_discrete_map={"High Risk": RISK, "Normal": GOOD},
    )
    st.plotly_chart(fig, width='stretch')

    st.caption(
        "Risk threshold: a recommendation is flagged **High Risk** when its projected profit impact is a loss "
        "exceeding 5% of that product's current total gross profit."
    )
