import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="OptiStock – AI Inventory Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# GLOBAL STYLE
# --------------------------------------------------
CUSTOM_CSS = """
<style>
/* General */
body, .stApp {
    background-color: #0A1018;
    color: #F5F7FA;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #050913;
}

/* KPI cards */
.kpi-card {
    padding: 1rem 1.2rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, #0F172A, #020617);
    border: 1px solid rgba(148, 163, 184, 0.4);
    box-shadow: 0 14px 28px rgba(15, 23, 42, 0.45);
}

/* Pill badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    border-radius: 999px;
    background: rgba(59, 130, 246, 0.12);
    color: #E5F2FF;
    border: 1px solid rgba(59, 130, 246, 0.6);
}

/* Insights panel */
.insight-card {
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    border-radius: 0.9rem;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.4);
}

/* Table tweaks */
table {
    font-size: 0.9rem;
}
thead tr th {
    background-color: #020617 !important;
}

/* Slider label fix */
[data-baseweb="slider"] {
    padding-top: 0.5rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------
# DEMO DATA
# --------------------------------------------------
def create_demo_inventory(n_skus: int = 60) -> pd.DataFrame:
    np.random.seed(42)
    today = dt.date.today()
    products = [f"SKU-{1000+i}" for i in range(n_skus)]
    product_names = [f"Pharma Product {i+1}" for i in range(n_skus)]
    warehouses = np.random.choice(["WH-NJ1", "WH-PA2", "WH-NY3"], size=n_skus)

    # Random horizons: some expired, some near expiry, some long-dated
    expiry_days = np.random.randint(-120, 365, size=n_skus)
    expiry_dates = [today + dt.timedelta(days=int(d)) for d in expiry_days]

    # Last movement between 0 and 365 days ago
    last_move_days = np.random.randint(0, 365, size=n_skus)
    last_movement = [today - dt.timedelta(days=int(d)) for d in last_move_days]

    on_hand_qty = np.random.randint(50, 5000, size=n_skus)
    unit_cost = np.random.uniform(5, 250, size=n_skus)

    df = pd.DataFrame(
        {
            "SKU": products,
            "Product_Name": product_names,
            "Warehouse": warehouses,
            "Expiry_Date": expiry_dates,
            "Last_Movement_Date": last_movement,
            "On_Hand_Qty": on_hand_qty,
            "Unit_Cost": np.round(unit_cost, 2),
        }
    )
    return df


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def compute_inventory_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Inventory_Value"] = df["On_Hand_Qty"] * df["Unit_Cost"]
    return df


def add_expiry_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    today = dt.date.today()

    df["Days_To_Expiry"] = (pd.to_datetime(df["Expiry_Date"]).dt.date - today).dt.days
    df["Expiry_Status"] = pd.cut(
        df["Days_To_Expiry"],
        bins=[-10_000, -1, 0, 30, 60, 90, 36500],
        labels=[
            "Expired",
            "Expiring Today",
            "0–30 Days",
            "31–60 Days",
            "61–90 Days",
            "> 90 Days",
        ],
    )

    return df


def add_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    today = dt.date.today()
    df["Days_Since_Move"] = (
        today - pd.to_datetime(df["Last_Movement_Date"], errors="coerce").dt.date
    ).dt.days
    return df


def compute_kpi_metrics(df: pd.DataFrame) -> dict:
    total_value = df["Inventory_Value"].sum()

    # At-risk buckets: 0-90 days + expired
    mask_at_risk = df["Days_To_Expiry"] <= 90
    at_risk_value = df.loc[mask_at_risk, "Inventory_Value"].sum()

    mask_expired = df["Days_To_Expiry"] < 0
    expired_value = df.loc[mask_expired, "Inventory_Value"].sum()

    # Slow-moving: > 90 days since last move
    slow_mask = df["Days_Since_Move"] > 90
    slow_value = df.loc[slow_mask, "Inventory_Value"].sum()

    return {
        "total_value": total_value,
        "at_risk_value": at_risk_value,
        "expired_value": expired_value,
        "slow_value": slow_value,
        "at_risk_pct": (at_risk_value / total_value * 100) if total_value else 0,
        "expired_pct": (expired_value / total_value * 100) if total_value else 0,
        "slow_pct": (slow_value / total_value * 100) if total_value else 0,
    }


def generate_ai_like_insights(df: pd.DataFrame, max_insights: int = 8) -> list:
    """Rule-based 'AI' insights – can later be replaced with real model output."""
    insights = []
    df_sorted = df.sort_values("Inventory_Value", ascending=False).reset_index(drop=True)

    # 1. High-value at risk
    high_risk = df_sorted[
        (df_sorted["Days_To_Expiry"] <= 90) & (df_sorted["Days_To_Expiry"] >= 0)
    ].head(3)
    for _, row in high_risk.iterrows():
        insights.append(
            {
                "type": "Expiry Risk",
                "severity": "High",
                "sku": row["SKU"],
                "text": f"{row['SKU']} ({row['Product_Name']}) is projected to expire in {int(row['Days_To_Expiry'])} days with ${row['Inventory_Value']:,.0f} at stake. Consider targeted discounts, sample programs, or reallocation.",
            }
        )

    # 2. Slow-moving with high value
    slow_mask = df["Days_Since_Move"] > 120
    slow_high = df_sorted[slow_mask].head(3)
    for _, row in slow_high.iterrows():
        insights.append(
            {
                "type": "Slow / Dead Stock",
                "severity": "Medium",
                "sku": row["SKU"],
                "text": f"{row['SKU']} has not moved for {int(row['Days_Since_Move'])} days. Inventory value is ${row['Inventory_Value']:,.0f}. Review demand assumptions and reduce production or liquidate.",
            }
        )

    # 3. Warehouse-level concentration
    wh = (
        df.groupby("Warehouse")["Inventory_Value"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    if len(wh) > 0:
        top_wh = wh.iloc[0]
        insights.append(
            {
                "type": "Working Capital",
                "severity": "Info",
                "sku": None,
                "text": f"Warehouse {top_wh['Warehouse']} holds ${top_wh['Inventory_Value']:,.0f} in inventory, the highest across the network. Consider network optimization or redistribution.",
            }
        )

    return insights[:max_insights]


# --------------------------------------------------
# PAGE BUILDERS
# --------------------------------------------------
def page_dashboard(df: pd.DataFrame, metrics: dict, insights: list):
    st.markdown("### 📊 Inventory Health Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="font-size:0.8rem; opacity:0.7;">Total Inventory Value</div>
                <div style="font-size:1.4rem; font-weight:600; margin-top:0.2rem;">
                    ${metrics['total_value']:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="font-size:0.8rem; opacity:0.7;">At-Risk (≤ 90 days)</div>
                <div style="font-size:1.4rem; font-weight:600; margin-top:0.2rem;">
                    ${metrics['at_risk_value']:,.0f}
                </div>
                <div style="font-size:0.75rem; opacity:0.7;">{metrics['at_risk_pct']:.1f}% of total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="font-size:0.8rem; opacity:0.7;">Expired Inventory</div>
                <div style="font-size:1.4rem; font-weight:600; margin-top:0.2rem;">
                    ${metrics['expired_value']:,.0f}
                </div>
                <div style="font-size:0.75rem; opacity:0.7;">{metrics['expired_pct']:.1f}% of total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="font-size:0.8rem; opacity:0.7;">Slow / Dead Stock</div>
                <div style="font-size:1.4rem; font-weight:600; margin-top:0.2rem;">
                    ${metrics['slow_value']:,.0f}
                </div>
                <div style="font-size:0.75rem; opacity:0.7;">{metrics['slow_pct']:.1f}% of total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left, right = st.columns([2, 1.2])

    with left:
        st.subheader("Expiry Risk Distribution")
        expiry_summary = (
            df.groupby("Expiry_Status")["Inventory_Value"]
            .sum()
            .reset_index()
            .sort_values("Inventory_Value", ascending=False)
        )
        expiry_summary["Inventory_Value_M"] = expiry_summary["Inventory_Value"] / 1_000_000

        chart = (
            alt.Chart(expiry_summary)
            .mark_bar()
            .encode(
                x=alt.X("Expiry_Status:N", title="Expiry Bucket"),
                y=alt.Y("Inventory_Value_M:Q", title="Inventory Value (USD, millions)"),
                tooltip=["Expiry_Status", alt.Tooltip("Inventory_Value", format="$.2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Inventory Value by Warehouse")
        wh_summary = (
            df.groupby("Warehouse")["Inventory_Value"].sum().reset_index().sort_values("Inventory_Value", ascending=False)
        )
        wh_summary["Inventory_Value_M"] = wh_summary["Inventory_Value"] / 1_000_000
        if not wh_summary.empty:
            chart_wh = (
                alt.Chart(wh_summary)
                .mark_bar()
                .encode(
                    x=alt.X("Warehouse:N"),
                    y=alt.Y("Inventory_Value_M:Q", title="Inventory Value (USD, millions)"),
                    tooltip=["Warehouse", alt.Tooltip("Inventory_Value", format="$.2f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_wh, use_container_width=True)
        else:
            st.info("No warehouse data available.")

    with right:
        st.subheader("AI Insights")
        st.caption("Rule-based insights – can be replaced with real ML model outputs later.")

        if not insights:
            st.info("No major risks detected in the current dataset.")
        else:
            for ins in insights:
                badge = ins["type"]
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <span class="badge">{badge}</span>
                        <div style="font-size:0.85rem; margin-top:0.4rem;">
                            {ins['text']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def page_expiry_risk(df: pd.DataFrame):
    st.markdown("### ⏱️ Expiry Risk & Obsolescence")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Inventory value by **expiry status**.")
        expiry_summary = (
            df.groupby("Expiry_Status")["Inventory_Value"]
            .sum()
            .reset_index()
            .sort_values("Inventory_Value", ascending=False)
        )
        expiry_summary["Inventory_Value_M"] = expiry_summary["Inventory_Value"] / 1_000_000

        chart = (
            alt.Chart(expiry_summary)
            .mark_bar()
            .encode(
                x=alt.X("Expiry_Status:N", title="Expiry Bucket"),
                y=alt.Y("Inventory_Value_M:Q", title="Inventory Value (USD, millions)"),
                tooltip=["Expiry_Status", alt.Tooltip("Inventory_Value", format="$.2f")],
            )
            .properties(height=340)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.write("Top SKUs at expiry risk (≤ 90 days).")
        at_risk = df[df["Days_To_Expiry"] <= 90].copy()
        at_risk = at_risk.sort_values("Inventory_Value", ascending=False).head(15)
        if at_risk.empty:
            st.success("No SKUs at expiry risk within the next 90 days.")
        else:
            st.dataframe(
                at_risk[
                    [
                        "SKU",
                        "Product_Name",
                        "Warehouse",
                        "Expiry_Date",
                        "Days_To_Expiry",
                        "On_Hand_Qty",
                        "Inventory_Value",
                    ]
                ],
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Expired Inventory")
    expired = df[df["Days_To_Expiry"] < 0].copy()
    if expired.empty:
        st.success("No expired inventory in the dataset – great job!")
    else:
        st.warning(
            f"There are {len(expired)} SKUs with expired inventory totalling ${expired['Inventory_Value'].sum():,.0f}."
        )
        st.dataframe(
            expired[
                [
                    "SKU",
                    "Product_Name",
                    "Warehouse",
                    "Expiry_Date",
                    "Days_To_Expiry",
                    "On_Hand_Qty",
                    "Inventory_Value",
                ]
            ],
            use_container_width=True,
        )


def page_slow_stock(df: pd.DataFrame):
    st.markdown("### 🐌 Slow & Dead Stock")

    threshold_days = st.slider(
        "Define slow-moving threshold (days since last movement):",
        min_value=30,
        max_value=360,
        value=90,
        step=15,
    )

    slow = df[df["Days_Since_Move"] > threshold_days].copy()
    st.caption(
        f"Showing SKUs with **no movement in more than {threshold_days} days**."
    )

    if slow.empty:
        st.success("No SKUs meet the slow-moving criteria for the selected threshold.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            slow[
                [
                    "SKU",
                    "Product_Name",
                    "Warehouse",
                    "Last_Movement_Date",
                    "Days_Since_Move",
                    "On_Hand_Qty",
                    "Inventory_Value",
                ]
            ].sort_values("Days_Since_Move", ascending=False),
            use_container_width=True,
        )

    with col2:
        by_wh = (
            slow.groupby("Warehouse")["Inventory_Value"]
            .sum()
            .reset_index()
            .sort_values("Inventory_Value", ascending=False)
        )
        by_wh["Inventory_Value_M"] = by_wh["Inventory_Value"] / 1_000_000
        st.write("Slow / dead stock by warehouse")
        chart = (
            alt.Chart(by_wh)
            .mark_bar()
            .encode(
                x="Warehouse:N",
                y=alt.Y("Inventory_Value_M:Q", title="Value (USD, millions)"),
                tooltip=["Warehouse", alt.Tooltip("Inventory_Value", format="$.2f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


def page_sku_drilldown(df: pd.DataFrame):
    st.markdown("### 🔍 SKU-Level Drilldown")

    skus = df["SKU"].unique().tolist()
    if not skus:
        st.info("No SKUs found in the dataset.")
        return

    sku_selected = st.selectbox("Select SKU:", sorted(skus))
    sku_df = df[df["SKU"] == sku_selected].copy()
    row = sku_df.iloc[0]

    col_info, col_chart = st.columns([1.1, 1.9])
    with col_info:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="font-size:0.9rem; opacity:0.8;">{row['SKU']}</div>
                <div style="font-size:1.1rem; font-weight:600; margin-top:0.2rem;">
                    {row['Product_Name']}
                </div>
                <div style="font-size:0.8rem; margin-top:0.6rem;">
                    Warehouse: <b>{row['Warehouse']}</b><br/>
                    On Hand: <b>{int(row['On_Hand_Qty'])}</b><br/>
                    Inventory Value: <b>${row['Inventory_Value']:,.0f}</b><br/>
                    Expiry Date: <b>{row['Expiry_Date']}</b> ({int(row['Days_To_Expiry'])} days)<br/>
                    Last Movement: <b>{row['Last_Movement_Date']}</b> ({int(row['Days_Since_Move'])} days ago)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_chart:
        # Simulated history just for visualization (since we don't have real time-series)
        st.caption("Simulated demand & stock history (illustrative for demo purposes).")
        num_points = 12
        months = pd.date_range(
            end=dt.date.today(), periods=num_points, freq="MS"
        ).date
        demand = np.maximum(
            0,
            np.random.normal(loc=row["On_Hand_Qty"] / 8, scale=row["On_Hand_Qty"] / 20, size=num_points),
        ).astype(int)
        stock = np.maximum(
            0,
            row["On_Hand_Qty"]
            - np.cumsum(np.random.normal(loc=demand.mean(), scale=demand.std(), size=num_points)),
        )

        history = pd.DataFrame(
            {
                "Month": months,
                "Demand": demand,
                "Stock_Level": stock,
            }
        )
        hist_melt = history.melt("Month", var_name="Metric", value_name="Value")

        chart = (
            alt.Chart(hist_melt)
            .mark_line(point=True)
            .encode(
                x=alt.X("Month:T", title="Month"),
                y=alt.Y("Value:Q", title="Units"),
                color="Metric:N",
                tooltip=["Month", "Metric", "Value"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)


def page_data_health(df: pd.DataFrame):
    st.markdown("### 🧪 Data Health Check")

    issues = []

    # Missing expiry
    missing_expiry = df["Expiry_Date"].isna().sum()
    if missing_expiry > 0:
        issues.append(f"{missing_expiry} rows have missing expiry dates.")

    # Negative or zero qty
    neg_qty = (df["On_Hand_Qty"] < 0).sum()
    zero_qty = (df["On_Hand_Qty"] == 0).sum()
    if neg_qty:
        issues.append(f"{neg_qty} rows with negative on-hand quantity.")
    if zero_qty:
        issues.append(f"{zero_qty} rows with zero on-hand quantity.")

    # Duplicates by SKU + Warehouse + Expiry
    duplicates = (
        df.duplicated(subset=["SKU", "Warehouse", "Expiry_Date"], keep=False).sum()
    )
    if duplicates:
        issues.append(
            f"{duplicates} potential duplicate rows detected (same SKU, warehouse, expiry)."
        )

    if not issues:
        st.success("No major data quality issues detected in the current dataset.")
    else:
        for issue in issues:
            st.warning(issue)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing or Invalid Dates")
        invalid_dates = df[df["Expiry_Date"].isna() | df["Last_Movement_Date"].isna()]
        if invalid_dates.empty:
            st.info("All expiry and movement dates are present and valid.")
        else:
            st.dataframe(
                invalid_dates[
                    [
                        "SKU",
                        "Product_Name",
                        "Warehouse",
                        "Expiry_Date",
                        "Last_Movement_Date",
                        "On_Hand_Qty",
                    ]
                ],
                use_container_width=True,
            )

    with col2:
        st.subheader("Duplicate Records (by SKU, Warehouse, Expiry)")
        dup_df = df[
            df.duplicated(subset=["SKU", "Warehouse", "Expiry_Date"], keep=False)
        ].copy()
        if dup_df.empty:
            st.info("No potential duplicate inventory records found.")
        else:
            st.dataframe(
                dup_df[
                    [
                        "SKU",
                        "Product_Name",
                        "Warehouse",
                        "Expiry_Date",
                        "On_Hand_Qty",
                        "Inventory_Value",
                    ]
                ],
                use_container_width=True,
            )


def page_insights(df: pd.DataFrame, insights: list):
    st.markdown("### 🤖 AI-Driven Recommendations (Rule-Based Prototype)")
    st.caption(
        "This page uses rule-based logic to simulate how OptiStock will surface AI-driven recommendations. "
        "Later you can plug in a real ML model and reuse the same UI."
    )

    if not insights:
        st.success("No significant risk signals detected in the current dataset.")
        return

    for ins in insights:
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.markdown(
                f"<span class='badge'>{ins['type']}</span>", unsafe_allow_html=True
            )
        with col2:
            st.write(ins["text"])

    st.markdown("---")
    st.subheader("Export Recommendations")
    export_df = pd.DataFrame(
        [
            {
                "Type": ins["type"],
                "Severity": ins["severity"],
                "SKU": ins["sku"],
                "Recommendation": ins["text"],
            }
            for ins in insights
        ]
    )
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Recommendations (CSV)",
        csv,
        file_name="optistock_recommendations.csv",
        mime="text/csv",
    )


def page_simulator(df: pd.DataFrame):
    st.markdown("### 🧮 What-If Simulator (Demand, Lead Time & Safety Stock)")
    st.caption(
        "Use this simple simulator to estimate how demand and lead-time changes could affect "
        "recommended stock levels for a given SKU."
    )

    if df.empty:
        st.info("Load data first to simulate scenarios.")
        return

    sku = st.selectbox("Select SKU for simulation:", sorted(df["SKU"].unique()))
    row = df[df["SKU"] == sku].iloc[0]

    col_inputs, col_outputs = st.columns([1.1, 1.4])

    with col_inputs:
        st.markdown("#### Assumptions")

        base_daily_demand = st.number_input(
            "Baseline average daily demand (units)",
            min_value=1.0,
            value=max(5.0, float(row["On_Hand_Qty"] // 45) or 5.0),
            step=1.0,
        )
        base_lead_time = st.number_input(
            "Baseline supplier lead time (days)",
            min_value=1.0,
            value=30.0,
            step=1.0,
        )
        service_level = st.slider(
            "Target service level (safety factor multiplier)",
            min_value=0.5,
            max_value=2.5,
            value=1.5,
            step=0.1,
        )

        st.markdown("#### Scenario Adjustments")
        demand_change = st.slider(
            "Demand change (%)",
            min_value=-50,
            max_value=100,
            value=10,
            step=5,
        )
        lead_time_change = st.slider(
            "Lead time change (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
        )

    with col_outputs:
        new_daily_demand = base_daily_demand * (1 + demand_change / 100)
        new_lead_time = base_lead_time * (1 + lead_time_change / 100)
        safety_stock = service_level * np.sqrt(new_lead_time) * new_daily_demand * 0.5
        reorder_point = new_daily_demand * new_lead_time + safety_stock

        st.markdown("#### Results")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="font-size:0.9rem; opacity:0.7;">Simulated for {sku} – {row['Product_Name']}</div>
                <div style="font-size:0.85rem; margin-top:0.7rem;">
                    New avg daily demand: <b>{new_daily_demand:.1f} units/day</b><br/>
                    New lead time: <b>{new_lead_time:.1f} days</b><br/>
                    Recommended safety stock: <b>{safety_stock:,.0f} units</b><br/>
                    Recommended reorder point: <b>{reorder_point:,.0f} units</b><br/>
                    Current on-hand: <b>{int(row['On_Hand_Qty'])} units</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if row["On_Hand_Qty"] > reorder_point:
            st.info(
                "Current on-hand inventory is **above** the simulated reorder point – potential excess capacity."
            )
        else:
            st.warning(
                "Current on-hand inventory is **below** the simulated reorder point – risk of stockout under this scenario."
            )


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_inventory_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return create_demo_inventory()

    suffix = uploaded_file.name.lower().split(".")[-1]
    if suffix in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Expect/rename columns if needed (edit these mappings to match your file)
    col_map = {
        "sku": "SKU",
        "product_name": "Product_Name",
        "product": "Product_Name",
        "warehouse": "Warehouse",
        "wh": "Warehouse",
        "expiry_date": "Expiry_Date",
        "expiration_date": "Expiry_Date",
        "exp_date": "Expiry_Date",
        "on_hand_qty": "On_Hand_Qty",
        "qty_on_hand": "On_Hand_Qty",
        "quantity": "On_Hand_Qty",
        "unit_cost": "Unit_Cost",
        "last_movement_date": "Last_Movement_Date",
        "last_sale_date": "Last_Movement_Date",
        "last_move_date": "Last_Movement_Date",
    }

    df_cols_lower = {c.lower(): c for c in df.columns}
    for logical, target in col_map.items():
        if logical in df_cols_lower:
            df.rename(columns={df_cols_lower[logical]: target}, inplace=True)

    required = ["SKU", "Product_Name", "Warehouse", "Expiry_Date", "On_Hand_Qty", "Unit_Cost", "Last_Movement_Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Please adjust your file or the column mappings in app.py."
        )

    df["Expiry_Date"] = parse_date_series(df["Expiry_Date"])
    df["Last_Movement_Date"] = parse_date_series(df["Last_Movement_Date"])

    df["On_Hand_Qty"] = pd.to_numeric(df["On_Hand_Qty"], errors="coerce").fillna(0)
    df["Unit_Cost"] = pd.to_numeric(df["Unit_Cost"], errors="coerce").fillna(0.0)

    return df


# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main():
    with st.sidebar:
        st.markdown("## 📦 OptiStock")
        st.markdown(
            "<span style='font-size:0.8rem; opacity:0.7;'>AI-Powered Inventory Analytics for Pharma</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Upload inventory file", type=["csv", "xlsx", "xls"]
        )
        st.caption(
            "Expected fields: SKU, Product_Name, Warehouse, Expiry_Date, Last_Movement_Date, On_Hand_Qty, Unit_Cost.\n"
            "If no file is uploaded, demo data will be used."
        )

        st.markdown("---")
        page = st.radio(
            "Navigation",
            (
                "Dashboard",
                "Expiry Risk",
                "Slow & Dead Stock",
                "SKU Drilldown",
                "Data Health",
                "AI Insights",
                "What-If Simulator",
            ),
        )

        st.markdown("---")
        st.markdown(
            "<small>Prototype UI – not for production use. Built for OptiStock concept demos.</small>",
            unsafe_allow_html=True,
        )

    try:
        df = load_inventory_data(uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    df = compute_inventory_value(df)
    df = add_expiry_features(df)
    df = add_movement_features(df)
    metrics = compute_kpi_metrics(df)
    insights = generate_ai_like_insights(df)

    st.title("OptiStock – AI-Powered Inventory Analytics")

    if uploaded_file is None:
        st.info(
            "Running in **demo mode** using synthetic pharma inventory data. "
            "Upload a real file in the sidebar to analyze your own inventory."
        )

    if page == "Dashboard":
        page_dashboard(df, metrics, insights)
    elif page == "Expiry Risk":
        page_expiry_risk(df)
    elif page == "Slow & Dead Stock":
        page_slow_stock(df)
    elif page == "SKU Drilldown":
        page_sku_drill
