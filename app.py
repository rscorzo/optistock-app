import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# -------------------------
# Helper functions
# -------------------------

def validate_columns(df: pd.DataFrame):
    """Ensure required columns are present."""
    required = [
        "SKU",
        "Date",
        "Monthly_Demand",
        "Lead_Time_Days",
        "Shelf_Life_Days",
        "Unit_Cost",
        "On_Hand",
        "Expiration_Date",
        "Days_To_Expiry",
        "Expiring_Soon_Flag",
        "Inventory_Value",
    ]
    missing = [c for c in required if c not in df.columns]
    return missing


def compute_light_forecast(df: pd.DataFrame):
    """
    Light forecasting model:
    - For each SKU, use the last 6 months avg demand * 12 for 12M forecast.
    - Safety stock based on std dev and lead time.
    - Recommended production = max(0, forecast_12m + safety_stock - on_hand).
    - Expiry metrics aggregated at SKU level.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    sku_list = df["SKU"].unique()
    results = []

    for sku in sku_list:
        sku_hist = df[df["SKU"] == sku].sort_values("Date")

        # Demand series
        demand = sku_hist["Monthly_Demand"].astype(float)

        # Last 6 months average (fallback to overall mean if < 6 points)
        if len(demand) >= 6:
            avg_6 = demand.tail(6).mean()
        else:
            avg_6 = demand.mean()

        forecast_12m_units = float(avg_6 * 12)

        # Safety stock
        lead_time_days = float(sku_hist["Lead_Time_Days"].iloc[0])
        lead_time_months = max(lead_time_days / 30.0, 0.5)
        sigma = float(demand.std()) if demand.std() > 0 else 1.0
        Z = 1.65  # ~95% service level
        safety_stock_units = int(Z * sigma * np.sqrt(lead_time_months))

        # On-hand snapshot: last non-null On_Hand
        sku_onhand = sku_hist[~sku_hist["On_Hand"].isna()]
        if not sku_onhand.empty:
            on_hand_units = float(sku_onhand["On_Hand"].iloc[-1])
        else:
            on_hand_units = 0.0

        # Cost
        unit_cost = float(sku_hist["Unit_Cost"].mean())

        # Expiry risk metrics (aggregate over rows where On_Hand is not null)
        exp_rows = sku_hist[~sku_hist["On_Hand"].isna()].copy()
        if exp_rows.empty:
            total_on_hand = 0.0
            total_inv_value = 0.0
            exp_0_30_units = 0.0
            exp_31_90_units = 0.0
            exp_gt_90_units = 0.0
            exp_soon_units = 0.0
            exp_soon_value = 0.0
            min_days_to_expiry = np.nan
        else:
            total_on_hand = float(exp_rows["On_Hand"].sum())
            total_inv_value = float(exp_rows["Inventory_Value"].sum())
            min_days_to_expiry = float(exp_rows["Days_To_Expiry"].min())

            exp_0_30_units = float(
                exp_rows.loc[exp_rows["Days_To_Expiry"] <= 30, "On_Hand"].sum()
            )
            exp_31_90_units = float(
                exp_rows.loc[
                    (exp_rows["Days_To_Expiry"] > 30)
                    & (exp_rows["Days_To_Expiry"] <= 90),
                    "On_Hand",
                ].sum()
            )
            exp_gt_90_units = float(
                exp_rows.loc[exp_rows["Days_To_Expiry"] > 90, "On_Hand"].sum()
            )

            exp_soon_units = float(
                exp_rows.loc[exp_rows["Days_To_Expiry"] <= 90, "On_Hand"].sum()
            )
            exp_soon_value = float(
                exp_rows.loc[exp_rows["Days_To_Expiry"] <= 90, "Inventory_Value"].sum()
            )

        # Recommended production
        recommended_prod_units = max(
            0, round(forecast_12m_units + safety_stock_units - on_hand_units)
        )
        recommended_prod_value = recommended_prod_units * unit_cost

        # Simple inventory risk label
        if total_on_hand == 0:
            risk_label = "No Stock"
        else:
            exp_ratio = exp_soon_units / total_on_hand if total_on_hand > 0 else 0
            if exp_ratio >= 0.5:
                risk_label = "High Expiry Risk"
            elif exp_ratio >= 0.2:
                risk_label = "Moderate Expiry Risk"
            else:
                risk_label = "Low Expiry Risk"

        # Try to capture a product family if present
        product_family = (
            sku_hist["Product_Family"].iloc[0]
            if "Product_Family" in sku_hist.columns
            else ""
        )

        results.append(
            {
                "SKU": sku,
                "Product_Family": product_family,
                "Lead_Time_Days": lead_time_days,
                "Shelf_Life_Days": float(sku_hist["Shelf_Life_Days"].iloc[0]),
                "Unit_Cost": unit_cost,
                # Forecast & stock
                "Forecast_12M_Units": round(forecast_12m_units),
                "Forecast_12M_Value": round(forecast_12m_units * unit_cost, 2),
                "Safety_Stock_Units": safety_stock_units,
                "Safety_Stock_Value": round(safety_stock_units * unit_cost, 2),
                "On_Hand_Units": round(on_hand_units),
                "On_Hand_Value": round(on_hand_units * unit_cost, 2),
                # Expiry metrics
                "Total_On_Hand_Units": round(total_on_hand),
                "Total_On_Hand_Value": round(total_inv_value, 2),
                "Expiring_0_30_Units": round(exp_0_30_units),
                "Expiring_31_90_Units": round(exp_31_90_units),
                "Expiring_>90_Units": round(exp_gt_90_units),
                "Expiring_Soon_Units_<=90": round(exp_soon_units),
                "Expiring_Soon_Value_<=90": round(exp_soon_value, 2),
                "Min_Days_To_Expiry": min_days_to_expiry,
                # Recommendation
                "Recommended_Production_Units": recommended_prod_units,
                "Recommended_Production_Value": round(recommended_prod_value, 2),
                # Risk label
                "Inventory_Risk_Label": risk_label,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert dataframe to Excel bytes for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="OptiStock_Output")
    return output.getvalue()


# -------------------------
# Streamlit App
# -------------------------

st.set_page_config(page_title="OptiStock – Inventory Optimization", layout="wide")

st.title("OptiStock – AI Inventory & Expiry Intelligence")

st.write(
    "Upload your combined inventory & demand dataset and OptiStock will:\n"
    "- Forecast demand (light model)\n"
    "- Calculate safety stock\n"
    "- Evaluate expiry risk\n"
    "- Recommend production levels\n"
    "- Highlight top SKUs by value and risk"
)

# ---- File upload (single file, used for all tabs) ----
uploaded_file = st.file_uploader(
    "Upload your combined OptiStock dataset (e.g., pharma_inventory_master_optistock.xlsx)",
    type=["xlsx"],
)

if uploaded_file is None:
    st.info("👆 Please upload your combined dataset to begin.")
    st.stop()

# Read and validate data
try:
    raw_df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

missing_cols = validate_columns(raw_df)
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

# Ensure proper types
raw_df["Date"] = pd.to_datetime(raw_df["Date"])
raw_df["Expiration_Date"] = pd.to_datetime(raw_df["Expiration_Date"])

# Run the light forecasting + expiry logic once
results_df = compute_light_forecast(raw_df)

# Cache in session (optional)
st.session_state["optistock_results"] = results_df
st.session_state["optistock_raw"] = raw_df

# -------------------------
# Executive metrics for summary
# -------------------------

total_forecast_units = results_df["Forecast_12M_Units"].sum()
total_recommended_value = results_df["Recommended_Production_Value"].sum()
total_onhand_value = results_df["Total_On_Hand_Value"].sum()
total_at_risk_value = results_df["Expiring_Soon_Value_<=90"].sum()
risk_pct = (total_at_risk_value / total_onhand_value * 100) if total_onhand_value > 0 else 0.0

# Top expiry-risk SKUs
top_expiry_by_value = (
    results_df[results_df["Expiring_Soon_Value_<=90"] > 0]
    .sort_values("Expiring_Soon_Value_<=90", ascending=False)
    .head(5)
)

# Top production SKUs
top_prod_by_value = (
    results_df.sort_values("Recommended_Production_Value", ascending=False)
    .head(5)
)

# Build narrative insights
insights = []

if not top_expiry_by_value.empty:
    top_expiry_names = ", ".join(top_expiry_by_value["SKU"].astype(str).tolist())
    share = (
        top_expiry_by_value["Expiring_Soon_Value_<=90"].sum() / total_at_risk_value * 100
        if total_at_risk_value > 0
        else 0
    )
    insights.append(
        f"SKUs **{top_expiry_names}** account for about **{share:,.0f}%** of total expiry risk (≤ 90 days)."
    )

if not top_prod_by_value.empty:
    top_prod_names = ", ".join(top_prod_by_value["SKU"].astype(str).tolist())
    insights.append(
        f"Recommended production value is concentrated in **{top_prod_names}**, indicating key manufacturing priorities."
    )

if total_onhand_value > 0:
    insights.append(
        f"Approximately **{risk_pct:,.1f}%** of total on-hand inventory value is at risk of expiring within 90 days."
    )

if total_forecast_units > 0:
    insights.append(
        f"12-month forecasted demand across all SKUs is **{total_forecast_units:,.0f} units**, driving production and stock positioning."
    )

if not insights:
    insights.append("No significant expiry or production risk patterns detected in the current dataset.")

# -------------------------
# TABS
# -------------------------

tab_summary, tab_forecast, tab_expiry, tab_top10, tab_download = st.tabs(
    ["📊 Executive Summary", "📈 Forecast & Plan", "⏳ Expiry Risk", "⭐ Top 10 SKUs", "📥 Download"]
)

# ========== EXECUTIVE SUMMARY TAB ==========
with tab_summary:
    st.subheader("Executive Summary")

    # KPI tiles
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Forecasted Demand (12M)", f"{total_forecast_units:,.0f} units")
    col2.metric("Total Recommended Production Value", f"${total_recommended_value:,.0f}")
    col3.metric("Inventory Value at Risk (≤ 90d)", f"${total_at_risk_value:,.0f}")
    col4.metric("% of Inventory Value at Risk", f"{risk_pct:,.1f}%")

    st.markdown("---")
    st.markdown("### Top Expiry Risk – By Value (≤ 90 Days)")

    if not top_expiry_by_value.empty:
        chart_data = top_expiry_by_value.set_index("SKU")["Expiring_Soon_Value_<=90"]
        st.bar_chart(chart_data)
    else:
        st.write("No SKUs with expiry risk within 90 days in this dataset.")

    st.markdown("---")
    st.markdown("### Key Insights")

    for insight in insights:
        st.markdown(f"- {insight}")

    st.markdown("---")
    st.markdown("### Data Snapshot")
    st.write("Preview of processed OptiStock results (first 10 rows):")
    st.dataframe(results_df.head(10))

# ========== FORECAST & PLAN TAB ==========
with tab_forecast:
    st.subheader("Forecast & Production Plan")

    st.write(
        "This table shows forecasted 12-month demand, safety stock, on-hand inventory, "
        "and recommended production by SKU."
    )

    st.dataframe(
        results_df[
            [
                "SKU",
                "Product_Family",
                "Forecast_12M_Units",
                "Forecast_12M_Value",
                "Safety_Stock_Units",
                "Safety_Stock_Value",
                "On_Hand_Units",
                "On_Hand_Value",
                "Recommended_Production_Units",
                "Recommended_Production_Value",
                "Inventory_Risk_Label",
            ]
        ].sort_values("Recommended_Production_Value", ascending=False)
    )

    st.markdown("### Summary – Total Recommended Production Value")
    st.metric(
        "Total Recommended Production (Value)",
        f"${results_df['Recommended_Production_Value'].sum():,.0f}",
    )

# ========== EXPIRY RISK TAB ==========
with tab_expiry:
    st.subheader("Expiry Risk Dashboard")

    st.write(
        "This view summarizes units and value at expiry risk across SKUs, focusing on inventory "
        "expiring within the next 90 days."
    )

    agg = results_df.copy()

    total_on_hand_units = agg["Total_On_Hand_Units"].sum()
    exp_0_30_units = agg["Expiring_0_30_Units"].sum()
    exp_31_90_units = agg["Expiring_31_90_Units"].sum()
    exp_soon_units = agg["Expiring_Soon_Units_<=90"].sum()
    exp_soon_value = agg["Expiring_Soon_Value_<=90"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total On-hand Units", f"{total_on_hand_units:,.0f}")
    col2.metric("Units Expiring ≤ 90 Days", f"{exp_soon_units:,.0f}")
    col3.metric("Value Expiring ≤ 90 Days", f"${exp_soon_value:,.0f}")

    st.markdown("### Expiry Buckets (Units)")
    expiry_buckets = (
        pd.DataFrame(
            {
                "Bucket": ["0–30 days", "31–90 days", ">90 days"],
                "Units": [
                    exp_0_30_units,
                    exp_31_90_units,
                    agg["Expiring_>90_Units"].sum(),
                ],
            }
        )
        .set_index("Bucket")
    )
    st.bar_chart(expiry_buckets)

    st.markdown("### SKUs with Highest Expiry Risk (Units ≤ 90 days)")
    top_expiry = (
        results_df[results_df["Expiring_Soon_Units_<=90"] > 0]
        .sort_values("Expiring_Soon_Units_<=90", ascending=False)
        .head(20)
    )
    st.dataframe(
        top_expiry[
            [
                "SKU",
                "Product_Family",
                "Total_On_Hand_Units",
                "Expiring_Soon_Units_<=90",
                "Expiring_Soon_Value_<=90",
                "Inventory_Risk_Label",
            ]
        ]
    )

# ========== TOP 10 SKUs TAB ==========
with tab_top10:
    st.subheader("Top 10 SKUs by Recommended Production Value")

    top10 = (
        results_df.sort_values("Recommended_Production_Value", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    st.dataframe(
        top10[
            [
                "SKU",
                "Product_Family",
                "Recommended_Production_Units",
                "Recommended_Production_Value",
                "Total_On_Hand_Units",
                "Expiring_Soon_Units_<=90",
                "Expiring_Soon_Value_<=90",
                "Inventory_Risk_Label",
            ]
        ]
    )

    st.markdown("### Recommended Production Value – Top 10 SKUs")
    chart_data = top10.set_index("SKU")["Recommended_Production_Value"]
    st.bar_chart(chart_data)

# ========== DOWNLOAD TAB ==========
with tab_download:
    st.subheader("Download OptiStock Results")

    st.write(
        "Download the full SKU-level table including forecasts, safety stock, expiry metrics, "
        "and recommended production values."
    )

    excel_bytes = to_excel_bytes(results_df)

    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_bytes,
        file_name="optistock_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.write("You can share this file with stakeholders, or use it as input to PowerPoint reports.")
