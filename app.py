import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

from prophet import Prophet

st.set_page_config(page_title="OptiStock – Inventory Optimization", layout="wide")

st.title("OptiStock – AI Inventory Forecast & Production Planner")
st.write(
    "Upload your inventory / demand dataset and OptiStock will forecast demand, "
    "calculate safety stock, and recommend production levels by SKU."
)

# --- File upload ---
uploaded_file = st.file_uploader(
    "Upload your synthetic inventory dataset (Excel)",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.info("👆 Upload your synthetic file (e.g., `pharma_inventory_synthetic.xlsx`) to begin.")
    st.stop()

# --- Read data ---
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(df.head())

st.write("Columns detected:", list(df.columns))

required_cols = ["SKU", "Date", "Monthly_Demand", "Lead_Time_Days", "Shelf_Life_Days", "Unit_Cost"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"❌ Missing required columns in dataset: {missing}")
    st.stop()

# Ensure Date is datetime
df["Date"] = pd.to_datetime(df["Date"])

# Button to run forecasting logic
if not st.button("Run Forecasting & Generate Recommended Production"):
    st.stop()

st.success("Running forecasts and optimization... this may take a bit for many SKUs.")

sku_list = df["SKU"].unique()
forecasts = {}
safety_stock = {}
risk_flags = {}

Z = 1.65  # approx 95% service level

# --- Forecast + safety stock + risk flag per SKU ---
for sku in sku_list:
    sku_data = df[df["SKU"] == sku][["Date", "Monthly_Demand"]].rename(
        columns={"Date": "ds", "Monthly_Demand": "y"}
    )

    # Basic Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(sku_data)

    future = model.make_future_dataframe(periods=12, freq="MS")
    forecast = model.predict(future)

    # keep last 12 months forecast
    forecasts[sku] = forecast[["ds", "yhat"]].tail(12)

    # Safety stock
    lead_time = df[df["SKU"] == sku]["Lead_Time_Days"].iloc[0]
    sigma = df[df["SKU"] == sku]["Monthly_Demand"].std()
    if np.isnan(sigma) or sigma == 0:
        sigma = 1  # avoid zero SS for static SKUs
    safety_stock[sku] = int(Z * sigma * np.sqrt(lead_time / 30))

    # Simple risk flag
    shelf_life = df[df["SKU"] == sku]["Shelf_Life_Days"].iloc[0]
    avg_demand = df[df["SKU"] == sku]["Monthly_Demand"].mean()
    if shelf_life < 365 or avg_demand < 50:
        risk_flags[sku] = "High Risk"
    else:
        risk_flags[sku] = "Low Risk"

# --- Build results table ---
results = []

has_on_hand = "On_Hand" in df.columns

for sku in sku_list:
    forecast_12m = forecasts[sku]["yhat"].sum()
    product_family = df[df["SKU"] == sku]["Product_Family"].iloc[0] if "Product_Family" in df.columns else ""
    unit_cost = df[df["SKU"] == sku]["Unit_Cost"].iloc[0]

    ss = safety_stock[sku]

    # Use On_Hand if present, else assume 0 for now
    if has_on_hand:
        on_hand_val = df[df["SKU"] == sku]["On_Hand"].iloc[0]
    else:
        on_hand_val = 0

    recommended_production_units = max(0, round(forecast_12m + ss - on_hand_val))

    results.append({
        "SKU": sku,
        "Product_Family": product_family,
        "On_Hand" if has_on_hand else "On_Hand (assumed 0)": on_hand_val,
        "Forecast_12M": round(forecast_12m),
        "Forecast_12M_Value": round(forecast_12m * unit_cost, 2),
        "Safety_Stock": ss,
        "Safety_Stock_Value": round(ss * unit_cost, 2),
        "Inventory_Risk": risk_flags[sku],
        "Recommended_Production": recommended_production_units,
        "Recommended_Production_Value": round(recommended_production_units * unit_cost, 2),
    })

results_df = pd.DataFrame(results)

st.subheader("Recommended Production by SKU")
st.dataframe(results_df)

# --- Simple summary by product family (if exists) ---
if "Product_Family" in results_df.columns:
    st.subheader("Summary by Product Family (Recommended Production Value)")
    summary_family = (
        results_df.groupby("Product_Family")["Recommended_Production_Value"]
        .sum()
        .reset_index()
        .sort_values("Recommended_Production_Value", ascending=False)
    )
    st.bar_chart(
        summary_family.set_index("Product_Family")["Recommended_Production_Value"]
    )

# --- Download Excel of results ---
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="OptiStock_Output")
    processed_data = output.getvalue()
    return processed_data

excel_bytes = to_excel_bytes(results_df)

st.download_button(
    label="📥 Download Recommended Production (Excel)",
    data=excel_bytes,
    file_name="optistock_recommended_production.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

if not has_on_hand:
    st.info(
        "Note: Your dataset does not contain an 'On_Hand' column. "
        "Recommended production is calculated assuming current on-hand = 0. "
        "You can enhance this later by adding an On_Hand snapshot to the dataset."
    )
