import streamlit as st
import pandas as pd

st.title("Marketing AI Analytics Platform")
st.write("Dashboard for Marketing KPI, Revenue Prediction, and Campaign Segmentation")

# --- Load data ---
path = "data/sample_marketing.csv"
data = pd.read_csv(path)

# Normalize column names (lowercase + remove spaces)
data.columns = (
    data.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

st.subheader("Dataset columns (debug)")
st.write(list(data.columns))

# Map possible column names -> standard names
# Adjust this mapping if you know the exact names in your CSV
rename_map = {}
if "cost" not in data.columns:
    for alt in ["spend", "ad_spend", "campaign_cost", "total_cost", "amount_spent"]:
        if alt in data.columns:
            rename_map[alt] = "cost"
            break

if "revenue" not in data.columns:
    for alt in ["sales", "income", "turnover", "total_revenue"]:
        if alt in data.columns:
            rename_map[alt] = "revenue"
            break

if rename_map:
    data = data.rename(columns=rename_map)

# Check required columns
required = ["cost", "revenue"]
missing = [c for c in required if c not in data.columns]
if missing:
    st.error(
        f"Missing required columns: {missing}. "
        f"Available columns: {list(data.columns)}"
    )
    st.stop()

# --- KPIs ---
st.subheader("Key Metrics")
total_revenue = float(data["revenue"].sum())
total_cost = float(data["cost"].sum())

roi = (total_revenue - total_cost) / total_cost if total_cost != 0 else 0.0

st.metric("Total Revenue", f"{total_revenue:,.2f}")
st.metric("Total Cost", f"{total_cost:,.2f}")
st.metric("ROI", f"{roi:.2f}")

st.subheader("Campaign Data")
st.dataframe(data)
