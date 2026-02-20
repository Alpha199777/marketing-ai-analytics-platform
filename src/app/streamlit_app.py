import streamlit as st
import pandas as pd

# Title
st.title("Marketing AI Analytics Platform")

# Description
st.write("Dashboard for Marketing KPI and Campaign Analytics")

# Load data
data = pd.read_csv("data/sample_marketing.csv")

# Normalize column names
data.columns = (
    data.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Debug: show columns
st.subheader("Dataset columns (debug)")
st.write(list(data.columns))

# Check revenue exists
if "revenue" not in data.columns:
    st.error(f"Missing 'revenue' column. Available columns: {list(data.columns)}")
    st.stop()

# KPIs
st.subheader("Key Metrics")

total_revenue = float(data["revenue"].sum())

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"{total_revenue:,.2f}")

if "clicks" in data.columns:
    col2.metric("Total Clicks", int(data["clicks"].sum()))

if "impressions" in data.columns:
    col3.metric("Total Impressions", int(data["impressions"].sum()))

# Optional additional metrics if columns exist
if "clicks" in data.columns:
    total_clicks = int(data["clicks"].sum())
    st.metric("Total Clicks", total_clicks)

if "impressions" in data.columns:
    total_impressions = int(data["impressions"].sum())
    st.metric("Total Impressions", total_impressions)

# Show data table
st.subheader("Campaign Data")
st.dataframe(data)
