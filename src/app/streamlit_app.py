import streamlit as st
import pandas as pd

# Title
st.title("Marketing AI Analytics Platform")

# Description
st.write("Dashboard for Marketing KPI, Revenue Prediction, and Campaign Segmentation")

# Load sample data
data = pd.read_csv("data/sample_marketing.csv")

# Show KPI
st.subheader("Key Metrics")

total_revenue = data["revenue"].sum()
total_cost = data["cost"].sum()
roi = (total_revenue - total_cost) / total_cost

st.metric("Total Revenue", f"${total_revenue:,.2f}")
st.metric("Total Cost", f"${total_cost:,.2f}")
st.metric("ROI", f"{roi:.2f}")

# Show data table
st.subheader("Campaign Data")
st.dataframe(data)
