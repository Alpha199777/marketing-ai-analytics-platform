import streamlit as st
import requests
import pandas as pd
import time

def load_marketing_kpi():
    host = st.secrets["DATABRICKS_HOST"]
    token = st.secrets["DATABRICKS_TOKEN"]
    warehouse_id = st.secrets["DATABRICKS_WAREHOUSE_ID"]

    url = f"{host}/api/2.0/sql/statements"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "statement": "SELECT * FROM marketing_kpi LIMIT 100",
        "warehouse_id": warehouse_id
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    statement_id = result["statement_id"]

    # attendre résultat
    while True:
        status_url = f"{host}/api/2.0/sql/statements/{statement_id}"
        status_response = requests.get(status_url, headers=headers)
        status_result = status_response.json()

        if status_result["status"]["state"] == "SUCCEEDED":
            break

        time.sleep(1)

    data = status_result["result"]["data_array"]
    columns = [col["name"] for col in status_result["result"]["schema"]["columns"]]

    return pd.DataFrame(data, columns=columns)


df = load_marketing_kpi()

st.title("Marketing AI Analytics Platform — LIVE Databricks")

st.dataframe(df)
