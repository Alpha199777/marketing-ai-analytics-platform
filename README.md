# Marketing AI Analytics Platform (Databricks / Fabric-ready)

This project demonstrates how to turn raw marketing campaign data into
an AI-ready analytics and decision platform:

- KPI metrics layer (CTR, CVR, CPL, ROI)
- Revenue prediction using PySpark MLlib
- Campaign segmentation with KMeans
- Ready to be consumed by BI tools, LLMs or AI agents

---

## Business use case

Marketing teams need to:
- Understand which campaigns generate value
- Predict revenue for future revenue
- Identify segments of high / low performance

This platform builds a **metrics and ML foundation** to support those decisions.

---

## Architecture

Raw marketing data
↓
Curated table (marketing_clean)
↓
KPI computation → marketing_kpi (Delta / Lakehouse)
↓
ML layer:

Revenue prediction (RandomForest)

Campaign clustering (KMeans)
↓
Consumption:

BI dashboards

AI assistants (RAG on structured data)

Web apps / APIs



## Notebook

The core logic lives in:

`notebooks/01_marketing_kpi_revenue_prediction.ipynb`

It performs:
1. Data loading from `marketing_clean`
2. KPI computation (CTR, CVR, CPL, ROI)
3. ML training (revenue prediction)
4. Revenue simulation for a new campaign
5. Campaign segmentation via clustering

---

## Why this matters for Applied AI

The table `marketing_kpi` acts as a **metrics layer** that can be:
- Queried by BI tools
- Used by RAG systems
- Used by LLM-based agents to answer questions like:  
  “Which campaign has the best ROI?”  
  “What happens if we double the budget?”

This is the foundation of **AI on enterprise data**.

---

## How to run

### Option 1 — In Databricks
Import the notebook and run it on a cluster with access to `marketing_clean`.

### Option 2 — Offline demo
A small CSV sample is provided in `data/sample_marketing.csv`.

---

## Evidence

Screenshots and notebooks in this repo show:
- KPI tables
- ML predictions
- Clustering outputs  

This is a real, end-to-end data & AI pipeline.


Improve README with notebook, AI and usage details

