# Marketing AI Analytics Platform (Databricks / Microsoft Fabric Ready)

## Overview

This project demonstrates how to transform raw marketing campaign data into a production-ready analytics and AI decision platform using Databricks, Delta Lake, and PySpark MLlib.

The platform builds a structured KPI and machine learning layer that can be consumed by:

* BI tools (Power BI, dashboards)
* Retrieval-Augmented Generation (RAG) systems
* LLM-based assistants and AI agents
* Web applications and APIs

It serves as a foundation for enterprise AI systems operating on structured business data.

---

## Business Use Case

Marketing teams need to:

* Identify which campaigns generate the highest ROI
* Predict expected revenue for future campaigns
* Detect underperforming campaigns early
* Segment campaigns into performance groups
* Enable AI assistants to answer business questions

Example AI-powered questions supported:

* "Which campaigns are underperforming?"
* "Which segment has the highest ROI?"
* "What revenue can we expect if budget increases by 20%?"

---

## System Architecture

```
Raw marketing data (CSV / Database)
        ↓
Curated table (marketing_clean) – Delta Lake
        ↓
KPI computation layer (PySpark)
        ↓
marketing_kpi (Delta Table / Lakehouse)
        ↓
Machine Learning Layer
    - Revenue prediction (RandomForestRegressor)
    - Campaign segmentation (KMeans clustering)
        ↓
Consumption Layer
    - BI dashboards (Power BI / Fabric)
    - LLM assistants (RAG on structured data)
    - Streamlit web app
    - APIs and AI agents
```

---

## Data Layer

### Input Table: marketing_clean

Contains raw campaign data including:

* campaign_id
* impressions
* clicks
* conversions
* cost
* revenue
* channel / segment

### KPI Table: marketing_kpi (Delta Table)

Computed metrics include:

* CTR (Click Through Rate)
* CVR (Conversion Rate)
* CPL (Cost Per Lead)
* ROI (Return on Investment)

This table acts as the structured knowledge base for analytics and AI.

---

## Machine Learning Layer

### Revenue Prediction

Model: RandomForestRegressor (PySpark MLlib)

Purpose:

* Predict expected revenue for future campaigns
* Enable scenario simulation for decision-making

Example:

"What happens if budget increases?"

The model estimates revenue impact.

---

### Campaign Segmentation

Model: KMeans clustering

Purpose:

* Identify high-performing and low-performing campaign groups
* Enable strategic budget allocation

Output:

Campaigns grouped into performance clusters with interpretable business meaning.

---

## RAG (Retrieval-Augmented Generation) Integration

The table `marketing_kpi` serves as a structured knowledge base for AI assistants.

Retrieval layer:

* SQL queries retrieve relevant campaign records
* Filtering based on ROI, CVR, CPL, revenue

Generation layer:

* Retrieved KPI rows are injected into LLM prompts
* LLM generates grounded explanations and recommendations

Example workflow:

```
User question
   ↓
SQL retrieval from marketing_kpi
   ↓
Relevant campaign KPI rows
   ↓
Injected into LLM prompt
   ↓
LLM generates grounded business explanation
```

This enables enterprise-safe AI grounded in real business data.

---

## Notebook

Core implementation:

```
notebooks/01_marketing_kpi_revenue_prediction.ipynb
```

Key steps:

* Load marketing_clean dataset
* Compute KPI metrics (CTR, CVR, CPL, ROI)
* Train revenue prediction model
* Predict revenue for new campaigns
* Perform campaign clustering
* Save structured Delta table for downstream consumption

---

## Streamlit Application

A Streamlit web application exposes:

* Campaign KPI metrics
* Revenue predictions
* Campaign segmentation

This acts as a lightweight AI-ready decision interface.

---

## How to Run

### Option 1 — Databricks (Recommended)

1. Import notebook into Databricks workspace
2. Connect to Delta Lake / Fabric Lakehouse
3. Run notebook on a cluster

Output:

* marketing_kpi Delta table
* ML models
* KPI metrics layer

---

### Option 2 — Local Demo

Use sample dataset:

```
data/sample_marketing.csv
```

Run notebook locally using PySpark.

---

## Evidence and Verification

This repository contains verifiable implementation artifacts.

### Notebooks

```
notebooks/01_marketing_kpi_revenue_prediction.ipynb
```

Includes:

* KPI computation
* ML training
* Segmentation logic

---

### Screenshots (/screenshots)

* kpi_table.PNG — KPI metrics layer
* Graphique clusters.PNG — Campaign clusters visualization
* Graphique et commentaires clusters.PNG — Cluster interpretation
* Tableau des clusters.PNG — Campaign performance grouping
* ai_marketing_decision_platform.png — End-to-end architecture

These artifacts demonstrate a fully functional analytics and AI pipeline.

---

## Applied AI Relevance

This platform demonstrates core Applied AI and AI Engineering capabilities:

* Data pipeline engineering (PySpark, Delta Lake)
* Feature engineering and KPI layer design
* Machine learning model training and deployment
* Structured RAG knowledge base design
* AI-ready data architecture
* Integration readiness for LLM assistants and AI agents

---

## Technology Stack

* Databricks
* Delta Lake
* PySpark
* MLlib
* Microsoft Fabric compatible
* Streamlit
* Python
* SQL

LLM integration:

* Databricks Genie (Lakehouse native Generative AI)
* Databricks Lakehouse AI
* Databricks SQL AI Functions

Compatible with external LLMs:

* Azure OpenAI
* OpenAI API
* LangChain

---

## Future Improvements

* Add vector embeddings for hybrid structured/unstructured RAG
* Deploy as API endpoint
* Add automated evaluation pipeline
* Integrate with Azure OpenAI for live assistant
* Deploy on Microsoft Fabric

---

## Author

AI-ready marketing analytics and decision platform built using modern data and AI engineering practices.
