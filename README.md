# Applied AI Analytics Platform – ML & Decision Intelligence (Databricks / Fabric Ready)

## Overview

This project demonstrates how to transform raw marketing campaign data into a production-ready analytics and AI decision platform using Databricks, Delta Lake, and PySpark MLlib.

The platform builds a structured KPI and machine learning layer that can be consumed by:

* BI tools (Power BI, dashboards)
* Retrieval-Augmented Generation (RAG) systems
* LLM-based assistants and AI agents
* Web applications and APIs

Public Streamlit application:

https://marketing-ai-platform-alpha.streamlit.app

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
* "Simulate +20% budget on social campaigns"
* "Top 5 campaigns by ROI"
* "Analyse performance by category"

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
    - Budget simulation (RandomForest — real-time training)
        ↓
Autonomous Agent Layer (LangGraph + GPT-4o-mini)
    - Natural language query interface
    - Parameterized SQL tools
    - Business recommendations
        ↓
Consumption Layer
    - BI dashboards (Power BI / Fabric)
    - LLM assistants (RAG on structured data)
    - Streamlit web app (5 tabs)
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
* ROAS (Return on Ad Spend)
* mark_spent (budget allocated per campaign)

This table acts as the structured knowledge base for analytics and AI.

---

## Machine Learning Layer

### Revenue Prediction

Model: RandomForestRegressor (PySpark MLlib / scikit-learn)

Purpose:

* Predict expected revenue for future campaigns
* Enable scenario simulation for decision-making

Example:

"What happens if budget increases?"

The model estimates revenue impact.

---

### Budget Simulation (simulate_budget tool)

Model: RandomForestRegressor — retrained in real-time on Databricks data

Purpose:

* Simulate the impact of a budget increase (%) on expected revenue
* Available per category (social, search, influencer, media, all)

Output:

```
Simulation +20% budget (social):
  Budget actuel   : CHF 48 231
  Budget simulé   : CHF 57 877
  Revenue actuel  : CHF 423 890
  Revenue simulé  : CHF 461 204
  Delta revenue   : +CHF 37 314 (+8.8%)
  ROI incrémental : 77.4%
```

---

### Campaign Segmentation

Model: KMeans clustering

Purpose:

* Identify high-performing and low-performing campaign groups
* Enable strategic budget allocation

Output:

Campaigns grouped into performance clusters with interpretable business meaning and color-coded commentary.

---

## Autonomous Agent — LangGraph

The platform includes a fully autonomous marketing agent built with **LangGraph** and **GPT-4o-mini**.

### Workflow

```
User question (natural language)
        ↓
route_intent     → kpi_qa | diagnostic | segmentation | budget_simulation
        ↓
plan_tools       → LLM selects 1-2 tools and generates JSON call
        ↓
execute_tools    → Parameterized SQL queries on Databricks SQL Warehouse
        ↓
compose_answer   → Business response in French with 2-4 recommendations
```

### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `rank_campaigns` | Top/Bottom campaigns by metric | `metric`, `direction`, `limit` |
| `get_underperforming_campaigns` | Campaigns below ROI threshold | `roi_threshold`, `limit` |
| `aggregate_by_dimension` | KPI aggregation by category | `group_by` |
| `simulate_budget` | RandomForest budget simulation | `budget_increase_pct`, `category` |

### Security

* Parameterized SQL queries (SQL injection prevention)
* Whitelisted table: `marketing_kpi` only
* Forbidden operations: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`
* Whitelisted metrics: `roi`, `revenue`, `ctr`, `cvr`

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

## Streamlit Application — 5 Tabs

| Tab | Description |
|-----|-------------|
| 📌 **Dashboard** | Global KPIs, time series, top campaigns, KPI glossary |
| 💰 **ROI/ROAS** | Return on investment analysis by campaign |
| 🔮 **Prediction** | RandomForest revenue prediction with scenario simulation |
| 🧩 **Clustering** | KMeans campaign segmentation with color-coded commentary |
| 🧠 **Agent AI** | Autonomous LangGraph agent — natural language queries on Databricks |

### Live Application Demo

Public Streamlit application:

https://marketing-ai-platform-alpha.streamlit.app

Features available in the live demo:

- KPI dashboard (Revenue, CTR, ROAS, ROI) in CHF
- Campaign performance analysis with filters
- Machine learning revenue prediction (RandomForest)
- Campaign clustering (KMeans segmentation)
- Budget simulation (+X% impact on revenue)
- Autonomous AI agent with 4 tools
- Interactive filters and visualizations

This demonstrates a production-ready AI analytics platform architecture.

No local setup required — fully accessible online.

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

Configure environment variables (PowerShell):

```powershell
$env:DATABRICKS_SERVER_HOSTNAME = "dbc-xxx.cloud.databricks.com"
$env:DATABRICKS_HTTP_PATH       = "/sql/1.0/warehouses/xxx"
$env:DATABRICKS_ACCESS_TOKEN    = "dapiXXX"
$env:OPENAI_API_KEY             = "sk-XXX"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch application:

```bash
python -m streamlit run src/app/streamlit_app.py
```

---

### Option 3 — Streamlit Cloud Deployment

Add secrets in Streamlit Cloud (Settings → Secrets):

```toml
DATABRICKS_SERVER_HOSTNAME = "dbc-xxx.cloud.databricks.com"
DATABRICKS_HTTP_PATH       = "/sql/1.0/warehouses/xxx"
DATABRICKS_ACCESS_TOKEN    = "dapiXXX"
OPENAI_API_KEY             = "sk-XXX"
```

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
* Autonomous agent design with LangGraph (tool use, intent routing, planning)
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

* LangGraph (agent orchestration)
* LangChain (tool framework)
* GPT-4o-mini (OpenAI)
* Databricks Genie (Lakehouse native Generative AI)
* Databricks Lakehouse AI
* Databricks SQL AI Functions

Compatible with external LLMs:

* Azure OpenAI
* OpenAI API
* LangChain

---

## 🧬 Fine-Tuning LoRA

TinyLlama 1.1B fine-tuned with LoRA (PEFT/TRL) on 500+ proprietary marketing KPI examples.

**Model on HuggingFace Hub:** [Doers97/marketing-lora-tinyllama](https://huggingface.co/Doers97/marketing-lora-tinyllama)

| Metric | Value |
|--------|-------|
| Technique | LoRA (r=8, alpha=16) |
| Trainable params | 0.10% (1.1M / 1.1B) |
| Val loss | 0.24 |
| Training time | 5 min — T4 GPU (Google Colab) |

Example inference:
- ROI = -20% → `REVIEW immediately` ✅
- ROI = +381% → `SCALE immediately` ✅
## Future Improvements

* Add vector embeddings for hybrid structured/unstructured RAG
* Deploy as API endpoint
* Add automated evaluation pipeline
* Integrate with Azure OpenAI for live assistant
* Deploy on Microsoft Fabric
* Export PDF reports
* Automated alerts (ROI below threshold)
* Multi-table agent support

---

## Author

AI-ready marketing analytics and decision platform built using modern data and AI engineering practices.
