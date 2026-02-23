# Marketing AI Analytics Platform (Databricks / Microsoft Fabric Ready)

## Overview

This project demonstrates how to transform raw marketing campaign data into a production-ready analytics and AI decision platform using Databricks, Delta Lake, and PySpark MLlib.

Databricks + Delta Lake

PySpark ML

Databricks SQL Warehouse

LangGraph (AI agent orchestration)

Streamlit (interactive web application)

The platform combines:

Structured KPI engineering

Machine learning (prediction + segmentation)

A fully deployed Autonomous Marketing AI Agent

Executive-ready dashboard visualization

🔗 Live Application:
https://marketing-ai-platform-alpha.streamlit.app

🎯 Business Objective

Marketing teams need to:

Identify top-performing campaigns (ROI, revenue, CTR)

Detect underperforming campaigns early

Segment campaigns by performance level

Simulate budget increases before investing

Ask business questions in natural language

Example supported questions:

“Which campaigns are underperforming?”

“What are the 5 campaigns with the highest ROI?”

“Aggregate revenue by channel.”

“Simulate +20% budget on social campaigns.”

🏗 System Architecture
Raw Marketing Data (CSV / Database)
        ↓
Curated Table (marketing_clean) – Delta Lake
        ↓
KPI Computation Layer (PySpark)
        ↓
marketing_kpi (Delta Table / Lakehouse)
        ↓
Machine Learning Layer
    - Revenue Prediction (RandomForest)
    - Campaign Segmentation (KMeans)
        ↓
AI & Consumption Layer
    - Streamlit Dashboard
    - Marketing AI Agent (LangGraph)
    - Databricks SQL Warehouse
    - BI tools (Power BI / Fabric)
📊 Data & KPI Layer
Input Table: marketing_clean

Contains:

campaign_id

impressions

clicks

leads / orders

mark_spent

revenue

category / channel

KPI Table: marketing_kpi

Computed metrics:

CTR (%)

CVR (%)

CPL

ROI (%)

Revenue (CHF)

Spend (CHF)

This Delta table acts as the structured knowledge base for both analytics and AI.

🤖 Marketing Autonomous Agent

A production-ready AI agent built with LangGraph and connected directly to Databricks SQL Warehouse.

The agent:

Understands natural language business questions

Selects the correct SQL or ML tool

Queries real campaign data

Generates grounded recommendations

Formats outputs using business-ready standards (CHF, %, integer spacing)

🔧 Agent Tools

rank_campaigns

aggregate_by_dimension

get_underperforming_campaigns

segment_campaigns

simulate_budget (RandomForest-based simulation)

📈 Budget Simulation

The simulate_budget tool:

Retrieves historical campaign data

Trains a RandomForest model in real time

Simulates revenue impact of a budget increase

Returns:

Current spend (CHF)

Simulated spend (CHF)

Current revenue (CHF)

Predicted revenue (CHF)

Incremental ROI (%)

🎨 Streamlit Application

The Streamlit app includes:

Dashboard

KPI summary cards

Performance analysis

ROI vs Revenue clustering

Clean CHF formatting

ROI / CTR displayed as percentages (1 decimal)

Integers formatted with spaces (2 999 919)

🤖 Agent AI Tab

Natural language chat interface

Example business prompts

Real-time SQL grounding

Budget simulation tool

🔐 Production Deployment (Streamlit Cloud)

The AI agent runs in production using Streamlit Cloud.

Required Secrets
DATABRICKS_SERVER_HOSTNAME = "..."
DATABRICKS_HTTP_PATH = "..."
DATABRICKS_ACCESS_TOKEN = "dapi..."
OPENAI_API_KEY = "sk-..."
Required Dependencies (requirements.txt)

streamlit

pandas

numpy

plotly

scikit-learn

databricks-sql-connector

langchain

langchain-openai

langgraph

pydantic

After pushing to GitHub, Streamlit Cloud redeploys automatically.

⚠️ Common Issues & Fixes
No module named 'databricks'
python -m pip install databricks-sql-connector
Environment variable KeyError

Ensure variables are defined in the active terminal session.

Invalid credential error

Generate a new Databricks token (tokens start with dapi...).

Streamlit FileNotFoundError

Run from project root:

python -m streamlit run src/app/streamlit_app.py
🧠 Applied AI Capabilities Demonstrated

KPI engineering for structured AI grounding

SQL-based RAG on enterprise data

LangGraph agent orchestration

ML-based decision simulation

Production deployment with secrets management

Executive-ready UI formatting and consistency

🛠 Technology Stack

Databricks

Delta Lake

PySpark

Databricks SQL Warehouse

LangGraph

LangChain

OpenAI API

Streamlit

scikit-learn

Python

SQL

🔮 Future Improvements

Hybrid RAG (vector + structured)

Agent evaluation framework

Budget optimization with constraints

API deployment version

Azure OpenAI integration

👤 Author = Me

AI-ready marketing analytics and autonomous decision platform built using modern data and AI engineering practices.
