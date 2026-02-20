import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Marketing AI Analytics Platform",
    page_icon="📈",
    layout="wide",
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(ttl=300)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def ensure_columns(df: pd.DataFrame):
    required = {"date", "campaign", "clicks", "impressions", "revenue"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes: {sorted(list(missing))}. Colonnes dispo: {list(df.columns)}")
        st.stop()

def safe_div(a, b):
    return np.where(b != 0, a / b, np.nan)

def format_num(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "—"

def format_money(x):
    try:
        return f"{float(x):,.0f} €"
    except Exception:
        return "—"

def format_pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"

# ----------------------------
# Header
# ----------------------------
st.title("Marketing AI Analytics Platform")
st.caption("📊 Dashboard KPI + Analytics (version portfolio) — basé sur `data/sample_marketing.csv`")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    data_mode = st.radio("Source de données", ["CSV du repo (recommandé)", "Uploader un CSV"], index=0)

    uploaded_file = None
    if data_mode == "Uploader un CSV":
        uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

    st.divider()
    st.subheader("🔎 Filtres")
    # Les filtres sont appliqués après chargement

    st.divider()
    st.subheader("🧪 Options")
    show_table = st.checkbox("Afficher la table détaillée", value=True)
    top_n = st.slider("Top campagnes (bar chart)", 5, 50, 15)

# ----------------------------
# Load data
# ----------------------------
if data_mode == "CSV du repo (recommandé)":
    df = load_csv("data/sample_marketing.csv")
else:
    if uploaded_file is None:
        st.info("Upload un CSV pour continuer.")
        st.stop()
    df = pd.read_csv(uploaded_file)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

ensure_columns(df)

# Parse date
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()

# Cast numerics
for c in ["clicks", "impressions", "revenue"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Derived metrics
df["ctr"] = safe_div(df["clicks"], df["impressions"])
df["rev_per_click"] = safe_div(df["revenue"], df["clicks"])
df["rev_per_1k_impr"] = safe_div(df["revenue"] * 1000, df["impressions"])

# ----------------------------
# Build filters
# ----------------------------
campaigns = sorted(df["campaign"].dropna().astype(str).unique().tolist())
min_date = df["date"].min().date()
max_date = df["date"].max().date()

with st.sidebar:
    selected_campaigns = st.multiselect("Campagnes", options=campaigns, default=campaigns)
    date_range = st.date_input("Période", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Normalize date_range output
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (
    df["campaign"].astype(str).isin(selected_campaigns)
    & (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
)
dff = df.loc[mask].copy()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📌 Dashboard", "🧠 Insights", "ℹ️ About"])

with tab1:
    # ----------------------------
    # KPIs
    # ----------------------------
    total_revenue = dff["revenue"].sum()
    total_clicks = dff["clicks"].sum()
    total_impr = dff["impressions"].sum()

    ctr_avg = np.nanmean(dff["ctr"]) if len(dff) else np.nan
    rpc_avg = np.nanmean(dff["rev_per_click"]) if len(dff) else np.nan
    rpm_avg = np.nanmean(dff["rev_per_1k_impr"]) if len(dff) else np.nan

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Revenue total", format_money(total_revenue))
    k2.metric("Clicks", format_num(total_clicks))
    k3.metric("Impressions", format_num(total_impr))
    k4.metric("CTR moyen", format_pct(ctr_avg))
    k5.metric("€ / Click (moy.)", format_money(rpc_avg))
    k6.metric("€ / 1k Impr (moy.)", format_money(rpm_avg))

    st.divider()

    # ----------------------------
    # Charts row
    # ----------------------------
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("📈 Tendance revenue dans le temps")
       ts = (
    dff.groupby(pd.Grouper(key="date", freq="D"))
    .agg(revenue=("revenue", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
    .reset_index()
)

# forcer types propres
ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
for c in ["revenue", "clicks", "impressions"]:
    ts[c] = pd.to_numeric(ts[c], errors="coerce").fillna(0)

# CTR en pandas (évite np.where -> ndarray)
ts["ctr"] = (ts["clicks"] / ts["impressions"]).replace([np.inf, -np.inf], np.nan).fillna(0)

# tri
ts = ts.sort_values("date")

metric = st.selectbox("Métrique", ["revenue", "clicks", "impressions", "ctr"], index=0)

if ts.empty:
    st.warning("Aucune donnée à afficher pour cette période / ces campagnes.")
else:
    fig = px.line(ts, x="date", y=metric, markers=True)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True) 
    with right:
        st.subheader("🏆 Top campagnes")
        by_campaign = (
            dff.groupby("campaign", as_index=False)
            .agg(revenue=("revenue", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        )
        by_campaign["ctr"] = safe_div(by_campaign["clicks"], by_campaign["impressions"])
        by_campaign["rev_per_click"] = safe_div(by_campaign["revenue"], by_campaign["clicks"])

        sort_by = st.selectbox("Trier par", ["revenue", "clicks", "impressions", "ctr", "rev_per_click"], index=0)
        top = by_campaign.sort_values(sort_by, ascending=False).head(top_n)

        fig2 = px.bar(top, x="campaign", y=sort_by)
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ----------------------------
    # Table + download
    # ----------------------------
    if show_table:
        st.subheader("📄 Données filtrées")
        show_cols = ["date", "campaign", "clicks", "impressions", "revenue", "ctr", "rev_per_click", "rev_per_1k_impr"]
        table = dff[show_cols].sort_values("date", ascending=False)
        st.dataframe(table, use_container_width=True, height=420)

        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger les données filtrées (CSV)",
            data=csv_bytes,
            file_name="marketing_filtered.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("🧠 Insights automatiques (simples mais utiles)")

    if dff.empty:
        st.warning("Aucune donnée pour ces filtres.")
    else:
        by_campaign = (
            dff.groupby("campaign", as_index=False)
            .agg(revenue=("revenue", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        )
        by_campaign["ctr"] = safe_div(by_campaign["clicks"], by_campaign["impressions"])
        by_campaign["rev_per_click"] = safe_div(by_campaign["revenue"], by_campaign["clicks"])

        best_rev = by_campaign.sort_values("revenue", ascending=False).head(1)
        best_ctr = by_campaign.sort_values("ctr", ascending=False).head(1)
        best_rpc = by_campaign.sort_values("rev_per_click", ascending=False).head(1)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🥇 Revenue")
            st.write(best_rev[["campaign", "revenue"]].rename(columns={"revenue": "total_revenue"}))
        with c2:
            st.markdown("### 🎯 CTR")
            st.write(best_ctr[["campaign", "ctr"]])
        with c3:
            st.markdown("### 💶 € / Click")
            st.write(best_rpc[["campaign", "rev_per_click"]])

        st.divider()
        st.markdown("### ✅ Recos rapides")
        recos = []
        if float(np.nanmean(dff["ctr"])) < 0.005:
            recos.append("CTR faible : teste de nouveaux visuels / messages, et vérifie le ciblage.")
        recos.append("Compare les campagnes avec € / Click le plus élevé pour identifier les segments qui convertissent le mieux.")
        recos.append("Ajoute une colonne de coût (spend) plus tard pour calculer ROI/ROAS et prioriser budget.")
        for r in recos:
            st.write(f"- {r}")

with tab3:
    st.subheader("ℹ️ About / Portfolio")
    st.markdown(
        """
**Marketing AI Analytics Platform** — démonstration d’une mini plateforme analytics prête à partager dans un portfolio.

- Source actuelle : CSV (`data/sample_marketing.csv`)
- Fonctionnalités :
  - KPI (Revenue, Clicks, Impressions, CTR, € / Click, € / 1k Impr)
  - Filtres (campagnes + dates)
  - Graphiques (time series + top campagnes)
  - Export CSV filtré
- Prochaine étape “enterprise” :
  - Brancher Databricks / Fabric / API pour données live
  - Ajouter coût (spend), leads, orders → KPI ROI/ROAS + ML prediction
        """
    )
