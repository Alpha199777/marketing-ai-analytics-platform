import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def safe_div_series(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return (a / b).replace([np.inf, -np.inf], np.nan)

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

def pick_col(df: pd.DataFrame, candidates):
    """Return first existing col from candidates else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ----------------------------
# Header
# ----------------------------
st.title("Marketing AI Analytics Platform")
st.caption("📊 KPI • ROI/ROAS • Prediction • Clustering (portfolio pro)")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")

    data_mode = st.radio(
        "Source de données",
        ["CSV du repo (recommandé)", "Uploader un CSV"],
        index=0
    )

    uploaded_file = None
    if data_mode == "Uploader un CSV":
        uploaded_file = st.file_uploader("Uploader un CSV", type=["csv"])

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

# ----------------------------
# Column mapping (robuste)
# ----------------------------
date_col = pick_col(df, ["date", "c_date"])
campaign_col = pick_col(df, ["campaign", "campaign_name"])
revenue_col = pick_col(df, ["revenue"])
clicks_col = pick_col(df, ["clicks"])
impr_col = pick_col(df, ["impressions"])
spend_col = pick_col(df, ["mark_spent", "cost", "spend"])
orders_col = pick_col(df, ["orders"])
leads_col = pick_col(df, ["leads"])
category_col = pick_col(df, ["category"])
campaign_id_col = pick_col(df, ["campaign_id", "id"])

required_min = [date_col, campaign_col, revenue_col]
if any(c is None for c in required_min):
    st.error(
        "Colonnes minimales requises manquantes.\n\n"
        "Il faut au minimum : (date/c_date), (campaign/campaign_name), revenue.\n"
        f"Colonnes trouvées: {list(df.columns)}"
    )
    st.stop()

# ----------------------------
# Clean & normalize
# ----------------------------
df = df.copy()
df["date"] = pd.to_datetime(df[date_col], errors="coerce")
df["campaign"] = df[campaign_col].astype(str)

# numeric casts if present
for col in [revenue_col, clicks_col, impr_col, spend_col, orders_col, leads_col]:
    if col is not None:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["revenue"] = df[revenue_col]

if clicks_col is not None:
    df["clicks"] = df[clicks_col]
else:
    df["clicks"] = 0

if impr_col is not None:
    df["impressions"] = df[impr_col]
else:
    df["impressions"] = 0

if spend_col is not None:
    df["spend"] = df[spend_col]
else:
    df["spend"] = np.nan  # spend absent

if orders_col is not None:
    df["orders"] = df[orders_col]
else:
    df["orders"] = np.nan

if leads_col is not None:
    df["leads"] = df[leads_col]
else:
    df["leads"] = np.nan

if category_col is not None:
    df["category"] = df[category_col].astype(str)
else:
    df["category"] = "unknown"

if campaign_id_col is not None:
    df["campaign_id"] = df[campaign_id_col]
else:
    df["campaign_id"] = np.nan

df = df.dropna(subset=["date"]).copy()

# Derived KPIs
df["ctr"] = safe_div_series(df["clicks"], df["impressions"])
df["rev_per_click"] = safe_div_series(df["revenue"], df["clicks"])
df["rev_per_1k_impr"] = safe_div_series(df["revenue"] * 1000, df["impressions"])

# ROI/ROAS if spend exists
if spend_col is not None:
    df["roas"] = safe_div_series(df["revenue"], df["spend"])
    df["roi"] = safe_div_series(df["revenue"] - df["spend"], df["spend"])
else:
    df["roas"] = np.nan
    df["roi"] = np.nan

# ----------------------------
# Filters
# ----------------------------
campaigns = sorted(df["campaign"].unique().tolist())
min_date = df["date"].min().date()
max_date = df["date"].max().date()

with st.sidebar:
    st.divider()
    st.subheader("🔎 Filtres")
    selected_campaigns = st.multiselect("Campagnes", options=campaigns, default=campaigns)
    date_range = st.date_input("Période", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (
    df["campaign"].isin(selected_campaigns)
    & (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
)
dff = df.loc[mask].copy()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Dashboard", "📈 ROI/ROAS", "🤖 Prediction", "🧩 Clustering"])

# =========================================================
# TAB 1 — Dashboard
# =========================================================
with tab1:
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

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("📈 Tendance (au choix)")

        ts = (
            dff.groupby(pd.Grouper(key="date", freq="D"))
            .agg(
                revenue=("revenue", "sum"),
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum"),
            )
            .reset_index()
            .sort_values("date")
        )
        ts["ctr"] = safe_div_series(ts["clicks"], ts["impressions"]).fillna(0)

        metric = st.selectbox("Métrique", ["revenue", "clicks", "impressions", "ctr"], index=0)

        if ts.empty:
            st.warning("Aucune donnée à afficher.")
        else:
            fig = px.line(ts, x="date", y=metric, markers=True)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("🏆 Top campagnes")

        by_campaign = (
            dff.groupby("campaign", as_index=False)
            .agg(
                revenue=("revenue", "sum"),
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum"),
            )
        )
        by_campaign["ctr"] = safe_div_series(by_campaign["clicks"], by_campaign["impressions"]).fillna(0)
        by_campaign["rev_per_click"] = safe_div_series(by_campaign["revenue"], by_campaign["clicks"]).fillna(0)

        sort_by = st.selectbox("Trier par", ["revenue", "clicks", "impressions", "ctr", "rev_per_click"], index=0)
        top = by_campaign.sort_values(sort_by, ascending=False).head(top_n)

        fig2 = px.bar(top, x="campaign", y=sort_by)
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    if show_table:
        st.subheader("📄 Données filtrées")
        show_cols = ["date", "campaign", "category", "revenue", "clicks", "impressions", "ctr", "rev_per_click", "rev_per_1k_impr", "spend", "roi", "roas", "orders", "leads"]
        show_cols = [c for c in show_cols if c in dff.columns]
        table = dff[show_cols].sort_values("date", ascending=False)
        st.dataframe(table, use_container_width=True, height=420)

        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger les données filtrées (CSV)",
            data=csv_bytes,
            file_name="marketing_filtered.csv",
            mime="text/csv"
        )

# =========================================================
# TAB 2 — ROI/ROAS
# =========================================================
with tab2:
    st.subheader("📈 ROI / ROAS")

    if spend_col is None:
        st.warning("Aucune colonne de coût/spend détectée (ex: mark_spent). ROI/ROAS indisponibles.")
    else:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Spend total", format_money(dff["spend"].sum()))
        r2.metric("ROAS moyen", format_num(np.nanmean(dff["roas"])))
        r3.metric("ROI moyen", format_pct(np.nanmean(dff["roi"])))
        r4.metric("Revenue total", format_money(dff["revenue"].sum()))

        st.divider()

        by_campaign = (
            dff.groupby("campaign", as_index=False)
            .agg(
                revenue=("revenue", "sum"),
                spend=("spend", "sum"),
            )
        )
        by_campaign["roas"] = safe_div_series(by_campaign["revenue"], by_campaign["spend"])
        by_campaign["roi"] = safe_div_series(by_campaign["revenue"] - by_campaign["spend"], by_campaign["spend"])

        metric = st.selectbox("Afficher", ["roas", "roi"], index=0)

        fig = px.bar(by_campaign.sort_values(metric, ascending=False), x="campaign", y=metric)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 — Prediction revenue
# =========================================================
with tab3:
    st.subheader("🤖 Prédire le revenue (RandomForest)")

    # features candidates
    feature_cols = []
    for c in ["impressions", "clicks", "leads", "orders", "spend", "ctr"]:
        if c in dff.columns and dff[c].notna().any():
            feature_cols.append(c)

    if len(feature_cols) < 2:
        st.warning("Pas assez de features (colonnes) pour entraîner un modèle. Il faut au moins 2 features utiles.")
    else:
        model_df = dff.dropna(subset=["revenue"]).copy()
        for c in feature_cols:
            model_df[c] = pd.to_numeric(model_df[c], errors="coerce").fillna(0)

        X = model_df[feature_cols]
        y = model_df["revenue"]

        if len(model_df) < 30:
            st.warning("Dataset trop petit (< 30 lignes). Ajoute plus de données pour une prédiction crédible.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            rf = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{r2:.3f}")
            c2.metric("MAPE (erreur %)", f"{mape*100:.1f}%")
            c3.metric("Features utilisées", ", ".join(feature_cols))

            st.divider()
            st.markdown("### 🧪 Simuler une campagne")

            inputs = {}
            cols = st.columns(len(feature_cols))
            for i, colname in enumerate(feature_cols):
                default_val = float(np.nanmedian(model_df[colname])) if np.isfinite(np.nanmedian(model_df[colname])) else 0.0
                inputs[colname] = cols[i].number_input(colname, value=float(default_val), step=1.0)

            X_new = pd.DataFrame([inputs])
            pred_new = float(rf.predict(X_new)[0])
            st.success(f"📌 Revenue prédit : **{format_money(pred_new)}**")

# =========================================================
# TAB 4 — Clustering
# =========================================================
with tab4:
    st.subheader("🧩 Segmentation des campagnes (KMeans)")

    cluster_features = []
    for c in ["revenue", "spend", "ctr", "clicks", "impressions", "orders", "leads", "roas", "roi"]:
        if c in dff.columns and dff[c].notna().any():
            cluster_features.append(c)

    if len(cluster_features) < 3:
        st.warning("Pas assez de features pour clusteriser (il en faut au moins 3).")
    else:
        # cluster par campagne (agrégation)
        agg = (
            dff.groupby("campaign", as_index=False)
            .agg({c: "mean" for c in cluster_features})
        )

        data_cluster = agg[cluster_features].copy()
        data_cluster = data_cluster.replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(data_cluster)

        k = st.slider("Nombre de clusters (k)", 2, 6, 4)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        agg["cluster"] = km.fit_predict(Xs)

        st.divider()
        st.markdown("### 📌 Résumé clusters")
        summary = agg.groupby("cluster")[cluster_features].mean().reset_index()
        st.dataframe(summary, use_container_width=True)

        st.divider()
        st.markdown("### 📍 Visualisation (scatter)")
        x_axis = st.selectbox("Axe X", cluster_features, index=0)
        y_axis = st.selectbox("Axe Y", cluster_features, index=min(1, len(cluster_features)-1))

        fig = px.scatter(
            agg,
            x=x_axis,
            y=y_axis,
            color="cluster",
            hover_data=["campaign"],
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
