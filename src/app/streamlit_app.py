import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def map_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supporte plusieurs schémas:
    - light: date, campaign, clicks, impressions, revenue
    - enrichi: c_date/campaign_name/mark_spent/leads/orders/...
    """
    df = norm_cols(df)

    rename_map = {
        "c_date": "date",
        "campaign_name": "campaign",
        "mark_spent": "spend",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

@st.cache_data(ttl=300)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return map_schema(df)

def safe_div(a, b):
    a = np.array(a, dtype="float64")
    b = np.array(b, dtype="float64")
    out = np.full_like(a, np.nan, dtype="float64")
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out

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

def ensure_min_columns(df: pd.DataFrame):
    required = {"date", "campaign", "clicks", "impressions", "revenue"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes: {sorted(list(missing))}\n\nColonnes dispo: {list(df.columns)}")
        st.stop()

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Numerics (min)
    for c in ["clicks", "impressions", "revenue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Optionnels
    if "spend" in df.columns:
        df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0)
    if "leads" in df.columns:
        df["leads"] = pd.to_numeric(df["leads"], errors="coerce").fillna(0)
    if "orders" in df.columns:
        df["orders"] = pd.to_numeric(df["orders"], errors="coerce").fillna(0)

    # CTR
    df["ctr"] = safe_div(df["clicks"], df["impressions"])
    df["rev_per_click"] = safe_div(df["revenue"], df["clicks"])
    df["rev_per_1k_impr"] = safe_div(df["revenue"] * 1000, df["impressions"])

    # ROI / ROAS si spend présent
    if "spend" in df.columns:
        df["roas"] = safe_div(df["revenue"], df["spend"])  # revenue/spend
        df["roi"] = safe_div(df["revenue"] - df["spend"], df["spend"])  # (rev-spend)/spend

    return df

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

    data_mode = st.radio("Source de données", ["CSV du repo (recommandé)", "Uploader un CSV"], index=0)

    uploaded_file = None
    if data_mode == "Uploader un CSV":
        uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

    st.divider()
    st.subheader("🧪 Options")
    show_table = st.checkbox("Afficher la table détaillée", value=True)
    top_n = st.slider("Top campagnes (table + bar)", 5, 50, 15)

# ----------------------------
# Load
# ----------------------------
if data_mode == "CSV du repo (recommandé)":
    # 👉 Mets ici ton fichier enrichi si tu l'as nommé autrement
    df = load_csv("data/sample_marketing.csv")
else:
    if uploaded_file is None:
        st.info("Upload un CSV pour continuer.")
        st.stop()
    df = map_schema(pd.read_csv(uploaded_file))

ensure_min_columns(df)
df = compute_metrics(df)

# ----------------------------
# Filters
# ----------------------------
campaigns = sorted(df["campaign"].dropna().astype(str).unique().tolist())
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
    df["campaign"].astype(str).isin(selected_campaigns)
    & (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
)
dff = df.loc[mask].copy()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Dashboard", "💰 ROI/ROAS", "🔮 Prediction", "🧩 Clustering"])

# ============================================================
# TAB 1 — DASHBOARD
# ============================================================
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

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("📈 Tendance (au choix)")

        ts = (
            dff.groupby(pd.Grouper(key="date", freq="D"))
            .agg(revenue=("revenue", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
            .reset_index()
            .sort_values("date")
        )
        ts["ctr"] = (ts["clicks"] / ts["impressions"]).replace([np.inf, -np.inf], np.nan).fillna(0)

        metric = st.selectbox("Métrique", ["revenue", "clicks", "impressions", "ctr"], index=0)
        if ts.empty:
            st.warning("Aucune donnée à afficher pour cette période / ces campagnes.")
        else:
            fig = px.line(ts, x="date", y=metric, markers=True)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("🏆 Top campagnes (table + bar)")

        by_campaign = (
            dff.groupby("campaign", as_index=False)
            .agg(
                revenue=("revenue", "sum"),
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum"),
            )
        )
        by_campaign["ctr"] = safe_div(by_campaign["clicks"], by_campaign["impressions"])
        by_campaign["rev_per_click"] = safe_div(by_campaign["revenue"], by_campaign["clicks"])

        sort_by = st.selectbox("Trier par", ["revenue", "clicks", "impressions", "ctr", "rev_per_click"], index=0)
        top = by_campaign.sort_values(sort_by, ascending=False).head(top_n).copy()

        st.dataframe(
            top.rename(columns={
                "rev_per_click": "€/click"
            }),
            use_container_width=True,
            height=220
        )

        fig2 = px.bar(top, x="campaign", y=sort_by)
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    if show_table:
        st.subheader("📄 Données filtrées")
        cols = ["date", "campaign", "clicks", "impressions", "revenue", "ctr", "rev_per_click", "rev_per_1k_impr"]
        extra = [c for c in ["spend", "roi", "roas", "leads", "orders"] if c in dff.columns]
        cols = cols + extra

        table = dff[cols].sort_values("date", ascending=False)
        st.dataframe(table, use_container_width=True, height=420)

        st.download_button(
            "⬇️ Télécharger les données filtrées (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="marketing_filtered.csv",
            mime="text/csv"
        )

# ============================================================
# TAB 2 — ROI/ROAS
# ============================================================
with tab2:
    st.subheader("💰 ROI / ROAS")

    if "spend" not in dff.columns:
        st.warning("Aucune colonne de coût/spend détectée (ex: `spend` ou `mark_spent`). ROI/ROAS indisponibles.")
        st.info("👉 Ajoute `mark_spent` dans ton CSV (ou renomme en `spend`) pour activer cette page.")
    else:
        spend_total = dff["spend"].sum()
        roas_avg = np.nanmean(dff["roas"]) if len(dff) else np.nan
        roi_avg = np.nanmean(dff["roi"]) if len(dff) else np.nan

        a, b, c = st.columns(3)
        a.metric("Spend total", format_money(spend_total))
        b.metric("ROAS moyen", f"{roas_avg:.2f}x" if np.isfinite(roas_avg) else "—")
        c.metric("ROI moyen", format_pct(roi_avg) if np.isfinite(roi_avg) else "—")

        st.markdown(
            """
**Interprétation rapide :**
- **ROAS = revenue / spend** → “combien je récupère pour 1€ dépensé”
- **ROI = (revenue - spend) / spend** → “mon gain net relatif”
            """
        )

        by_campaign = (
            dff.groupby("campaign", as_index=False)
            .agg(revenue=("revenue", "sum"), spend=("spend", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        )
        by_campaign["roas"] = safe_div(by_campaign["revenue"], by_campaign["spend"])
        by_campaign["roi"] = safe_div(by_campaign["revenue"] - by_campaign["spend"], by_campaign["spend"])
        by_campaign["ctr"] = safe_div(by_campaign["clicks"], by_campaign["impressions"])

        left, right = st.columns(2)
        with left:
            st.markdown("### ✅ Top ROAS")
            top_roas = by_campaign.sort_values("roas", ascending=False).head(10)
            st.dataframe(top_roas, use_container_width=True, height=260)
        with right:
            st.markdown("### ⚠️ Worst ROI")
            worst_roi = by_campaign.sort_values("roi", ascending=True).head(10)
            st.dataframe(worst_roi, use_container_width=True, height=260)

# ============================================================
# TAB 3 — Prediction
# ============================================================
with tab3:
    st.subheader("🔮 Prédire le revenue (RandomForest)")
    st.markdown(
        """
Cette page entraîne un petit modèle **sur les données filtrées** pour prédire `revenue` à partir de signaux marketing.
- On applique **log1p(revenue)** pour limiter l’effet des campagnes “très grosses” (distribution asymétrique).
- On affiche **R²** (qualité globale) et **MAPE** (erreur relative) de façon stable.
        """
    )

    features_candidates = ["impressions", "clicks", "ctr"]
    if "spend" in dff.columns:
        features_candidates.append("spend")
    if "leads" in dff.columns:
        features_candidates.append("leads")
    if "orders" in dff.columns:
        features_candidates.append("orders")

    st.caption(f"Features possibles: {', '.join(features_candidates)}")
    features = st.multiselect("Features utilisées", options=features_candidates, default=features_candidates[:3])

    if len(dff) < 20 or len(features) < 2:
        st.warning("Il faut au moins ~20 lignes et 2 features pour entraîner un modèle stable.")
    else:
        X = dff[features].copy()
        y = dff["revenue"].copy()

        # log transform
        y_log = np.log1p(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)

        pred_log = model.predict(X_test)
        pred = np.expm1(pred_log)
        y_true = np.expm1(y_test)

        # R² sur log-space (plus stable)
        r2 = model.score(X_test, y_test)

        # MAPE stable: ignore y_true == 0
        y_true_np = np.array(y_true, dtype="float64")
        pred_np = np.array(pred, dtype="float64")
        mask = y_true_np > 0
        mape = np.mean(np.abs((y_true_np[mask] - pred_np[mask]) / y_true_np[mask])) if mask.any() else np.nan

        a, b, c = st.columns(3)
        a.metric("R² (log space)", f"{r2:.3f}")
        b.metric("MAPE (sur y>0)", format_pct(mape) if np.isfinite(mape) else "—")
        c.metric("Features utilisées", ", ".join(features))

        st.divider()
        st.markdown("### ✍️ Simuler une campagne")
        cols = st.columns(len(features))
        user_vals = {}
        for i, f in enumerate(features):
            with cols[i]:
                default_val = float(np.nanmedian(dff[f])) if np.isfinite(np.nanmedian(dff[f])) else 0.0
                user_vals[f] = st.number_input(f, value=float(default_val), step=1.0)

        sim = pd.DataFrame([user_vals])
        pred_sim = np.expm1(model.predict(sim)[0])
        st.success(f"📈 Revenue prédit : **{format_money(pred_sim)}**")

        st.info("💡 Astuce portfolio : explique que ce modèle est une **démo** (meilleur avec +features: spend/leads/orders, et +données).")

# ============================================================
# TAB 4 — Clustering
# ============================================================
with tab4:
    st.subheader("🧩 Segmentation des campagnes (KMeans)")

    # Agrégation par campagne (sinon trop bruité)
    grp_cols = ["revenue", "clicks", "impressions", "ctr"]
    if "spend" in dff.columns:
        grp_cols += ["spend", "roas", "roi"]

    agg = (
        dff.groupby("campaign", as_index=False)
        .agg({c: "mean" for c in grp_cols})
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(agg) < 6:
        st.warning("Pas assez de campagnes distinctes pour clusteriser (il faut au moins ~6).")
    else:
        k = st.slider("Nombre de clusters (k)", 2, 6, 4)

        features = [c for c in grp_cols if c in agg.columns]
        X = agg[features].values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        agg["cluster"] = km.fit_predict(Xs)

        st.markdown("### 📌 Résumé clusters")
        summary = agg.groupby("cluster", as_index=False).agg({c: "mean" for c in features} | {"campaign": "count"})
        summary = summary.rename(columns={"campaign": "n_campaigns"})
        st.dataframe(summary, use_container_width=True, height=220)

        st.markdown("### 🎨 Visualisation (couleur + taille)")
        x_axis = st.selectbox("Axe X", options=features, index=features.index("revenue") if "revenue" in features else 0)
        y_axis = st.selectbox("Axe Y", options=features, index=features.index("ctr") if "ctr" in features else 1)
        size_axis = st.selectbox("Taille des points", options=features, index=features.index("impressions") if "impressions" in features else 0)

    # ============================
    # Graphique clustering PRO
    # ============================
        
        fig = px.scatter(
            agg,
            x="roi",
            y="revenue",
        
            color="cluster",
            size="revenue",
        
            hover_name="campaign",
        
            color_discrete_map={
                0: "#2ecc71",  # vert
                1: "#3498db",  # bleu
                2: "#f39c12",  # orange
                3: "#e74c3c",  # rouge
                4: "#9b59b6",  # violet
                5: "#1abc9c"   # turquoise
            },
        
            size_max=40
        )
        
        fig.update_layout(
            height=600,
            title="Segmentation des campagnes (ROI vs Revenue)",
            legend_title="Cluster",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        
  # ============================
  # Résumé clusters
  # ============================
        
        st.markdown("### 📊 Résumé clusters")
        
        summary = (
            agg.groupby("cluster")
            .agg({
                "revenue": "mean",
                "roi": "mean",
                "clicks": "mean",
                "impressions": "mean",
                "campaign": "count"
            })
            .rename(columns={"campaign": "nb_campaigns"})
            .reset_index()
        )
        
        st.dataframe(summary, use_container_width=True)
        
        
   # ============================
        # Commentaire automatique PORTFOLIO
        # ============================
        
        st.markdown("### 🧠 Commentaire (portfolio)")
        
        for _, row in summary.iterrows():
        
            cluster = row["cluster"]
            roi = row["roi"]
            revenue = row["revenue"]
        
            if roi > 1:
                st.success(
                    f"🔵 Cluster {cluster} : campagnes très rentables "
                    f"(ROI {roi:.2f}) → scaler en priorité."
                )
        
            elif roi > 0:
                st.info(
                    f"🟢 Cluster {cluster} : campagnes rentables "
                    f"(ROI {roi:.2f}) → optimiser et développer."
                )
        
            elif roi > -0.5:
                st.warning(
                    f"🟠 Cluster {cluster} : campagnes peu performantes "
                    f"(ROI {roi:.2f}) → optimisation recommandée."
                )
        
            else:
                st.error(
                    f"🔴 Cluster {cluster} : campagnes non rentables "
                    f"(ROI {roi:.2f}) → à revoir ou arrêter."
                )
        
        
        # ============================
        # Commentaire global portfolio
        # ============================
        
        best_cluster = summary.sort_values("roi", ascending=False).iloc[0]
        worst_cluster = summary.sort_values("roi").iloc[0]
        
        st.markdown("---")
        
        st.markdown("### 🎯 Analyse stratégique")
        
        st.write(
            f"Le cluster le plus performant est le Cluster {best_cluster['cluster']} "
            f"avec ROI moyen de {best_cluster['roi']:.2f}."
        )
        
        st.write(
            f"Le cluster le moins performant est le Cluster {worst_cluster['cluster']} "
            f"avec ROI moyen de {worst_cluster['roi']:.2f}."
        )
        
        st.write(
            "Recommandation : réallouer le budget vers les clusters les plus performants "
            "et optimiser ou arrêter les campagnes non rentables."
        )  
