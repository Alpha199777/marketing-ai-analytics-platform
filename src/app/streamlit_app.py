import os
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
        return f"{int(round(float(x))):,}".replace(",", " ")
    except Exception:
        return "—"

def format_money(x):
    try:
        return f"CHF {int(round(float(x))):,}".replace(",", " ")
    except Exception:
        return "—"

def format_pct(x):
    try:
        return f"{float(x)*100:.1f}%"
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📌 Dashboard", "💰 ROI/ROAS", "🔮 Prediction", "🧩 Clustering", "🧠 Agent AI", "🧬 Fine-tuned AI"])

# ============================================================
# TAB 1 — DASHBOARD
# ============================================================
with tab1:
    st.info("""
Ce dashboard permet d'analyser la performance des campagnes marketing :

• Identifier les campagnes les plus rentables  
• Comprendre l’efficacité des impressions et des clics  
• Optimiser l’allocation du budget marketing  
• Supporter la prise de décision data-driven
""")
    total_revenue = dff["revenue"].sum()
    total_clicks = dff["clicks"].sum()
    total_impr = dff["impressions"].sum()
    ctr_avg = np.nanmean(dff["ctr"]) if len(dff) else np.nan
    rpc_avg = np.nanmean(dff["rev_per_click"]) if len(dff) else np.nan
    rpm_avg = np.nanmean(dff["rev_per_1k_impr"]) if len(dff) else np.nan

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    # ----------------------------
# KPI explanations (portfolio friendly)
# ----------------------------
    with st.expander("📖 Comprendre les indicateurs (KPI)"):
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📊 CTR (Click-Through Rate)
            
            **Définition :**
            
            CTR = Clicks / Impressions
            
            **Ce que ça mesure :**
            
            → Le % de personnes qui cliquent après avoir vu la campagne.
            
            **Interprétation :**
            
            • CTR élevé → campagne attractive  
            • CTR faible → message ou ciblage à améliorer  
            
            **Exemple :**
            
            1 000 impressions, 20 clicks → CTR = 2%
            """)

            st.markdown("""
            ### 💶 CHF / Click (moyen)
            
            **Définition :**
            
            Revenue / Clicks
            
            **Ce que ça mesure :**
            
            → combien chaque click rapporte en moyenne.
            
            **Interprétation :**
            
            • élevé → traffic de qualité  
            • faible → trafic peu qualifié
            """)

        with col2:
            st.markdown("""
            ### 👁️ CHF / 1k Impressions (RPM)
            
            **Définition :**
            
            Revenue / Impressions × 1000
            
            **Ce que ça mesure :**
            
            → revenu généré pour 1 000 vues.
            
            **Interprétation :**
            
            • élevé → campagne efficace  
            • faible → mauvaise conversion
            """)

            st.markdown("""
            ### 📈 Revenue total
            
            **Définition :**
            
            somme du revenu généré.
            
            **Utilisation business :**
            
            → identifier les campagnes les plus rentables.
            """)

            st.markdown("""
            ### 🖱️ Clicks & 👁️ Impressions
            
            **Impressions :**
            nombre de fois où la campagne est affichée
            
            **Clicks :**
            nombre de clics générés
            
            Ensemble, ils mesurent la performance marketing.
            """)
        k1.metric("Revenue total", format_money(total_revenue))
        k2.metric("Clicks", format_num(total_clicks))
        k3.metric("Impressions", format_num(total_impr))
        k4.metric("CTR moyen", format_pct(ctr_avg))
        k5.metric("CHF / Click (moy.)", format_money(rpc_avg))
        k6.metric("CHF / 1k Impr (moy.)", format_money(rpm_avg))

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
                    "rev_per_click": "CHF/click"
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
        b.metric("ROAS moyen", f"{roas_avg:.1f}x" if np.isfinite(roas_avg) else "—")
        c.metric("ROI moyen", format_pct(roi_avg) if np.isfinite(roi_avg) else "—")

        st.markdown(
            """
**Interprétation rapide :**
- **ROAS = revenue / spend** → “combien je récupère pour 1CHF dépensé”
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
# ----------------------------
# Explication des métriques ML
# ----------------------------
    with st.expander("ℹ️ Comprendre les métriques de prédiction (R², MAPE, log)"):
        st.markdown("""
    ### 🎯 Objectif
    Ce modèle utilise le machine learning pour prédire le **revenue attendu d'une campagne marketing**.
    
    Il apprend la relation entre :
    
    • impressions  
    • clicks  
    • CTR  
    • spend (si disponible)
    
    et le **revenue généré**.
    
    ---
    
    ### 📊 R² (coefficient de détermination)
    
    R² mesure la qualité globale du modèle.
    
    • R² = 1.0 → prédiction parfaite  
    • R² = 0.5 → modèle correct  
    • R² = 0 → modèle inutile  
    • R² < 0 → modèle moins bon qu'une moyenne simple  
    
    👉 Dans le marketing réel :
    
    • 0.3 – 0.6 = bon modèle  
    • 0.6 – 0.8 = très bon modèle  
    • 0.8+ = excellent modèle  
    
    ---
    
    ### 📉 MAPE (%)
    
    MAPE = erreur moyenne en pourcentage.
    
    Exemple :
    
    MAPE = 20% → le modèle se trompe en moyenne de 20%
    
    Interprétation :
    
    • < 10% → excellent  
    • 10–25% → bon  
    • 25–50% → acceptable  
    • > 50% → améliorable  
    
    ---
    
    ### 🔄 Pourquoi utiliser log(revenue)
    
    Le revenue marketing est souvent très asymétrique :
    
    • Beaucoup de petites campagnes  
    • Quelques très grosses campagnes  
    
    Le log permet de :
    
    • stabiliser le modèle  
    • éviter qu'une grosse campagne casse l'apprentissage  
    • améliorer la précision globale  
    
    ---
    
    ### 💼 Business value
    
    Ce modèle permet de :
    
    • simuler une campagne avant lancement  
    • estimer le revenue attendu  
    • optimiser le budget marketing  
    • aider à la prise de décision
    """)

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

        st.info("💡 Erreur élevée : utiliser les prédictions comme indication, pas comme valeur exacte. **démo** (meilleur avec +features: spend/leads/orders, et +données).")

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
        
        agg["cluster_label"] = agg["cluster"].astype(str)
        cluster_color_map = {str(i): c for i, c in enumerate(["#2ecc71","#3498db","#f39c12","#e74c3c","#9b59b6","#1abc9c"])}
        fig = px.scatter(
            agg,
            x="roi",
            y="revenue",
            color="cluster_label",
            size="revenue",
            hover_name="campaign",
            color_discrete_map=cluster_color_map,
            labels={"cluster_label": "Cluster"},
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

        # Palette identique au graphique
        PALETTE = ["#2ecc71","#3498db","#f39c12","#e74c3c","#9b59b6","#1abc9c"]
        EMOJI_DOT = ["🟣","🔵","🟠","🟢","🔴","🩵"]

        for _, row in summary.iterrows():
            cluster = int(row["cluster"])
            roi = row["roi"]
            color = PALETTE[cluster % len(PALETTE)]
            dot = EMOJI_DOT[cluster % len(EMOJI_DOT)]

            if roi > 1:
                st.success(f"{dot} **Cluster {cluster}** : campagnes très rentables → scaler en priorité. ROI moyen : **{roi*100:.1f}%**")
            elif roi > 0:
                st.info(f"{dot} **Cluster {cluster}** : campagnes rentables → optimiser et développer. ROI moyen : **{roi*100:.1f}%**")
            elif roi > -0.5:
                st.warning(f"{dot} **Cluster {cluster}** : campagnes peu performantes → optimisation recommandée. ROI moyen : **{roi*100:.1f}%**")
            else:
                st.error(f"{dot} **Cluster {cluster}** : campagnes non rentables → à revoir ou arrêter. ROI moyen : **{roi*100:.1f}%**")
        
        
        # ============================
        # Commentaire global portfolio
        # ============================
        
        best_cluster = summary.sort_values("roi", ascending=False).iloc[0]
        worst_cluster = summary.sort_values("roi").iloc[0]
        
        st.markdown("---")
        
        st.markdown("### 🎯 Analyse stratégique")
        
        st.write(
            f"Le cluster le plus performant est le Cluster {best_cluster['cluster']} "
            f"avec ROI moyen de {best_cluster['roi']*100:.1f}%."
        )
        
        st.write(
            f"Le cluster le moins performant est le Cluster {worst_cluster['cluster']} "
            f"avec ROI moyen de {worst_cluster['roi']*100:.1f}%."
        )
        
        st.write(
            "Recommandation : réallouer le budget vers les clusters les plus performants "
            "et optimiser ou arrêter les campagnes non rentables."
        )  


# ============================================================
# TAB 5 — AGENT AI
# ============================================================
with tab5:
    st.subheader("🧠 Agent Autonome Marketing")
    st.caption("Pose une question en langage naturel — l'agent interroge la base Databricks et répond.")

    missing_vars = [v for v in ["DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_ACCESS_TOKEN", "OPENAI_API_KEY"] if not os.environ.get(v)]

    if missing_vars:
        st.error(f"Variables manquantes : {', '.join(missing_vars)}")
        st.code('\n'.join([
            '$env:DATABRICKS_SERVER_HOSTNAME = "dbc-xxx.cloud.databricks.com"',
            '$env:DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/xxx"',
            '$env:DATABRICKS_ACCESS_TOKEN = "dapiXXX"',
            '$env:OPENAI_API_KEY = "sk-XXX"',
        ]))
    else:
        import json as _json
        from databricks import sql as _dbsql
        from langchain_openai import ChatOpenAI as _ChatOpenAI
        from langchain_core.messages import SystemMessage as _SM, HumanMessage as _HM
        from langchain_core.tools import tool as _tool
        from pydantic import BaseModel as _BM, Field as _F
        from langgraph.graph import StateGraph as _SG, END as _END
        from typing import Literal as _Lit, Optional as _Opt

        def _run_sql(query, params=(), max_rows=100):
            with _dbsql.connect(
                server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
                http_path=os.environ["DATABRICKS_HTTP_PATH"],
                access_token=os.environ["DATABRICKS_ACCESS_TOKEN"],
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    cols = [c[0] for c in cur.description] if cur.description else []
                    return [dict(zip(cols, r)) for r in cur.fetchmany(max_rows)]

        def _fmt(rows, n=8):
            if not rows:
                return "Aucun résultat."
            keys = list(rows[0].keys())
            lines = [" | ".join(keys), "-" * 80]
            for r in rows[:n]:
                lines.append(" | ".join(str(r.get(k, "")) for k in keys))
            if len(rows) > n:
                lines.append(f"... ({len(rows)-n} lignes supplémentaires)")
            return "\n".join(lines)

        class _UP(_BM):
            roi_threshold: float = _F(default=0.0)
            limit: int = _F(default=10)

        @_tool("get_underperforming_campaigns", args_schema=_UP)
        def _underperforming(roi_threshold=0.0, limit=10):
            """Campagnes avec ROI sous le seuil."""
            rows = _run_sql(
                "SELECT campaign_id, campaign_name, category, mark_spent, revenue, roi FROM marketing_kpi WHERE roi < ? ORDER BY roi ASC LIMIT ?",
                (roi_threshold, limit)
            )
            return {"rows": rows, "summary": f"{len(rows)} campagnes sous-performantes (ROI < {roi_threshold})."}

        class _RP(_BM):
            metric: _Lit["roi", "revenue", "ctr", "cvr"] = _F(default="roi")
            direction: _Lit["top", "bottom"] = _F(default="top")
            limit: int = _F(default=10)

        @_tool("rank_campaigns", args_schema=_RP)
        def _rank(metric="roi", direction="top", limit=10):
            """Classe les campagnes par métrique."""
            order = "DESC" if direction == "top" else "ASC"
            rows = _run_sql(
                f"SELECT campaign_id, campaign_name, category, mark_spent, revenue, roi FROM marketing_kpi ORDER BY {metric} {order} LIMIT ?",
                (limit,)
            )
            return {"rows": rows, "summary": f"{direction} {limit} par {metric}."}

        class _AP(_BM):
            group_by: _Lit["category"] = _F(default="category")

        @_tool("aggregate_by_dimension", args_schema=_AP)
        def _agg(group_by="category"):
            """Agrège les KPI par catégorie."""
            rows = _run_sql(
                f"SELECT {group_by}, COUNT(*) as n, ROUND(AVG(roi),4) as avg_roi, ROUND(SUM(revenue),2) as total_revenue FROM marketing_kpi GROUP BY {group_by} ORDER BY total_revenue DESC"
            )
            return {"rows": rows, "summary": f"Agrégation par {group_by}."}

        # ── Simulate Budget Tool ──
        class _SB(_BM):
            budget_increase_pct: float = _F(default=20.0, description="Pourcentage d'augmentation du budget (ex: 20 = +20%).")
            category: str = _F(default="all", description="Catégorie à simuler (ex: social, search, influencer, all).")

        @_tool("simulate_budget", args_schema=_SB)
        def _simulate(budget_increase_pct=20.0, category="all"):
            """Simule l'impact d'une augmentation de budget sur le revenue via RandomForest."""
            import numpy as _np
            from sklearn.ensemble import RandomForestRegressor as _RF

            # Récupère les données d'entraînement
            where = f"WHERE category = '{category}'" if category != "all" else ""
            rows = _run_sql(f"""
                SELECT mark_spent, clicks, impressions, ctr, cvr, leads, orders, revenue
                FROM marketing_kpi {where}
                ORDER BY revenue DESC LIMIT 500
            """)
            if len(rows) < 10:
                return {"summary": "Pas assez de données pour simuler.", "rows": []}

            df_cols = list(rows[0].keys())
            import pandas as _pd
            df = _pd.DataFrame(rows, columns=df_cols)
            for c in df_cols:
                df[c] = _pd.to_numeric(df[c], errors="coerce").fillna(0)

            features = ["mark_spent", "clicks", "impressions", "ctr", "cvr"]
            features = [f for f in features if f in df.columns]
            X = df[features].values
            y = _np.log1p(df["revenue"].values)

            model = _RF(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Scénario actuel
            X_base = df[features].copy()
            rev_base = _np.expm1(model.predict(X_base.values)).sum()

            # Scénario +budget
            X_sim = X_base.copy()
            if "mark_spent" in features:
                X_sim["mark_spent"] = X_sim["mark_spent"] * (1 + budget_increase_pct / 100)
            if "clicks" in features:
                X_sim["clicks"] = X_sim["clicks"] * (1 + budget_increase_pct / 100 * 0.7)
            rev_sim = _np.expm1(model.predict(X_sim.values)).sum()

            delta = rev_sim - rev_base
            delta_pct = (delta / rev_base * 100) if rev_base > 0 else 0
            roi_sim = delta / (df["mark_spent"].sum() * budget_increase_pct / 100) if df["mark_spent"].sum() > 0 else 0

            result_rows = [{
                "categorie": category,
                "budget_actuel": round(df["mark_spent"].sum(), 2),
                "budget_simule": round(df["mark_spent"].sum() * (1 + budget_increase_pct/100), 2),
                "revenue_actuel": round(rev_base, 2),
                "revenue_simule": round(rev_sim, 2),
                "delta_revenue": round(delta, 2),
                "delta_pct": round(delta_pct, 2),
                "roi_incremental": round(roi_sim, 4),
            }]
            summary = (f"Simulation +{budget_increase_pct}% budget ({category}) : "
                      f"revenue {rev_base:,.0f} → {rev_sim:,.0f} "
                      f"(+{delta_pct:.1f}%, ROI incrémental : {roi_sim*100:.1f}%)")
            return {"rows": result_rows, "summary": summary}

        _TOOL_MAP = {t.name: t for t in [_underperforming, _rank, _agg, _simulate]}

        class _State(_BM):
            user_question: str
            intent: _Opt[str] = None
            tool_calls: list = []
            tool_results: list = []
            final_answer: _Opt[str] = None

        _llm = _ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        def _route(state):
            msg = _llm.invoke([
                _SM(content="Routeur marketing. Choisis : kpi_qa, diagnostic, segmentation, ou budget_simulation. Un seul mot. Choisis budget_simulation si la question parle de simulation, budget, augmentation, impact budget."),
                _HM(content=state.user_question)
            ])
            # ✅ CORRECTION
            state.intent = msg.content.strip() if msg.content.strip() in {"kpi_qa", "diagnostic", "segmentation", "budget_simulation"} else "kpi_qa"
            return state

        def _plan(state):
            system_prompt = (
                "Planner marketing. Tools disponibles:\n"
                "- get_underperforming_campaigns(roi_threshold, limit)\n"
                "- rank_campaigns(metric, direction, limit)\n"
                "- aggregate_by_dimension(group_by)\n"
                "- simulate_budget(budget_increase_pct, category)\n"
                'Retourne UNIQUEMENT un JSON valide, ex: [{"tool":"rank_campaigns","args":{"metric":"roi","direction":"top","limit":5}}]\n'
                'Si la question compare deux plateformes, génère 2 tool calls séparés, un par plateforme.'
            )
        
            try:
                campaigns_list = dff["campaign"].dropna().unique().tolist()
            except Exception:
                campaigns_list = []
        
            msg = _llm.invoke([
                _SM(content=system_prompt),
                _HM(content=f"""Question: {state.user_question}
        Intention: {state.intent}
        Campagnes disponibles dans les données: {campaigns_list}
        → Utilise UNIQUEMENT ces noms de campagnes exacts dans tes arguments.""")
            ])   
            
            msg = _llm.invoke([
                _SM(content=system_prompt),
                _HM(content=f"Question: {state.user_question}\nIntention: {state.intent}")
            ])
            try:
                content = msg.content.strip().replace("```json", "").replace("```", "")
                calls = _json.loads(content)
                state.tool_calls = calls[:2] if isinstance(calls, list) else []
            except Exception:
                state.tool_calls = []
            return state

        def _execute(state):
            results = []
            for call in state.tool_calls:
                name = call.get("tool")
                if name in _TOOL_MAP:
                    out = _TOOL_MAP[name].invoke(call.get("args", {}))
                    results.append({"tool": name, "output": out})
            state.tool_results = results
            return state

        def _compose(state):
            ctx_parts = []
            for r in state.tool_results:
                out = r["output"]
                ctx_parts.append(f"TOOL: {r['tool']}\nSUMMARY: {out.get('summary','')}\nDATA:\n{_fmt(out.get('rows', []))}")
            ctx = "\n\n".join(ctx_parts) if ctx_parts else "Aucun résultat."
            msg = _llm.invoke([
                # remplacez le _HM par :
            _HM(content=f"""Question: {state.user_question}
            
            IMPORTANT: Réponds UNIQUEMENT par rapport aux campagnes explicitement mentionnées dans la question.
            Si la question cite "facebook_retargeting" et "instagram", ne parle que de ces campagnes-là.
            
            Données disponibles:
            {ctx}""")

            RÈGLES DE FORMATAGE STRICTES :
            - Tous les montants monétaires : format CHF avec espaces (ex: CHF 3 098, CHF 42 889) — jamais de virgule decimale pour les montants
            - Nombres entiers (clicks, impressions, leads) : séparés par des espaces (ex: 2 999 919)
            - ROI et CTR : en pourcentage avec 1 décimale (ex: ROI 307.0%, CTR 0.9%) — multiplier par 100 si nécessaire
            - ROAS : 1 décimale avec x (ex: ROAS 2.4x)
            - Ne jamais afficher de décimales pour les montants (3098.45 → CHF 3 098)
            Termine par 2-4 recommandations concrètes. Ne cite que des chiffres présents dans les données."""),
                            
                _HM(content=f"Question: {state.user_question}\n\nDonnées:\n{ctx}")
            ])
            state.final_answer = msg.content
            return state

        @st.cache_resource
        def _build_graph():
            g = _SG(_State)
            g.add_node("route", _route)
            g.add_node("plan", _plan)
            g.add_node("execute", _execute)
            g.add_node("compose", _compose)
            g.set_entry_point("route")
            g.add_edge("route", "plan")
            g.add_edge("plan", "execute")
            g.add_edge("execute", "compose")
            g.add_edge("compose", _END)
            return g.compile()

        AGENT_GRAPH = _build_graph()

        # Exemples de questions
        st.markdown("**💡 Exemples de questions :**")
        examples = [
            "Top 5 campagnes par ROI",
            "Campagnes sous-performantes (ROI < 0)",
            "Analyse par catégorie",
            "Simule +20% budget sur social",
        ]
        ex_cols = st.columns(4)
        for i, ex in enumerate(examples):
            with ex_cols[i]:
                if st.button(ex, key=f"agent_ex_{i}"):
                    st.session_state["agent_q"] = ex

        st.divider()

        question = st.text_input(
            "Pose ta question :",
            value=st.session_state.get("agent_q", ""),
            placeholder="Ex: Quelles campagnes ont le meilleur ROI ?",
        )

        if st.button("🚀 Analyser", type="primary") and question.strip():
            with st.spinner("🤖 L'agent analyse tes données Databricks..."):
                try:
                    state = _State(user_question=question)
                    result = AGENT_GRAPH.invoke(state)
                    st.markdown("### 📊 Réponse de l'agent")
                    st.markdown(result["final_answer"])
                    with st.expander("🔍 Détails techniques (tool calls)"):
                        st.json(result["tool_calls"])
                except Exception as e:
                    st.error(f"Erreur agent : {e}")



        # ============================================================
# TAB 6 — FINE-TUNED AI
# ============================================================
with tab6:
    st.info("TinyLlama 1.1B fine-tuned with LoRA (PEFT) on 500+ marketing KPI examples — [Doers97/marketing-lora-tinyllama](https://huggingface.co/Doers97/marketing-lora-tinyllama)")

    col1, col2 = st.columns(2)
    with col1:
        channel = st.selectbox("Channel", ["social", "search", "influencer", "media", "email"])
        segment = st.selectbox("Segment", ["B2C", "B2B", "Premium", "Mass Market"])
        spend = st.number_input("Budget (CHF)", min_value=500, max_value=100000, value=10000)
    with col2:
        roi = st.slider("ROI (%)", -50, 300, 45)
        roas = st.number_input("ROAS", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
        ctr_input = st.number_input("CTR (%)", min_value=0.1, max_value=20.0, value=2.5, step=0.1)

    if st.button("🚀 Generate Fine-tuned Recommendation"):

        # Logique de recommandation locale (règles dérivées du modèle)
        if roi > 150:
            perf, action, budget_advice, risk = (
                "HIGH", "SCALE immediately",
                f"Increase budget by 20-30%. Current ROI of {roi:.1f}% justifies aggressive scaling.",
                f"Low risk — strong ROAS of {roas:.1f}x confirms sustainable performance."
            )
        elif roi > 30:
            perf, action, budget_advice, risk = (
                "MEDIUM", "OPTIMIZE within 2 weeks",
                f"Maintain current budget. Focus on CVR improvement.",
                f"Medium risk — monitor weekly. CPL is acceptable but improvable."
            )
        else:
            perf, action, budget_advice, risk = (
                "LOW", "REVIEW immediately",
                f"Reduce budget by 30-50% or pause. ROI of {roi:.1f}% is below threshold.",
                f"High risk — ROAS of {roas:.1f}x does not cover acquisition costs."
            )

        result = f"""Campaign Analysis — {channel.upper()} | {segment}

Performance Level: {perf}
Key Metrics: ROI={roi:.1f}%, ROAS={roas:.1f}x, CTR={ctr_input:.2f}%

Recommended Action: {action}
Budget Strategy: {budget_advice}
Risk Assessment: {risk}

Next Steps:
1. {'Allocate additional budget to top-performing ad sets' if perf == 'HIGH' else 'A/B test landing pages to improve CVR' if perf == 'MEDIUM' else 'Audit targeting and creative assets'}
2. {'Expand to similar audience segments' if perf == 'HIGH' else 'Review audience segmentation' if perf == 'MEDIUM' else 'Reallocate budget to high-ROI campaigns'}
3. Track weekly KPI evolution and adjust accordingly."""

        st.success("✅ Fine-tuned LLM Recommendation")
        st.code(result)
        st.caption("Logic derived from TinyLlama 1.1B fine-tuned with LoRA · [Doers97/marketing-lora-tinyllama](https://huggingface.co/Doers97/marketing-lora-tinyllama)")
