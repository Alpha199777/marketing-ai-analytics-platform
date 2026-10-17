import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from budget_simulation import simulate_budget_increase, SQL_QUERY as SIM_SQL_QUERY

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Marketing AI Analytics Platform", page_icon="📈", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def norm_cols(df):
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    return df

def map_schema(df):
    df = norm_cols(df)
    rename_map = {"c_date": "date", "campaign_name": "campaign", "mark_spent": "spend"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["campaign"] = df["campaign"].astype(str).str.lower().str.strip()
    return df

@st.cache_data(ttl=300)
def load_csv(path):
    df = pd.read_csv(path)
    return map_schema(df)

def safe_div(a, b):
    a = np.array(a, dtype="float64"); b = np.array(b, dtype="float64")
    out = np.full_like(a, np.nan, dtype="float64"); mask = b != 0
    out[mask] = a[mask] / b[mask]; return out

def format_num(x):
    try: return f"{int(round(float(x))):,}".replace(",", " ")
    except: return "-"

def format_money(x):
    try: return f"CHF {int(round(float(x))):,}".replace(",", " ")
    except: return "-"

def format_pct(x):
    try: return f"{float(x)*100:.1f}%"
    except: return "-"

def ensure_min_columns(df):
    required = {"date", "campaign", "clicks", "impressions", "revenue"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes: {sorted(list(missing))}\n\nColonnes dispo: {list(df.columns)}")
        st.stop()

def compute_metrics(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    for c in ["clicks", "impressions", "revenue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "spend" in df.columns:
        df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0)
    if "leads" in df.columns:
        df["leads"] = pd.to_numeric(df["leads"], errors="coerce").fillna(0)
    if "orders" in df.columns:
        df["orders"] = pd.to_numeric(df["orders"], errors="coerce").fillna(0)
    df["ctr"] = safe_div(df["clicks"], df["impressions"])
    df["rev_per_click"] = safe_div(df["revenue"], df["clicks"])
    df["rev_per_1k_impr"] = safe_div(df["revenue"] * 1000, df["impressions"])
    if "orders" in df.columns:
        df["cvr"] = safe_div(df["orders"], df["clicks"])
    if "spend" in df.columns and "leads" in df.columns:
        df["cpl"] = safe_div(df["spend"], df["leads"])
    if "spend" in df.columns:
        df["roas"] = safe_div(df["revenue"], df["spend"])
        df["roi"] = safe_div(df["revenue"] - df["spend"], df["spend"])
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
    df = load_csv("data/sample_marketing.csv")
else:
    if uploaded_file is None:
        st.info("Upload un CSV pour continuer."); st.stop()
    df = map_schema(pd.read_csv(uploaded_file))

ensure_min_columns(df)
df = compute_metrics(df)

# ----------------------------
# Filters
# ----------------------------
campaigns = sorted(df["campaign"].dropna().astype(str).unique().tolist())
min_date = df["date"].min().date(); max_date = df["date"].max().date()

with st.sidebar:
    st.divider(); st.subheader("🔎 Filtres")
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
# TAB 1 - DASHBOARD
# ============================================================
with tab1:
    st.info("""
Ce dashboard permet d'analyser la performance des campagnes marketing :

• Identifier les campagnes les plus rentables  
• Comprendre l'efficacité des impressions et des clics  
• Optimiser l'allocation du budget marketing  
• Supporter la prise de décision data-driven
""")
    total_revenue = dff["revenue"].sum()
    total_clicks = dff["clicks"].sum()
    total_impr = dff["impressions"].sum()
    ctr_avg = np.nanmean(dff["ctr"]) if len(dff) else np.nan
    rpc_avg = np.nanmean(dff["rev_per_click"]) if len(dff) else np.nan
    rpm_avg = np.nanmean(dff["rev_per_1k_impr"]) if len(dff) else np.nan

    # FIX 1: KPI metrics OUTSIDE the expander
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Revenue total", format_money(total_revenue))
    k2.metric("Clicks", format_num(total_clicks))
    k3.metric("Impressions", format_num(total_impr))
    k4.metric("CTR moyen", format_pct(ctr_avg))
    k5.metric("CHF / Click (moy.)", format_money(rpc_avg))
    k6.metric("CHF / 1k Impr (moy.)", format_money(rpm_avg))

    with st.expander("📖 Comprendre les indicateurs (KPI)"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
### 📊 CTR (Click-Through Rate)
**Définition :** CTR = Clicks / Impressions  
**Ce que ça mesure :** Le % de personnes qui cliquent après avoir vu la campagne.  
**Interprétation :**  
• CTR élevé → campagne attractive  
• CTR faible → message ou ciblage à améliorer  
**Exemple :** 1 000 impressions, 20 clicks → CTR = 2%
""")
            st.markdown("""
### 💶 CHF / Click (moyen)
**Définition :** Revenue / Clicks  
**Ce que ça mesure :** combien chaque click rapporte en moyenne.
""")
        with col2:
            st.markdown("""
### 👁️ CHF / 1k Impressions (RPM)
**Définition :** Revenue / Impressions x 1000  
**Ce que ça mesure :** revenu généré pour 1 000 vues.
""")
            st.markdown("""
### 📈 Revenue total
**Définition :** somme du revenu généré.  
**Utilisation business :** identifier les campagnes les plus rentables.
""")
            st.markdown("""
### 🖱️ Clicks & 👁️ Impressions
**Impressions :** nombre de fois où la campagne est affichée  
**Clicks :** nombre de clics générés
""")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("""
### 🛒 CVR (Conversion Rate)
**Définition :** CVR = Commandes / Clicks  
**Ce que ça mesure :** le % de visiteurs qui achètent après avoir cliqué.  
**Interprétation :**  
• CVR élevé → l'offre et la page d'arrivée convainquent  
• CVR faible → les gens viennent mais n'achètent pas (prix, page, ciblage)  
**Exemple :** 1 000 clicks, 30 commandes → CVR = 3%
""")
        with c4:
            st.markdown("""
### 🎯 CPL (Coût par lead)
**Définition :** CPL = Budget dépensé / Leads  
**Ce que ça mesure :** combien coûte un contact intéressé (inscription, demande de devis…).  
**Interprétation :**  
• CPL bas → on obtient des prospects à bon prix  
• CPL élevé → la campagne coûte cher pour ce qu'elle rapporte en contacts  
**Exemple :** 5 000 CHF dépensés, 100 leads → CPL = 50 CHF
""")
        st.markdown("""
---
### 🧭 Pourquoi ces indicateurs ?
Une campagne fait avancer un client étape par étape :  
**affichage → clic → contact (lead) → commande → revenu**.  
Chaque indicateur mesure **une étape**, ce qui permet de savoir **où** une campagne perd de la valeur :

| Étape | Indicateur | Question à laquelle il répond |
|---|---|---|
| Affichage → clic | **CTR** | Le message attire-t-il l'attention ? |
| Clic → commande | **CVR** | Les visiteurs achètent-ils ? |
| Coût d'un contact | **CPL** | Combien paie-t-on pour un prospect ? |
| Résultat final | **ROAS / ROI** | La campagne rapporte-t-elle plus qu'elle ne coûte ? |

Une campagne peut avoir un très bon CTR et un mauvais ROI : elle attire, mais ne convertit pas. Regarder un seul  
indicateur ne suffit donc pas.
""")

    st.divider()

    # FIX 1 cont.: Charts OUTSIDE the expander
    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("📈 Tendance (au choix)")
        ts = (
            dff.groupby(pd.Grouper(key="date", freq="D"))
            .agg(revenue=("revenue", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"), spend=("spend", "sum"))
            .reset_index().sort_values("date")
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
            .agg(revenue=("revenue", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        )
        by_campaign["ctr"] = safe_div(by_campaign["clicks"], by_campaign["impressions"])
        by_campaign["rev_per_click"] = safe_div(by_campaign["revenue"], by_campaign["clicks"])
        sort_by = st.selectbox("Trier par", ["revenue", "clicks", "impressions", "ctr", "rev_per_click"], index=0)
        top = by_campaign.sort_values(sort_by, ascending=False).head(top_n).copy()
        st.dataframe(top.rename(columns={"rev_per_click": "CHF/click"}), use_container_width=True, height=220)
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
        st.download_button("⬇️ Télécharger les données filtrées (CSV)", data=table.to_csv(index=False).encode("utf-8"), file_name="marketing_filtered.csv", mime="text/csv")

# ============================================================
# TAB 2 - ROI/ROAS
# ============================================================
with tab2:
    st.subheader("💰 ROI / ROAS")
    if "spend" not in dff.columns:
        st.warning("Aucune colonne de coût/spend détectée. ROI/ROAS indisponibles.")
        st.info("👉 Ajoute `mark_spent` dans ton CSV (ou renomme en `spend`) pour activer cette page.")
    else:
        spend_total = dff["spend"].sum()
        roas_avg = np.nanmean(dff["roas"]) if len(dff) else np.nan
        roi_avg = np.nanmean(dff["roi"]) if len(dff) else np.nan

        a, b, c = st.columns(3)
        a.metric("Spend total", format_money(spend_total))
        b.metric("ROAS moyen", f"{roas_avg:.1f}x" if np.isfinite(roas_avg) else "-")
        c.metric("ROI moyen", format_pct(roi_avg) if np.isfinite(roi_avg) else "-")

        st.markdown("""
**Interprétation rapide :**
- **ROAS = revenue / spend** "combien je récupère pour 1CHF dépensé"
- **ROI = (revenue - spend) / spend** "mon gain net relatif"

**Exemple :** je dépense **1 000 CHF** et la campagne rapporte **1 500 CHF**
→ ROAS = 1 500 / 1 000 = **1,5x** (je récupère 1,50 CHF pour 1 CHF investi)
→ ROI = (1 500 − 1 000) / 1 000 = **+50 %** (je gagne 50 % de ce que j'ai investi)

**Repères :** ROAS < 1 (ou ROI < 0) → la campagne **perd de l'argent** ; ROAS = 1 (ROI = 0) → elle rembourse
juste son coût ; ROAS > 1 (ROI > 0) → elle est **rentable**.
        """)

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
            st.dataframe(by_campaign.sort_values("roas", ascending=False).head(10), use_container_width=True, height=260)
        with right:
            st.markdown("### ⚠️ Worst ROI")
            st.dataframe(by_campaign.sort_values("roi", ascending=True).head(10), use_container_width=True, height=260)

# ============================================================
# TAB 3 - Prediction
# ============================================================
N_FOLDS = 5


def reg_metrics(y_true, y_pred) -> dict:
    """R², erreur médiane et MAPE (sur les lignes à revenu > 0)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    pos = y_true > 0
    ape = np.abs(y_true[pos] - y_pred[pos]) / y_true[pos]
    return {
        "R²": r2_score(y_true, y_pred),
        "Erreur médiane": float(np.median(ape)) if pos.any() else np.nan,
        "MAPE": float(np.mean(ape)) if pos.any() else np.nan,
    }


@st.cache_data(show_spinner="Validation croisée en cours…")
def evaluate_explain_models(data: pd.DataFrame, features: tuple, has_orders: bool):
    """Compare en validation croisée le RandomForest et la référence « commandes x panier moyen »."""
    X, y = data[list(features)].values, data["revenue"].values
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rows = [{"Modèle": "RandomForest (" + ", ".join(features) + ")", **reg_metrics(y, cross_val_predict(rf, X, y, cv=cv))}]
    if has_orders:
        base = LinearRegression(fit_intercept=False)
        p = cross_val_predict(base, data[["orders"]].values, y, cv=cv)
        rows.append({"Modèle": "Référence : commandes x panier moyen", **reg_metrics(y, p)})
    importances = pd.Series(rf.fit(X, y).feature_importances_, index=list(features)).sort_values()
    return pd.DataFrame(rows), importances


def sim_design(spend, category, categories) -> pd.DataFrame:
    X = pd.DataFrame({"spend": np.asarray(spend, dtype="float64")})
    for c in categories:
        X["cat_" + c] = (np.asarray(category) == c).astype(float)
    return X


@st.cache_data(show_spinner="Entraînement du modèle de simulation…")
def train_simulation_model(data: pd.DataFrame):
    """Modèle avant lancement : n'utilise que le budget et le canal."""
    categories = tuple(sorted(data["category"].astype(str).unique()))
    X = sim_design(data["spend"], data["category"].astype(str), categories)
    y = data["revenue"].values
    model = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1)
    p = cross_val_predict(model, X, y, cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=42))
    ratio = y[p > 0] / p[p > 0]
    q_low, q_high = np.quantile(ratio, [0.1, 0.9])
    model.fit(X, y)
    ranges = data.groupby(data["category"].astype(str))["spend"].agg(["min", "max"])
    return model, categories, reg_metrics(y, p), float(q_low), float(q_high), ranges


with tab3:
    st.subheader("🔮 Revenu : expliquer, puis simuler")

    with st.expander("ℹ️ Comprendre cette page : objectif, métriques et méthode"):
        st.markdown("""
### 🎯 Objectif
Cette page répond à deux questions différentes, avec deux modèles différents :

1. **Expliquer** : « pourquoi telle campagne terminée a-t-elle rapporté ce revenu ? »  
   → on utilise tout ce qu'on sait de la campagne (clics, leads, commandes…).  
2. **Simuler** : « si je lance une campagne avec tel budget sur tel canal, combien puis-je espérer ? »  
   → on n'utilise que ce qu'on connaît **avant** de lancer : le budget et le canal.

---

### 📊 R² (coefficient de détermination)
**Ce que ça mesure :** quelle part des différences de revenu entre campagnes le modèle arrive à expliquer.  
• R² = 1.0 → prédiction parfaite | R² = 0 → pas mieux que de toujours prédire la moyenne | R² < 0 → pire que la moyenne  
**Repères en marketing :** 0.3–0.6 = bon | 0.6–0.8 = très bon | 0.8+ = excellent (ou suspect : à vérifier !)

---

### 📉 Erreur médiane et MAPE (%)
**MAPE** = erreur moyenne en pourcentage du revenu réel. < 10% excellent | 10–25% bon | > 50% améliorable  
**Erreur médiane** = la moitié des prédictions se trompent de moins que ce pourcentage.  
On affiche les deux car le MAPE est faussé par les petites campagnes : se tromper de 500 CHF sur une campagne  
qui a rapporté 100 CHF compte comme une erreur de 500 %.

---

### 🧪 Validation croisée (5 parts) : pourquoi les scores sont « honnêtes »
On découpe les données en 5 parts. Le modèle apprend sur 4 parts et est noté sur la 5e, qu'il n'a jamais vue.  
On recommence 5 fois pour que chaque ligne soit notée une fois.  
→ C'est comme un examen avec des questions que l'élève n'a pas vues en cours : le score reflète ce que le modèle  
fera sur de nouvelles campagnes, pas sa capacité à réciter le passé.

---

### 🚫 Pourquoi le ROI n'est pas utilisé pour prédire le revenu
Le ROI se calcule **à partir du revenu** : ROI = (revenu − budget) / budget.  
Donner le ROI au modèle, c'est lui donner la réponse : il aurait l'air excellent, mais il tricherait.  
C'est une erreur classique appelée **fuite de données** (data leakage).

---

### ⚖️ Pourquoi comparer avec une « référence » simple
Un modèle de machine learning n'a d'intérêt que s'il fait **mieux qu'une règle de bon sens**.  
Ici la règle est : **revenu ≈ nombre de commandes × panier moyen**. Si elle fait aussi bien que le RandomForest,  
le modèle n'apporte rien de plus : c'est un vrai résultat, pas un échec.

---

### 🔮 Comment lire la simulation
• **Revenu prévu** : l'estimation centrale.  
• **Fourchette** : 8 campagnes passées sur 10 se situaient dans cet intervalle par rapport à la prévision.  
C'est l'information la plus utile : elle montre l'incertitude réelle.  
• **Courbe par canal** : si elle s'aplatit quand le budget augmente, chaque franc supplémentaire rapporte moins  
(**rendements décroissants**).  
• **Hors plage observée** : budget jamais vu dans les données pour ce canal → la prévision n'est pas fiable.

---

### 💼 Business value
• comparer les canaux avant d'investir  
• estimer un ordre de grandeur de revenu et le risque associé (fourchette)  
• repérer le budget au-delà duquel investir davantage rapporte peu
""")

    if len(dff) < 30:
        st.warning("Il faut au moins ~30 lignes pour une validation croisée fiable : élargis les filtres.")
    else:
        # ---------- 1. Expliquer ----------
        st.markdown("### 1. Expliquer le revenu d'une campagne terminée")
        explain_feats = tuple(f for f in ["impressions", "clicks", "leads", "orders", "spend", "ctr", "cvr", "cpl"]
                              if f in dff.columns)
        data_ml = dff[list(explain_feats) + ["revenue"]].replace([np.inf, -np.inf], np.nan).fillna(0)
        results, importances = evaluate_explain_models(data_ml, explain_feats, "orders" in dff.columns)

        st.dataframe(
            results.style.format({"R²": "{:.2f}", "Erreur médiane": "{:.0%}", "MAPE": "{:.0%}"}),
            use_container_width=True, hide_index=True,
        )
        left, right = st.columns([3, 2])
        with left:
            if len(results) > 1 and results.loc[1, "R²"] >= results.loc[0, "R²"]:
                st.info("La règle simple **commandes x panier moyen** fait au moins aussi bien que le RandomForest : "
                        "le revenu se déduit des commandes. Ce modèle explique le passé, il ne peut pas simuler une "
                        "campagne future (on ne connaît pas ses commandes à l'avance).")
            else:
                st.info("Ce modèle utilise des résultats de campagne (clics, commandes…) : il explique le passé "
                        "mais ne peut pas simuler une campagne avant son lancement.")
        with right:
            fig_imp = px.bar(importances, orientation="h", labels={"value": "importance", "index": ""},
                             title="Variables utilisées par le RandomForest")
            fig_imp.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

        # ---------- 2. Simuler ----------
        st.divider()
        st.markdown("### 2. Simuler une campagne avant son lancement")
        if not {"spend", "category"}.issubset(dff.columns):
            st.warning("La simulation demande les colonnes **spend** (ou mark_spent) et **category** dans les données.")
        else:
            st.caption("Le modèle ne reçoit que ce que l'on décide avant le lancement : le **budget** et le **canal**.")
            sim_data = dff[["spend", "category", "revenue"]].dropna()
            model, categories, sim_m, q_low, q_high, ranges = train_simulation_model(sim_data)

            a, b = st.columns(2)
            a.metric("R² (validation croisée)", f"{sim_m['R²']:.2f}")
            b.metric("Erreur médiane", format_pct(sim_m["Erreur médiane"]))

            c1, c2 = st.columns(2)
            with c1:
                sim_cat = st.selectbox("Canal", options=list(categories))
            with c2:
                sim_budget = st.number_input("Budget (CHF)", min_value=0.0, step=1000.0,
                                             value=float(round(np.nanmedian(sim_data["spend"]), -3)))

            pred = float(model.predict(sim_design([sim_budget], [sim_cat], categories))[0])
            roi_pred = (pred - sim_budget) / sim_budget if sim_budget > 0 else np.nan
            r1, r2c, r3 = st.columns(3)
            r1.metric("Revenu prévu", format_money(pred))
            r2c.metric("Fourchette (80 % des cas passés)", f"{format_money(pred * q_low)} – {format_money(pred * q_high)}")
            r3.metric("ROI prévu", f"{roi_pred:+.2f}" if np.isfinite(roi_pred) else "—")

            lo, hi = ranges.loc[sim_cat, "min"], ranges.loc[sim_cat, "max"]
            if not lo <= sim_budget <= hi:
                st.warning(f"Budget hors de la plage observée pour **{sim_cat}** ({format_money(lo)} – {format_money(hi)}) : "
                           "le modèle extrapole, la prévision n'est pas fiable.")

            # Courbe de réponse : revenu prévu selon le budget, par canal
            grid = np.linspace(np.nanpercentile(sim_data["spend"], 5), np.nanpercentile(sim_data["spend"], 95), 40)
            curve = pd.concat([
                pd.DataFrame({"budget": grid, "canal": c,
                              "revenu_prevu": model.predict(sim_design(grid, [c] * len(grid), categories))})
                for c in categories
            ])
            fig_curve = px.line(curve, x="budget", y="revenu_prevu", color="canal",
                                labels={"budget": "Budget (CHF)", "revenu_prevu": "Revenu prévu (CHF)", "canal": "Canal"},
                                title="Revenu prévu selon le budget et le canal")
            fig_curve.add_scatter(x=[sim_budget], y=[pred], mode="markers", marker=dict(size=12, color="black"),
                                  name="Ta simulation")
            fig_curve.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_curve, use_container_width=True)

            st.info("💡 Lis la fourchette plutôt que le chiffre central : budget et canal n'expliquent qu'une partie "
                    "du revenu (création, ciblage et saisonnalité ne sont pas dans les données).")

# ============================================================
# TAB 4 - Clustering
# ============================================================
with tab4:
    st.subheader("🧩 Segmentation des campagnes (KMeans)")
    st.caption("Chaque point est une ligne campagne x date. Les segments sont formés sur le **profil** "
               "(CTR, CVR, CPL, budget) ; le ROI et le revenu servent seulement à les décrire.")

    with st.expander("ℹ️ Comprendre la segmentation : objectif, choix des métriques, lecture"):
        st.markdown("""
### 🎯 Objectif
Regrouper automatiquement les campagnes **qui se ressemblent**, pour décider quoi faire de chaque groupe  
(développer, optimiser, couper) au lieu d'analyser les campagnes une par une.

---

### 🤖 Comment ça marche (KMeans)
On choisit un nombre de groupes **k**. L'algorithme place chaque campagne dans le groupe dont elle est la plus  
proche, en comparant ses indicateurs. Il n'utilise aucune étiquette « bonne » ou « mauvaise » campagne :  
il découvre les groupes tout seul.

---

### 🧭 Pourquoi ces 4 métriques
| Métrique | Ce qu'elle décrit |
|---|---|
| **CTR** | l'attractivité du message |
| **CVR** | la capacité à transformer un visiteur en client |
| **CPL** | le coût d'acquisition d'un prospect |
| **Budget** (en log) | la taille de la campagne |

Elles décrivent **comment** une campagne fonctionne. Le budget est pris en logarithme pour que quelques très  
grosses campagnes n'écrasent pas toutes les autres. Toutes les métriques sont mises à la même échelle avant le  
calcul, sinon le budget (en milliers) compterait plus que le CTR (en %).

### 🚫 Pourquoi pas le ROI ni le revenu
Ce sont les **résultats** que l'on cherche à expliquer. Les utiliser pour former les groupes reviendrait à trier  
les campagnes entre « celles qui ont marché » et « celles qui n'ont pas marché », ce qu'on sait déjà.  
En les gardant de côté, on peut vérifier ensuite que les groupes ont des ROI très différents : c'est la preuve  
que le **profil** d'une campagne suffit à prévoir si elle sera rentable.

---

### 📏 Score de silhouette
Mesure si les groupes sont bien séparés, de −1 à 1 :  
**> 0,5** groupes nets | **0,25–0,5** groupes acceptables, avec des chevauchements | **< 0,25** groupes peu distincts.  
Le meilleur k mathématique n'est pas toujours le plus utile : 2 groupes peuvent être bien séparés mais trop  
grossiers pour décider d'un budget. Utilise le curseur pour comparer.

---

### 👀 Lire les graphiques
• **Gauche (CTR vs CPL)** : où se situe chaque groupe. En bas à droite = attire beaucoup et coûte peu (idéal).  
• **Droite (ROI par segment)** : la boîte contient la moitié des campagnes du groupe, la ligne  
au milieu est le ROI médian. En dessous de la ligne pointillée (0), le groupe perd de l'argent.
""")

    profile_feats = [f for f in ["ctr", "cvr", "cpl"] if f in dff.columns]
    data_c = dff.replace([np.inf, -np.inf], np.nan).copy()
    if "spend" in data_c.columns:
        data_c["log_spend"] = np.log1p(data_c["spend"])
        profile_feats.append("log_spend")
    data_c[profile_feats] = data_c[profile_feats].fillna(0)

    if len(profile_feats) < 2:
        st.warning("Pas assez de variables de profil : il faut au moins spend, leads ou orders en plus des clics.")
    elif len(data_c) < 20:
        st.warning("Pas assez de lignes pour segmenter (il faut au moins ~20) : élargis les filtres.")
    else:
        Xs = StandardScaler().fit_transform(data_c[profile_feats].values)

        sil = {k: silhouette_score(Xs, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs))
               for k in range(2, 7)}
        best_k = max(sil, key=sil.get)
        c1, c2 = st.columns([1, 2])
        with c1:
            k = st.slider("Nombre de segments (k)", 2, 6, 4)
            st.caption("Silhouette par k (plus haut = segments mieux séparés) : "
                       + " · ".join(f"k={kk} : {v:.2f}" for kk, v in sil.items())
                       + f". Meilleur k : **{best_k}**.")

        data_c["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xs)

        desc_cols = [c for c in ["roi", "revenue", "spend", "ctr", "cvr", "cpl"] if c in data_c.columns]
        summary = (data_c.groupby("cluster")
                   .agg(n_lignes=("revenue", "size"), **{c: (c, "median") for c in desc_cols})
                   .reset_index())
        has_roi = "roi" in summary.columns
        summary = summary.sort_values("roi" if has_roi else "revenue", ascending=False).reset_index(drop=True)

        # --- Nom lisible de chaque segment : rang de rentabilité + verdict + trait distinctif ---
        def verdict(roi):
            if roi > 0.5:
                return "très rentable", "🟢", "développer progressivement", st.success
            if roi > 0:
                return "rentable", "🔵", "maintenir et optimiser", st.info
            if roi > -0.5:
                return "peu performant", "🟠", "travailler la conversion ou le ciblage", st.warning
            return "perd de l'argent", "🔴", "couper ou repenser", st.error

        TRAITS = {"ctr": ("CTR élevé", "CTR faible"), "cvr": ("forte conversion", "faible conversion"),
                  "cpl": ("CPL élevé", "CPL faible"), "spend": ("gros budget", "petit budget")}
        overall = data_c[[c for c in TRAITS if c in data_c.columns]].median()

        def trait(row):
            """Caractéristique qui distingue le plus le segment de la médiane de toutes les campagnes."""
            best_name, best_gap = "profil moyen", np.log(1.25)
            for col, (high, low) in TRAITS.items():
                if col in row and col in overall and overall[col] > 0:
                    gap = np.log((row[col] + 1e-9) / overall[col])
                    if abs(gap) > best_gap:
                        best_name, best_gap = (high if gap > 0 else low), abs(gap)
            return best_name

        summary["rang"] = np.arange(1, len(summary) + 1)
        summary["segment"] = [
            f"S{r} · {verdict(row['roi'])[0]} · {trait(row)}" if has_roi else f"S{r} · {trait(row)}"
            for r, (_, row) in zip(summary["rang"], summary.iterrows())
        ]
        label_of = dict(zip(summary["cluster"], summary["segment"]))
        data_c["segment"] = data_c["cluster"].map(label_of)
        order = list(summary["segment"])
        # Du vert (le plus rentable) au rouge (le moins rentable), mêmes couleurs partout
        # (palette contrastée : pas de jaune pâle, invisible sur fond blanc)
        scale = [[0.0, "#1b7837"], [0.35, "#7fbc41"], [0.55, "#e6a700"], [0.75, "#f46d43"], [1.0, "#b2182b"]]
        ramp = px.colors.sample_colorscale(scale, [i / max(len(order) - 1, 1) for i in range(len(order))])
        colors = dict(zip(order, ramp))

        with c2:
            st.markdown("**Profil médian de chaque segment** (classés du plus rentable au moins rentable)")
            table = summary[["segment", "n_lignes"] + desc_cols].rename(columns={
                "segment": "Segment", "n_lignes": "Nb lignes", "roi": "ROI médian", "revenue": "Revenu médian",
                "spend": "Budget médian", "ctr": "CTR", "cvr": "CVR", "cpl": "CPL"})
            fmt = {"ROI médian": "{:+.2f}", "Revenu médian": "{:,.0f}", "Budget médian": "{:,.0f}",
                   "CTR": "{:.2%}", "CVR": "{:.2%}", "CPL": "{:,.0f}"}
            st.dataframe(table.style.format({c: f for c, f in fmt.items() if c in table.columns}),
                         use_container_width=True, hide_index=True)

        st.caption("🎨 **Lecture des couleurs** : S1 est le segment le plus rentable (vert), le dernier le moins "
                   "rentable (rouge). Les couleurs et les noms sont les mêmes dans le tableau, les deux graphiques "
                   "et la lecture ci-dessous.")
        g1, g2 = st.columns(2)
        with g1:
            x_ax, y_ax = profile_feats[0], ("cpl" if "cpl" in profile_feats else profile_feats[1])
            fig = px.scatter(data_c, x=x_ax, y=y_ax, color="segment", hover_name="campaign",
                             category_orders={"segment": order}, color_discrete_map=colors,
                             labels={"ctr": "CTR", "cpl": "CPL (CHF)", "cvr": "CVR", "segment": "Segment"},
                             title=f"Profil des segments : {x_ax.upper()} vs {y_ax.upper()}")
            fig.update_layout(height=470, margin=dict(l=10, r=10, t=40, b=10),
                              legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            if has_roi:
                data_c["rang"] = data_c["segment"].map({s: f"S{i + 1}" for i, s in enumerate(order)})
                fig_roi = px.box(data_c, x="rang", y="roi", color="segment", points="outliers",
                                 category_orders={"segment": order, "rang": [f"S{i + 1}" for i in range(len(order))]},
                                 color_discrete_map=colors,
                                 labels={"rang": "Segment", "roi": "ROI (sous la ligne 0 = perte)"},
                                 title="ROI par segment (non utilisé pour former les segments)")
                fig_roi.add_hline(y=0, line_dash="dash", line_color="grey")
                fig_roi.update_layout(height=470, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_roi, use_container_width=True)

        if has_roi:
            st.markdown("### 🧠 Lecture des segments")
            for _, row in summary.iterrows():
                name, emoji, action, box = verdict(row["roi"])
                details = (f"ROI médian {row['roi']:+.2f}"
                           + (f" · CTR {row['ctr']:.2%}" if "ctr" in row else "")
                           + (" · CPL " + format_money(row["cpl"]) if "cpl" in row else "")
                           + (" · budget médian " + format_money(row["spend"]) if "spend" in row else ""))
                box(f"{emoji} **{row['segment']}** ({row['n_lignes']} lignes) — {details}  \n→ **{action}**.")

            best, worst = summary.iloc[0], summary.iloc[-1]
            st.markdown("### 🎯 Analyse stratégique")
            st.write(f"Le segment le plus rentable (**{best['segment']}**, ROI médian {best['roi']:+.2f}) et le moins "
                     f"rentable (**{worst['segment']}**, ROI médian {worst['roi']:+.2f}) ont des profils différents. "
                     "Le ROI n'ayant pas servi à former les segments, cet écart montre que le profil d'une campagne "
                     "(CTR, conversion, CPL, budget) suffit à repérer celles qui perdent de l'argent.")
            st.write("Recommandation : réallouer le budget des segments à ROI négatif vers les segments rentables, "
                     "en surveillant les rendements décroissants (voir la courbe de l'onglet Prediction).")

# ============================================================
# TAB 5 - AGENT AI
# ============================================================
with tab5:
    st.subheader("🧠 Agent Autonome Marketing")
    st.caption("Pose une question en langage naturel - l'agent interroge la base Databricks et répond.")
    with st.expander("ℹ️ Comment fonctionne l'agent"):
        st.markdown("""
1. **Comprendre** : un modèle de langage (GPT-4o-mini) identifie le type de question  
   (classement, diagnostic, segmentation, simulation de budget).  
2. **Choisir un outil** : il choisit parmi 4 outils prédéfinis (classer les campagnes, trouver les campagnes  
   sous-performantes, agréger par canal, simuler une variation de budget).  
3. **Interroger les données** : l'outil exécute une requête SQL **prédéfinie, qui ne fait que lire** la table  
   `marketing_kpi` de Databricks. Le modèle de langage n'écrit jamais de SQL lui-même : il ne peut ni modifier  
   ni supprimer de données.  
4. **Répondre** : il rédige une réponse business à partir des chiffres obtenus, avec des recommandations.

**Limites :** l'agent ne connaît que la table `marketing_kpi`. La simulation de budget est un ordre de grandeur  
(modèle budget + canal), pas une garantie. Le détail des outils appelés est visible sous chaque réponse.
""")

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
        from math import isfinite as _np_isfinite
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
                    cur.execute(query, params or None)
                    cols = [c[0] for c in cur.description] if cur.description else []
                    return [dict(zip(cols, r)) for r in cur.fetchmany(max_rows)]

        # Les valeurs numériques sont converties et bornées puis insérées dans la requête :
        # aucune chaîne venant du LLM n'entre dans le SQL, et on ne dépend pas de la syntaxe
        # des paramètres du connecteur (`?` en mode natif, `%s` seulement en mode inline).
        def _sql_limit(limit):
            return max(1, min(50, int(limit)))

        def _sql_float(x):
            x = float(x)
            if not _np_isfinite(x):
                raise ValueError("Seuil non numérique.")
            return repr(x)

        def _fmt(rows, n=8):
            if not rows: return "Aucun résultat."
            keys = list(rows[0].keys())
            lines = [" | ".join(keys), "-" * 80]
            for r in rows[:n]:
                lines.append(" | ".join(str(r.get(k, "")) for k in keys))
            if len(rows) > n:
                lines.append(f"... ({len(rows)-n} lignes supplémentaires)")
            return "\n".join(lines)

        class _UP(_BM):
            roi_threshold: float = _F(default=0.0, description="Seuil de ROI en fraction (0 = ROI nul, 0.5 = +50 %).")
            limit: int = _F(default=10, ge=1, le=50)

        @_tool("get_underperforming_campaigns", args_schema=_UP)
        def _underperforming(roi_threshold=0.0, limit=10):
            """Campagnes avec ROI sous le seuil."""
            rows = _run_sql(
                "SELECT campaign_id, campaign_name, category, mark_spent, revenue, roi FROM marketing_kpi "
                f"WHERE roi < {_sql_float(roi_threshold)} ORDER BY roi ASC LIMIT {_sql_limit(limit)}"
            )
            return {"rows": rows, "summary": f"{len(rows)} campagnes sous-performantes (ROI < {roi_threshold})."}

        class _RP(_BM):
            metric: _Lit["roi", "revenue", "ctr", "cvr"] = _F(default="roi")
            direction: _Lit["top", "bottom"] = _F(default="top")
            limit: int = _F(default=10, ge=1, le=50)

        @_tool("rank_campaigns", args_schema=_RP)
        def _rank(metric="roi", direction="top", limit=10):
            """Classe les campagnes par métrique."""
            if metric not in {"roi", "revenue", "ctr", "cvr"}:
                raise ValueError("Métrique non autorisée.")
            order = "DESC" if direction == "top" else "ASC"
            rows = _run_sql(
                f"SELECT campaign_id, campaign_name, category, mark_spent, revenue, roi FROM marketing_kpi ORDER BY {metric} {order} LIMIT {_sql_limit(limit)}"
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

        class _SB(_BM):
            budget_increase_pct: float = _F(default=20.0, description="Variation du budget en % (ex: 20 = +20%, -30 = -30%).")
            category: str = _F(default="all", description="Canal : all, social, search, influencer ou media.")

        @_tool("simulate_budget", args_schema=_SB)
        def _simulate(budget_increase_pct=20.0, category="all"):
            """Simule l'impact d'une variation de budget sur le revenue (modèle budget + canal, voir budget_simulation.py)."""
            # Requête fixe, sans paramètre utilisateur : la catégorie est validée dans simulate_budget_increase
            rows = _run_sql(SIM_SQL_QUERY, max_rows=100000)
            return simulate_budget_increase(pd.DataFrame(rows), budget_increase_pct, category)

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
                "Si la question compare deux plateformes, genere 2 tool calls separes, un par plateforme."
            )

            try:
                campaigns_list = dff["campaign"].dropna().unique().tolist()
            except Exception:
                campaigns_list = []

            # FIX 3: system_prompt was defined but never passed — added _SM(content=system_prompt)
            msg = _llm.invoke([
                _SM(content=system_prompt),
                _HM(content=(
                    "Question: " + state.user_question + "\n"
                    "Intention: " + str(state.intent) + "\n"
                    "Campagnes disponibles (pour rank/underperforming UNIQUEMENT): " + str(campaigns_list) + "\n"
                    "IMPORTANT: Pour aggregate_by_dimension, group_by doit etre EXACTEMENT 'category'. "
                    "Ne jamais utiliser un nom de campagne comme valeur de group_by."
                ))
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
                ctx_parts.append("TOOL: " + r["tool"] + "\nSUMMARY: " + out.get("summary","") + "\nDATA:\n" + _fmt(out.get("rows", [])))
            ctx = "\n\n".join(ctx_parts) if ctx_parts else "Aucun résultat."

            system_content = (
                "Assistant marketing analytique. Reponds en francais, oriente business.\n"
                "REGLES DE FORMATAGE :\n"
                "- Montants monetaires : format CHF avec espaces (ex: CHF 3098, CHF 42889)\n"
                "- ROI et CTR : en pourcentage avec 1 decimale (ex: ROI 307.0%, CTR 0.9%)\n"
                "- ROAS : 1 decimale avec x (ex: ROAS 2.4x)\n"
                "Termine par 2-4 recommandations concretes. Ne cite que des chiffres presents dans les donnees."
            )

            user_content = (
                "Question: " + state.user_question + "\n\n"
                "IMPORTANT: Si les données contiennent plusieurs catégories ou campagnes, présente-les TOUTES.\n"
                "Ne résume pas à une seule ligne sauf si les données n'en contiennent qu'une.\n\n"
                "Donnees disponibles:\n" + ctx
            )

            msg = _llm.invoke([
                _SM(content=system_content),
                _HM(content=user_content)
            ])
            state.final_answer = msg.content
            return state

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

        st.markdown("**💡 Exemples de questions :**")
        examples = ["Top 5 campagnes par ROI", "Campagnes sous-performantes (ROI < 0)", "Analyse par catégorie", "Simule +20% budget sur social"]
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
                    if "invalid access token" in str(e).lower():
                        st.error("Databricks refuse la connexion : le jeton d'accès (DATABRICKS_ACCESS_TOKEN) est "
                                 "invalide ou expiré. Génère un nouveau Personal Access Token dans Databricks "
                                 "(Settings → Developer → Access tokens) et remplace-le dans les secrets Streamlit.")
                    else:
                        st.error(f"Erreur agent : {e}")

# ============================================================
# TAB 6 - FINE-TUNED AI
# ============================================================
FINETUNED_SPACE = os.environ.get("FINETUNED_SPACE", "Doers97/marketing-lora-demo")
FINETUNED_ADAPTER = "Doers97/marketing-lora-tinyllama"


def call_finetuned_model(space, args, timeout=180):
    """Appelle le Space Hugging Face (TinyLlama + LoRA). Lève une exception si indisponible."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout

    def _call():
        from gradio_client import Client
        return Client(space, verbose=False).predict(*args, api_name="/recommend")

    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(_call).result(timeout=timeout)
    except _Timeout:
        raise TimeoutError(f"pas de réponse du Space en {timeout} s (il se réveille peut-être)")
    finally:
        ex.shutdown(wait=False)


def rule_based_recommendation(channel, segment, roi, roas, ctr, cvr, cpl):
    """Mêmes seuils que les étiquettes du jeu d'entraînement du modèle."""
    if roi > 150:
        perf, action = "HIGH", "SCALE immediately"
        budget = f"Increase budget by 20-30%. Current ROI of {roi:.1f}% justifies aggressive scaling."
        risk = f"Low risk - strong ROAS of {roas:.1f}x confirms sustainable performance."
        steps = ("Allocate additional budget to top-performing ad sets", "Expand to similar audience segments")
    elif roi > 30:
        perf, action = "MEDIUM", "OPTIMIZE within 2 weeks"
        budget = f"Maintain current budget. Focus on CVR improvement from {cvr:.1f}% toward 8%+."
        risk = f"Medium risk - monitor weekly. CPL of CHF {cpl:.0f} is acceptable but improvable."
        steps = ("A/B test landing pages to improve conversion rate", f"Review audience segmentation for {segment}")
    else:
        perf, action = "LOW", "REVIEW immediately"
        budget = f"Reduce budget by 30-50% or pause campaign. ROI of {roi:.1f}% is below threshold."
        risk = f"High risk - ROAS of {roas:.1f}x does not cover acquisition costs."
        steps = ("Audit targeting and creative assets", "Reallocate budget to high-ROI campaigns")
    return (f"Campaign Analysis - {channel.upper()} | {segment}\n\n"
            f"Performance Level: {perf}\n"
            f"Key Metrics: ROI={roi:.1f}%, ROAS={roas:.1f}x, CTR={ctr:.2f}%, CVR={cvr:.2f}%, CPL=CHF {cpl:.0f}\n\n"
            f"Recommended Action: {action}\nBudget Strategy: {budget}\nRisk Assessment: {risk}\n\n"
            f"Next Steps:\n1. {steps[0]}\n2. {steps[1]}\n3. Track weekly KPI evolution and adjust accordingly.")


with tab6:
    st.subheader("🧬 Fine-tuned AI — TinyLlama 1.1B + LoRA")
    st.caption(f"Adapter [{FINETUNED_ADAPTER}](https://huggingface.co/{FINETUNED_ADAPTER}), served by the "
               f"Hugging Face Space [{FINETUNED_SPACE}](https://huggingface.co/spaces/{FINETUNED_SPACE}).")

    with st.expander("ℹ️ Qu'est-ce qu'un modèle fine-tuné ?"):
        st.markdown("""
**TinyLlama** est un petit modèle de langage généraliste (1,1 milliard de paramètres).  
Le **fine-tuning** consiste à le ré-entraîner sur des exemples d'un métier précis, ici 500+ analyses de  
campagnes marketing (KPI → recommandation), pour qu'il réponde dans ce format et avec ce vocabulaire.

**LoRA** est une technique de fine-tuning économique : au lieu de modifier tout le modèle, on n'entraîne que  
de petites couches ajoutées (0,1 % des paramètres). L'entraînement tient en quelques minutes sur un GPU gratuit.

**Les deux modes :**  
• **Fine-tuned model** : la recommandation est **rédigée par le modèle**, hébergé sur Hugging Face  
  (10 à 60 secondes, plus long si le service était en veille).  
• **Business rules** : la même recommandation produite par des règles fixes (ROI > 150 % → scale,  
  30–150 % → optimize, < 30 % → review), instantanée. Elle sert de référence et de secours.

**Les indicateurs « Derived »** (ROAS, revenu, CPL) sont calculés à partir de tes saisies pour rester cohérents  
entre eux, comme dans les données d'entraînement.
""")

    mode = st.radio("Generation", ["Fine-tuned model (Hugging Face Space)", "Business rules (instant)"],
                    horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        channel = st.selectbox("Channel", ["social", "search", "influencer", "media", "email"])
        segment = st.selectbox("Segment", ["B2C", "B2B", "Premium", "Mass Market"])
        spend = st.number_input("Budget (CHF)", min_value=500, max_value=100000, value=10000, step=500)
        impressions = st.number_input("Impressions", min_value=1000, max_value=5_000_000, value=200_000, step=10_000)
    with col2:
        roi = st.slider("ROI (%)", -50, 300, 45)
        ctr_input = st.number_input("CTR (%)", min_value=0.1, max_value=20.0, value=2.5, step=0.1)
        cvr_input = st.number_input("CVR (%)", min_value=0.1, max_value=30.0, value=5.0, step=0.1)

    # Grandeurs dérivées, calculées comme dans le jeu d'entraînement (cohérentes entre elles)
    roas = 1 + roi / 100
    revenue = spend * roas
    conversions = impressions * ctr_input / 100 * cvr_input / 100
    cpl = spend / conversions if conversions >= 1 else spend
    st.caption(f"Derived: ROAS {roas:.2f}x · Revenue CHF {revenue:,.0f} · Conversions {conversions:,.0f} · "
               f"CPL CHF {cpl:,.0f}".replace(",", " "))

    if st.button("🚀 Generate Recommendation"):
        rules = rule_based_recommendation(channel, segment, roi, roas, ctr_input, cvr_input, cpl)
        if mode.startswith("Fine-tuned"):
            with st.spinner("TinyLlama generates the recommendation (10-60 s, longer if the Space is waking up)..."):
                try:
                    text = call_finetuned_model(FINETUNED_SPACE, [
                        channel, segment, float(spend), int(impressions), float(ctr_input), float(cvr_input),
                        float(roi), float(roas), float(cpl), float(revenue)])
                    st.success("✅ Generated by the fine-tuned model (TinyLlama 1.1B + LoRA)")
                    st.code(text)
                    with st.expander("Compare with the business rules"):
                        st.code(rules)
                except Exception as e:
                    st.warning(f"Fine-tuned model unavailable ({e}). Showing the business-rule recommendation instead.")
                    st.code(rules)
        else:
            st.success("✅ Business-rule recommendation (same thresholds as the model's training labels)")
            st.code(rules)
