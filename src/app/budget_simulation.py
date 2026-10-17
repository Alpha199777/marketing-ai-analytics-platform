"""Simulation d'une variation de budget, partagée par l'app Streamlit et l'agent LangGraph.

Principe : le modèle ne reçoit que des leviers connus avant d'investir, le budget (mark_spent)
et le canal (category). Il n'utilise ni les clics ni les commandes, qui sont des résultats.
L'effet d'une variation de budget = revenu prédit avec le nouveau budget − revenu prédit avec
le budget actuel ; cet écart est ajouté au revenu réellement observé.

Aucune entrée utilisateur n'entre dans une requête SQL : les données sont lues en entier, puis la
catégorie est validée et filtrée ici.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REQUIRED_COLUMNS = ("mark_spent", "category", "revenue")
SQL_QUERY = "SELECT mark_spent, category, revenue FROM marketing_kpi"


def _design(spend, category, categories) -> pd.DataFrame:
    X = pd.DataFrame({"mark_spent": np.asarray(spend, dtype="float64")})
    category = np.asarray(category, dtype=str)
    for c in categories:
        X["cat_" + c] = (category == c).astype(float)
    return X


def simulate_budget_increase(df: pd.DataFrame, budget_increase_pct: float = 20.0,
                             category: str = "all") -> Dict[str, Any]:
    """Retourne {"rows": [...], "summary": str} au format des outils de l'agent."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return {"rows": [], "summary": f"Colonnes manquantes pour simuler : {missing}."}

    data = df[list(REQUIRED_COLUMNS)].copy()
    data["mark_spent"] = pd.to_numeric(data["mark_spent"], errors="coerce")
    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
    data["category"] = data["category"].astype(str).str.strip().str.lower()
    data = data.dropna()

    categories = tuple(sorted(data["category"].unique()))
    category = str(category).strip().lower()
    if category != "all" and category not in categories:
        return {"rows": [], "summary": f"Catégorie inconnue : '{category}'. Valeurs possibles : all, {', '.join(categories)}."}
    if not -90 <= float(budget_increase_pct) <= 300:
        return {"rows": [], "summary": "La variation de budget doit être comprise entre -90 % et +300 %."}

    target = data if category == "all" else data[data["category"] == category]
    if len(data) < 30 or len(target) < 10:
        return {"rows": [], "summary": "Pas assez de données pour simuler."}

    # Entraîné sur tous les canaux (plus de données), appliqué aux lignes du canal demandé
    model = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(_design(data["mark_spent"], data["category"], categories), data["revenue"].values)

    factor = 1 + float(budget_increase_pct) / 100
    new_spend = target["mark_spent"] * factor
    pred_now = model.predict(_design(target["mark_spent"], target["category"], categories)).sum()
    pred_new = model.predict(_design(new_spend, target["category"], categories)).sum()

    budget_now = float(target["mark_spent"].sum())
    extra_budget = budget_now * (factor - 1)
    revenue_now = float(target["revenue"].sum())
    delta = float(pred_new - pred_now)

    # Au-delà du budget maximal observé pour un canal, la forêt ne peut pas extrapoler (prévision plate)
    max_by_cat = data.groupby("category")["mark_spent"].max()
    out_of_range = float((new_spend > target["category"].map(max_by_cat)).mean())

    row = {
        "categorie": category,
        "budget_actuel": round(budget_now, 2),
        "budget_simule": round(budget_now * factor, 2),
        "revenue_actuel": round(revenue_now, 2),
        "revenue_simule": round(revenue_now + delta, 2),
        "delta_revenue": round(delta, 2),
        "delta_pct": round(delta / revenue_now * 100, 2) if revenue_now > 0 else 0.0,
        # revenu supplémentaire par franc supplémentaire, et ROI de ce franc ((revenu - coût) / coût)
        "roas_incremental": round(delta / extra_budget, 4) if extra_budget != 0 else 0.0,
        "roi_incremental": round((delta - extra_budget) / extra_budget, 4) if extra_budget != 0 else 0.0,
        "part_hors_plage": round(out_of_range, 3),
    }
    summary = (f"Simulation {budget_increase_pct:+g}% de budget ({category}) : revenu {revenue_now:,.0f} -> "
               f"{revenue_now + delta:,.0f} ({row['delta_pct']:+.1f} %) ; chaque franc ajouté ou retiré pèse "
               f"{row['roas_incremental']:.2f} de revenu (ROI incrémental {row['roi_incremental']:+.0%}). "
               "Modèle budget + canal, précision limitée (R² ≈ 0,6 en validation croisée) : ordre de grandeur, pas une certitude.")
    if out_of_range > 0.2:
        summary += (f" Attention : {out_of_range:.0%} des lignes dépassent le budget maximal observé pour leur canal ; "
                    "le modèle ne prévoit pas au-delà, résultat peu fiable.")
    return {"rows": [row], "summary": summary}
