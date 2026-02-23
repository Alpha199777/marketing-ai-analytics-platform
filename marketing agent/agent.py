import os
import json
from typing import Literal, Optional, List, Dict, Any

from databricks import sql as dbsql
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END


# ─────────────────────────────────────────
# 1) CONNEXION DATABRICKS SQL WAREHOUSE
# ─────────────────────────────────────────
def run_sql(query: str, params: tuple = (), max_rows: int = 200) -> List[Dict[str, Any]]:
    hostname  = os.environ["DATABRICKS_SERVER_HOSTNAME"]
    http_path = os.environ["DATABRICKS_HTTP_PATH"]
    token     = os.environ["DATABRICKS_ACCESS_TOKEN"]

    with dbsql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=token
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            cols = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows)
            return [dict(zip(cols, r)) for r in rows]


def format_rows(rows: List[Dict[str, Any]], max_items: int = 10) -> str:
    if not rows:
        return "Aucun résultat."
    keys = list(rows[0].keys())
    lines = [" | ".join(keys), "-" * 80]
    for r in rows[:max_items]:
        lines.append(" | ".join(str(r.get(k, "")) for k in keys))
    if len(rows) > max_items:
        lines.append(f"... ({len(rows) - max_items} lignes supplémentaires)")
    return "\n".join(lines)


# ─────────────────────────────────────────
# 2) TOOLS (actions disponibles pour l'agent)
# ─────────────────────────────────────────

class UnderperformingParams(BaseModel):
    roi_threshold: float = Field(default=0.0, description="Seuil ROI en dessous duquel une campagne est sous-performante.")
    limit: int = Field(default=10, ge=1, le=50)

@tool("get_underperforming_campaigns", args_schema=UnderperformingParams)
def get_underperforming_campaigns(roi_threshold: float = 0.0, limit: int = 10) -> Dict[str, Any]:
    """Retourne les campagnes avec un ROI inférieur au seuil donné."""
    query = """
        SELECT campaign_id, campaign_name, category, mark_spent, revenue, roi, ctr, cvr, cpl
        FROM marketing_kpi
        WHERE roi < ?
        ORDER BY roi ASC
        LIMIT ?
    """
    rows = run_sql(query, (roi_threshold, limit))
    return {"rows": rows, "summary": f"{len(rows)} campagnes sous-performantes (ROI < {roi_threshold})."}


class RankParams(BaseModel):
    metric: Literal["roi", "revenue", "ctr", "cvr"] = Field(description="Métrique de classement.")
    direction: Literal["top", "bottom"] = Field(default="top")
    limit: int = Field(default=10, ge=1, le=50)

@tool("rank_campaigns", args_schema=RankParams)
def rank_campaigns(metric: str, direction: str = "top", limit: int = 10) -> Dict[str, Any]:
    """Classe les campagnes par une métrique (top ou bottom)."""
    if metric not in {"roi", "revenue", "ctr", "cvr"}:
        raise ValueError("Métrique non autorisée.")
    order = "DESC" if direction == "top" else "ASC"
    query = f"""
        SELECT campaign_id, campaign_name, category, mark_spent, revenue, roi, ctr, cvr, cpl
        FROM marketing_kpi
        ORDER BY {metric} {order}
        LIMIT ?
    """
    rows = run_sql(query, (limit,))
    return {"rows": rows, "summary": f"{direction} {limit} campagnes par {metric}."}


class AggParams(BaseModel):
    group_by: Literal["category"] = Field(description="Dimension d'agrégation (ex: category).")

@tool("aggregate_by_dimension", args_schema=AggParams)
def aggregate_by_dimension(group_by: str = "category") -> Dict[str, Any]:
    """Agrège les KPI par catégorie (social, search, etc.)."""
    if group_by not in {"category"}:
        raise ValueError("Dimension non autorisée.")
    query = f"""
        SELECT
            {group_by},
            COUNT(*) as n_campaigns,
            ROUND(SUM(mark_spent), 2) as total_cost,
            ROUND(SUM(revenue), 2) as total_revenue,
            ROUND(AVG(roi), 4) as avg_roi,
            ROUND(AVG(ctr), 4) as avg_ctr,
            ROUND(AVG(cvr), 4) as avg_cvr
        FROM marketing_kpi
        GROUP BY {group_by}
        ORDER BY total_revenue DESC
    """
    rows = run_sql(query)
    return {"rows": rows, "summary": f"Agrégation par {group_by} ({len(rows)} groupes)."}


class SimulateBudgetParams(BaseModel):
    budget_increase_pct: float = Field(default=20.0, description="Pourcentage d'augmentation du budget.")
    category: str = Field(default="all", description="Catégorie à simuler (social, search, influencer, all).")

@tool("simulate_budget", args_schema=SimulateBudgetParams)
def simulate_budget(budget_increase_pct: float = 20.0, category: str = "all") -> Dict[str, Any]:
    """Simule l'impact d'une augmentation de budget sur le revenue via RandomForest."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    where = f"WHERE category = '{category}'" if category != "all" else ""
    rows = run_sql(f"""
        SELECT mark_spent, clicks, impressions, ctr, cvr, revenue
        FROM marketing_kpi {where}
        ORDER BY revenue DESC LIMIT 500
    """)
    if len(rows) < 10:
        return {"summary": "Pas assez de données pour simuler.", "rows": []}

    df = pd.DataFrame(rows)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    features = [f for f in ["mark_spent", "clicks", "impressions", "ctr", "cvr"] if f in df.columns]
    X = df[features].values
    y = np.log1p(df["revenue"].values)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    X_base = df[features].copy()
    rev_base = np.expm1(model.predict(X_base.values)).sum()

    X_sim = X_base.copy()
    X_sim["mark_spent"] = X_sim["mark_spent"] * (1 + budget_increase_pct / 100)
    if "clicks" in features:
        X_sim["clicks"] = X_sim["clicks"] * (1 + budget_increase_pct / 100 * 0.7)
    rev_sim = np.expm1(model.predict(X_sim.values)).sum()

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
               f"(+{delta_pct:.1f}%, ROI incrémental : {roi_sim:.2f})")
    return {"rows": result_rows, "summary": summary}


TOOLS = [get_underperforming_campaigns, rank_campaigns, aggregate_by_dimension, simulate_budget]
TOOL_MAP = {t.name: t for t in TOOLS}


# ─────────────────────────────────────────
# 3) ETAT DE L'AGENT
# ─────────────────────────────────────────
class AgentState(BaseModel):
    user_question: str
    intent: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None


# ─────────────────────────────────────────
# 4) LLM
# ─────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# ─────────────────────────────────────────
# 5) NOEUDS DU GRAPHE
# ─────────────────────────────────────────
ROUTER_SYSTEM = """Tu es un routeur pour un assistant marketing analytique.
Choisis UNE intention parmi :
- kpi_qa       : question descriptive, ranking, top/bottom campagnes
- diagnostic   : identifier les campagnes sous-performantes et leurs causes
- segmentation : analyse par canal / catégorie (social, search, etc.)
- budget_simulation : simulation d'impact d'un changement de budget

Réponds UNIQUEMENT avec un seul mot : kpi_qa, diagnostic, segmentation, ou budget_simulation.
"""

def route_intent(state: AgentState) -> AgentState:
    msg = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=state.user_question)
    ])
    intent = msg.content.strip()
    if intent not in {"kpi_qa", "diagnostic", "segmentation"}:
        intent = "kpi_qa"
    state.intent = intent
    return state


PLANNER_SYSTEM = """Tu es un planner pour un assistant marketing.
Tools disponibles :
- get_underperforming_campaigns(roi_threshold, limit)  → campagnes sous-performantes
- rank_campaigns(metric, direction, limit)             → top/bottom par métrique (roi, revenue, ctr, cvr)
- aggregate_by_dimension(group_by)                     → agrégation par category
- simulate_budget(budget_increase_pct, category)       → simulation impact budget

Choisis 1 à 2 tools maximum selon la question et l'intention.
Retourne UNIQUEMENT un JSON valide, exemple :
[{"tool":"rank_campaigns","args":{"metric":"roi","direction":"top","limit":10}}]
"""

def plan_tools(state: AgentState) -> AgentState:
    msg = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Question: {state.user_question}\nIntention: {state.intent}")
    ])
    content = msg.content.strip().replace("```json", "").replace("```", "")
    try:
        calls = json.loads(content)
        if not isinstance(calls, list):
            calls = []
    except Exception:
        calls = []
    state.tool_calls = calls[:2]
    return state


def execute_tools(state: AgentState) -> AgentState:
    results = []
    for call in state.tool_calls:
        tool_name = call.get("tool")
        args = call.get("args", {})
        if tool_name not in TOOL_MAP:
            continue
        out = TOOL_MAP[tool_name].invoke(args)
        results.append({"tool": tool_name, "args": args, "output": out})
    state.tool_results = results
    return state


COMPOSER_SYSTEM = """Tu es un assistant marketing analytique expert.
Règles IMPORTANTES :
- Ne fais AUCUNE affirmation chiffrée qui n'apparaît pas dans les données.
- Si aucun résultat : dis clairement que tu ne peux pas conclure.
- Mentionne toujours quels filtres/critères ont été utilisés.
- Termine par 2 à 4 recommandations concrètes et actionnables.
Réponds en français, ton professionnel et orienté business.
"""

def compose_answer(state: AgentState) -> AgentState:
    ctx_parts = []
    for r in state.tool_results:
        out = r["output"]
        rows = out.get("rows", [])
        ctx_parts.append(
            f"TOOL: {r['tool']} | ARGS: {r['args']}\n"
            f"RÉSUMÉ: {out.get('summary','')}\n"
            f"DONNÉES:\n{format_rows(rows)}\n"
        )
    context = "\n\n".join(ctx_parts) if ctx_parts else "Aucun résultat disponible."

    msg = llm.invoke([
        SystemMessage(content=COMPOSER_SYSTEM),
        HumanMessage(content=f"Question: {state.user_question}\n\nDonnées:\n{context}")
    ])
    state.final_answer = msg.content
    return state


# ─────────────────────────────────────────
# 6) CONSTRUCTION DU GRAPHE LANGGRAPH
# ─────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route_intent",   route_intent)
    g.add_node("plan_tools",     plan_tools)
    g.add_node("execute_tools",  execute_tools)
    g.add_node("compose_answer", compose_answer)

    g.set_entry_point("route_intent")
    g.add_edge("route_intent",   "plan_tools")
    g.add_edge("plan_tools",     "execute_tools")
    g.add_edge("execute_tools",  "compose_answer")
    g.add_edge("compose_answer", END)
    return g.compile()

GRAPH = build_graph()


# ─────────────────────────────────────────
# 7) FONCTION PRINCIPALE
# ─────────────────────────────────────────
def ask_agent(question: str) -> str:
    state = AgentState(user_question=question)
    out = GRAPH.invoke(state)
    return out["final_answer"]


# ─────────────────────────────────────────
# TEST RAPIDE (lance : python agent.py)
# ─────────────────────────────────────────
if __name__ == "__main__":
    question = "Quelles sont les 5 campagnes avec le meilleur ROI ?"
    print(f"\n❓ Question : {question}\n")
    print("⏳ L'agent réfléchit...\n")
    reponse = ask_agent(question)
    print("🤖 Réponse de l'agent :\n")
    print(reponse)
