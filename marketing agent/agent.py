import os

import json

from typing import Literal, Optional, List, Dict, Any



from databricks import sql as dbsql

from langchain_openai import ChatOpenAI

from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.tools import tool

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END





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

    group_by: Literal[

        "category",

        "campaign_name",

        "campaign_id",

        "c_date",

    ] = Field(description="Dimension d'agrégation.")





@tool("aggregate_by_dimension", args_schema=AggParams)

def aggregate_by_dimension(group_by: str = "category") -> Dict[str, Any]:

    """Agrège les KPI par catégorie (category, campaign_name, campaign_id, c_date)."""

    # FIX 1: indentation error — leading space removed

    allowed = {"category", "campaign_name", "campaign_id", "c_date"}

    if group_by not in allowed:

        raise ValueError(f"Dimension non autorisée. Valeurs acceptées : {allowed}")

    query = f"""

        SELECT

            {group_by},

            COUNT(*) as n_campaigns,

            ROUND(SUM(mark_spent), 2) as total_cost,

            ROUND(SUM(revenue), 2) as total_revenue,

            ROUND(AVG(roi), 4) as avg_roi,

            ROUND(AVG(ctr), 4) as avg_ctr,

            ROUND(AVG(cvr), 4) as avg_cvr,

            ROUND(SUM(clicks), 0) as total_clicks,

            ROUND(SUM(leads), 0) as total_leads,

            ROUND(SUM(orders), 0) as total_orders

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





class AgentState(BaseModel):

    user_question: str

    intent: Optional[str] = None

    tool_calls: List[Dict[str, Any]] = []

    tool_results: List[Dict[str, Any]] = []

    final_answer: Optional[str] = None





llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)





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

    # FIX 2: budget_simulation was missing from the valid set → always fell back to kpi_qa

    if intent not in {"kpi_qa", "diagnostic", "segmentation", "budget_simulation"}:

        intent = "kpi_qa"

    state.intent = intent

    return state





PLANNER_SYSTEM = """Tu es un planner pour un assistant marketing.

Tools disponibles :

- get_underperforming_campaigns(roi_threshold, limit)

- rank_campaigns(metric, direction, limit)

    → metric : roi, revenue, ctr, cvr uniquement

    → direction : top, bottom uniquement

- aggregate_by_dimension(group_by)

    → group_by : category, campaign_name, campaign_id, c_date uniquement

    ❌ JAMAIS mettre un nom de campagne comme valeur de group_by

- simulate_budget(budget_increase_pct, category)

    → category : social, search, influencer, media, all



RÈGLES :

- Choisis 1 à 2 tools maximum.

- Retourne UNIQUEMENT un JSON valide.

- Exemple :

[{{"tool":"rank_campaigns","args":{{"metric":"roi","direction":"top","limit":10}}}}]

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
