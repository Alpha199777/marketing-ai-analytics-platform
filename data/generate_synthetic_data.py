"""Génère data/sample_marketing.csv : données marketing 100 % synthétiques (même schéma que marketing_clean).

Aucune donnée réelle : les paramètres par canal sont inventés pour produire un jeu de démo plausible.
Usage : python data/generate_synthetic_data.py data/sample_marketing.csv
"""
import numpy as np, pandas as pd, sys
rng = np.random.default_rng(2026)
# paramètres par canal : budget médian, CPM médian, CTR médian, taux de commande / clic, panier moyen
CAT = {
    "influencer": dict(spend=60000, cpm=90, ctr=0.010, order=0.0033, aov=6000),
    "media":      dict(spend=60000, cpm=13, ctr=0.0012, order=0.0035, aov=3900),
    "search":     dict(spend=25000, cpm=75, ctr=0.012, order=0.0018, aov=3900),
    "social":     dict(spend=28000, cpm=55, ctr=0.006, order=0.0016, aov=4700),
}
CAMPAIGNS = [("creator_video","influencer"),("creator_stories","influencer"),
             ("display_network","media"),
             ("search_brand","search"),("search_generic","search"),
             ("social_prospecting","social"),("social_lookalike","social"),("social_retargeting","social"),
             ("social_broad","social"),("social_interest","social"),("social_video","social")]
dates = pd.date_range("2025-03-01", periods=28, freq="D")
rows, rid = [], 1
for name, cat in CAMPAIGNS:
    p = CAT[cat]
    camp_id = int(rng.integers(100000, 9999999))
    eff = rng.lognormal(0, 0.15)          # qualité propre à la campagne
    for d in dates:
        spend = round(p["spend"] * rng.lognormal(0, 1.0), 2)
        impressions = max(1, int(spend / (p["cpm"] * rng.lognormal(0, 0.35)) * 1000))
        clicks = int(rng.poisson(impressions * p["ctr"] * eff * rng.lognormal(0, 0.25)))
        leads = int(rng.poisson(clicks * 0.02 * rng.lognormal(0, 0.4)))
        orders = int(rng.poisson(clicks * p["order"] * eff * rng.lognormal(0, 0.2)))
        revenue = float(round(orders * p["aov"] * rng.lognormal(0, 0.2)))
        rows.append(dict(id=rid, c_date=d.date().isoformat(), campaign_name=name, category=cat,
                         campaign_id=camp_id, impressions=impressions, mark_spent=spend,
                         clicks=clicks, leads=leads, orders=orders, revenue=revenue))
        rid += 1
pd.DataFrame(rows).to_csv(sys.argv[1] if len(sys.argv) > 1 else "sample_marketing.csv", index=False)
