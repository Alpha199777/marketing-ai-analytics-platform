# Evaluation

This project evaluates both the analytics layer and the ML models
to ensure the system produces reliable and actionable results.

---

## 1. KPI correctness

To validate the KPI layer (`marketing_kpi`):

- Random campaigns are selected
- Raw values (clicks, impressions, spend, revenue) are manually aggregated
- KPI formulas (CTR, CVR, CPL, ROI) are recomputed
- Results are compared with the values stored in `marketing_kpi`

Acceptance criteria:
- Differences < 1%
- No division by zero errors
- No nulls for active campaigns

---

## 2. Revenue prediction model

The RandomForest regression model is evaluated using:

- Train / test split
- R² (coefficient of determination)
- RMSE (root mean squared error)

Additional checks:
- Compare predicted vs actual revenue for high- and low-spend campaigns
- Verify that errors are not biased toward a single campaign type

---

## 3. Campaign segmentation (clustering)

KMeans clustering is validated by:

- Inspecting cluster centroids
- Comparing distributions of ROI, CVR and CPL across clusters
- Verifying that each cluster corresponds to a meaningful business segment
  (e.g. high ROI, low conversion, high volume, etc.)

---

## 4. RAG quality (structured data)

The RAG system is evaluated using a set of test questions:

Examples:
- "Which campaigns have the lowest ROI?"
- "What happens if we increase the budget by 20%?"
- "Which segment should receive more investment?"

For each question:
- Retrieved rows from `marketing_kpi` are checked
- Generated answers are validated against the KPI values

Acceptance criteria:
- Answers must be numerically consistent
- Recommendations must be supported by retrieved metrics
