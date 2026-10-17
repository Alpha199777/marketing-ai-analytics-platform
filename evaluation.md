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

## 2. Revenue models

Method:

- **5-fold cross-validation**: metrics are computed on out-of-fold predictions for every row.
  A single 80/20 split (~50 test rows) is too unstable to be reported.
- **Grouped cross-validation by campaign**: all rows of a campaign are in the same fold,
  to measure performance on campaigns the model has never seen.
- **Metrics**: R², RMSE, MAPE and median absolute percentage error (rows with revenue > 0).
  The median error is reported first: MAPE is dominated by very small campaigns.
- **Baseline**: `revenue = orders × average basket`. A model is only kept if it beats it.
- **Leakage check**: no input may be computed from the target. `roi` is excluded
  (`roi = (revenue − spend) / spend`).

Two distinct models:

| Model | Inputs | Purpose |
|---|---|---|
| Explanatory | impressions, clicks, leads, orders, spend, CTR, CVR, CPL | explain revenue of completed campaigns |
| Simulation | spend, channel | forecast before launch |

The simulation model reports a range (10th–90th percentile of actual / predicted in cross-validation)
and flags budgets outside the range observed for the channel.

Results are in the README.

---

## 3. Campaign segmentation (clustering)

KMeans clustering is validated by:

- Building segments on campaign **profile** only (CTR, CVR, CPL, log budget);
  revenue and ROI are excluded from the inputs and used to describe segments
- Choosing k with the silhouette score (k = 2 to 6)
- Comparing distributions of ROI, CVR and CPL across clusters
- Verifying that each cluster corresponds to a meaningful business segment
  (e.g. high CTR, poor conversion, high CPL)

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
