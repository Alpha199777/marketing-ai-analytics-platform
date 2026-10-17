# Model training

Data: `marketing_kpi` (308 rows, 11 campaigns). All metrics from 5-fold cross-validation.

Explanatory model: RandomForestRegressor (PySpark MLlib, 50 trees, max depth 8), without `roi`.
- R² = 0.77, median error = 26 %
- Baseline `revenue = orders × average basket`: R² = 0.88 → the baseline wins.

Simulation model: RandomForestRegressor (200 trees, max depth 6, all features per split),
inputs = budget + channel.
- R² = 0.63, median error = 45 %

See `evaluation.md` for the method and the README for the full results.
