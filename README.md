# notebooks/ — Exploration & Analysis

Jupyter notebooks for EDA, model tuning, and threshold analysis.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda_card.ipynb` | Explore credit card fraud dataset — distributions, correlations, class imbalance |
| `02_eda_login.ipynb` | Login event analysis — impossible travel, device patterns |
| `03_model_tuning.ipynb` | Hyperparameter search, cross-validation, SHAP explainability |
| `04_threshold_analysis.ipynb` | Precision-recall tradeoff, cost-benefit analysis per threshold |

---

## Setup

```bash
pip install jupyter notebook ipykernel matplotlib seaborn shap
jupyter notebook
```

---

## Recommended Notebook Workflow

```python
# 1. Load real or synthetic data
from train import gen_card
df, feat_cols = gen_card(n=50_000, fraud_rate=0.02)

# 2. Quick EDA
df.describe()
df["label"].value_counts(normalize=True)
df.hist(figsize=(16, 10))

# 3. Train detector
from src.models.card_detector import CardFraudDetector
det = CardFraudDetector()
det.fit(df)

# 4. Check feature importance
det.feature_importance().head(10)

# 5. Tune threshold
tdf = det.tune_threshold(df)
tdf[tdf["f1"] > 0.80].sort_values("f1", ascending=False).head(10)

# 6. SHAP explainability (in 03_model_tuning.ipynb)
import shap
explainer = shap.TreeExplainer(det.model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feat_cols)
```
