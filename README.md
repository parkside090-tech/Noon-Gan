# Noon Fraud Lab — Results Dashboard

Streamlit dashboard showing the lab's before/after metrics, confusion matrices and comparison charts.

## Run
```
pip install -r requirements.txt
streamlit run dashboard.py
```

## Tabs
1. **Before vs. after GAN** — my seed's held-out test set: 12 real fraud rows vs. 12 + 300 GAN rows (and the 320-row ceiling): grouped recall/precision/F1, false positives, three confusion matrices, metrics table.
2. **Real data: me vs. Noon** — the 20,000-row referee set: my model before/after GAN vs. Noon's (plus Noon's reported real-only benchmark), confusion matrices.
3. **My GAN vs. Noon's GAN** — distributions of six key columns (optional real-fraud overlay), binary-flag rates, lie-detector feature importances, fraud proportions.
4. **Assumptions check** — my seed's column means vs. real legit and real fraud.

## Files
- `dashboard.py` — the app (reads only `data/`, no training at runtime)
- `data/metrics.json` — all precomputed metrics and confusion matrices
- `data/my_gan_synthetic_fraud.csv`, `data/noon_synthetic_fraud.csv`, `data/real_fraud_referee.csv` — rows for the distribution plots
- `build_dashboard_data.py` — rebuilds `data/` from the notebook if you rerun the lab

The fraud-check tool is separate: `streamlit run app.py` (needs `fraud_model.joblib`).
