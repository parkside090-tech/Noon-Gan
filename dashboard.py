# dashboard.py — Noon Fraud Lab results dashboard
# Run locally:  streamlit run dashboard.py
# Reads precomputed results from ./data (built from the lab notebook), so it needs no model training.
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
# Files may sit in ./data (as shipped) or next to dashboard.py (if the data folder was flattened on upload)
SEARCH = [HERE / "data", HERE]
MINE_LIGHT, MINE, NOON_LIGHT, NOON = "#86b6ef", "#2a78d6", "#f5b49a", "#eb6834"
GREY = "#9a9994"

st.set_page_config(page_title="Noon Fraud Lab — Results", page_icon="📊", layout="wide")

# Fallback copy of data/metrics.json, so the metric and confusion-matrix tabs work even if the file is missing.
EMBEDDED_METRICS = json.loads(r"""{"seed_test": {"(A) Full natural (320 real fraud)": {"recall": 0.9375, "precision": 0.974025974025974, "f1": 0.9554140127388535, "accuracy": 0.993, "cm": [[918, 2], [5, 75]], "train_fraud": 320, "train_rows": 4000}, "(B) BEFORE: scarce 12 real fraud only": {"recall": 0.375, "precision": 1.0, "f1": 0.5454545454545454, "accuracy": 0.95, "cm": [[920, 0], [50, 30]], "train_fraud": 12, "train_rows": 3692}, "(C) AFTER: 12 real + 300 GAN-synthetic fraud": {"recall": 0.2125, "precision": 1.0, "f1": 0.35051546391752575, "accuracy": 0.937, "cm": [[920, 0], [63, 17]], "train_fraud": 312, "train_rows": 3992}}, "referee": {"Mine — before (seed only)": {"recall": 0.46153846153846156, "precision": 0.14705882352941177, "f1": 0.22304832713754646, "accuracy": 0.98955, "cm": [[19761, 174], [35, 30]]}, "Mine — after (seed + my GAN)": {"recall": 0.6, "precision": 0.021487603305785124, "f1": 0.04148936170212766, "accuracy": 0.9099, "cm": [[18159, 1776], [26, 39]]}, "Noon (real + Noon GAN)": {"recall": 0.8153846153846154, "precision": 0.4274193548387097, "f1": 0.5608465608465608, "accuracy": 0.99585, "cm": [[19864, 71], [12, 53]]}, "Noon real-only (reported)": {"recall": 0.14, "precision": 0.9, "f1": 0.24, "accuracy": null, "cm": null, "false_positives": 1}}, "lie_detector": {"accuracy": 0.9888333333333333, "importances": {"delivery_distance_km": 0.2122, "num_orders_last_7d": 0.154, "account_age_days": 0.1321, "order_amount": 0.1105, "hour_of_day": 0.1028, "prev_refunds": 0.0806, "payment_method_card": 0.0618, "payment_method_wallet": 0.0539, "payment_method_applepay": 0.0231, "payment_method_cod": 0.0212, "location_mismatch": 0.0184, "device_change": 0.0173, "discount_applied": 0.0121}}, "proportions": {"My synthetic file": [3000, 1.0], "Noon synthetic file": [3000, 1.0], "My seed (train)": [4000, 0.08], "My training set after GAN": [7000, 0.4742857142857143], "Real referee data": [20000, 0.00325]}, "assumption_check": {"My seed legit": {"order_amount": 61.693, "account_age_days": 613.369, "hour_of_day": 14.117, "num_orders_last_7d": 1.515, "delivery_distance_km": 3.988, "discount_applied": 0.244, "prev_refunds": 0.296, "device_change": 0.032, "location_mismatch": 0.018}, "Real legit": {"order_amount": 99.669, "account_age_days": 263.711, "hour_of_day": 11.583, "num_orders_last_7d": 3.014, "delivery_distance_km": 5.034, "discount_applied": 0.346, "prev_refunds": 0.305, "device_change": 0.08, "location_mismatch": 0.049}, "My seed fraud": {"order_amount": 168.262, "account_age_days": 279.39, "hour_of_day": 11.53, "num_orders_last_7d": 1.898, "delivery_distance_km": 12.503, "discount_applied": 0.388, "prev_refunds": 3.138, "device_change": 0.575, "location_mismatch": 0.372}, "My GAN fraud": {"order_amount": 272.6, "account_age_days": 289.596, "hour_of_day": 10.539, "num_orders_last_7d": 1.487, "delivery_distance_km": 8.786, "discount_applied": 0.246, "prev_refunds": 1.942, "device_change": 0.682, "location_mismatch": 0.479}, "Noon GAN fraud": {"order_amount": 135.015, "account_age_days": 190.354, "hour_of_day": 7.061, "num_orders_last_7d": 3.982, "delivery_distance_km": 5.051, "discount_applied": 0.428, "prev_refunds": 0.724, "device_change": 0.648, "location_mismatch": 0.629}, "Real fraud": {"order_amount": 138.483, "account_age_days": 155.446, "hour_of_day": 8.908, "num_orders_last_7d": 3.723, "delivery_distance_km": 5.268, "discount_applied": 0.4, "prev_refunds": 0.692, "device_change": 0.708, "location_mismatch": 0.6}}}""")


def find(name):
    for d in SEARCH:
        if (d / name).exists():
            return d / name
    return None


@st.cache_data
def load():
    p = find("metrics.json")
    m = json.loads(p.read_text(encoding="utf-8")) if p else EMBEDDED_METRICS
    frames = []
    for name in ["my_gan_synthetic_fraud.csv", "noon_synthetic_fraud.csv", "real_fraud_referee.csv"]:
        f = find(name)
        frames.append(pd.read_csv(f) if f else None)
    return (m, *frames)


M, MY_FRAUD, NOON_FRAUD, REAL_FRAUD = load()


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#52514e")


def cm_plot(cm, title, cmap="Blues"):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap=cmap)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{labels[i][j]}\n{v:,}", ha="center", va="center", fontsize=11,
                    color="white" if v > cm.max() * 0.5 else "#0b0b0b")
    ax.set_xticks([0, 1], ["legit", "fraud"]); ax.set_yticks([0, 1], ["legit", "fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def metrics_table(block):
    rows = []
    for name, v in block.items():
        cm = v.get("cm")
        fp = cm[0][1] if cm else v.get("false_positives")
        fn = cm[1][0] if cm else None
        rows.append({"Model": name, "Recall (fraud caught)": v["recall"], "Precision": v["precision"],
                     "F1": v["f1"], "Accuracy": v.get("accuracy"), "False positives": fp, "Missed fraud": fn})
    return pd.DataFrame(rows).set_index("Model")


def grouped_metric_bars(df, colors, title):
    metrics = ["Recall (fraud caught)", "Precision", "F1"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics)); n = len(df); w = 0.8 / n
    for i, (name, c) in enumerate(zip(df.index, colors)):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, df.loc[name, metrics].astype(float), w - 0.02, color=c, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, metrics); ax.set_ylim(0, 1.1); ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    tidy(ax); fig.tight_layout()
    return fig


def fp_bars(df, colors, title):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fp = df["False positives"].astype(int)
    bars = ax.barh(df.index[::-1], fp[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%d", fontsize=9, padding=3)
    ax.set_xlim(0, max(fp.max() * 1.18, 5)); ax.set_title(title, fontsize=11); tidy(ax); fig.tight_layout()
    return fig


def _phase2():
    show_real = REAL_FRAUD is not None and st.checkbox("Overlay the real fraud from the referee set (65 rows)", value=False)
    cont = ["order_amount", "account_age_days", "num_orders_last_7d", "delivery_distance_km", "hour_of_day", "prev_refunds"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    for ax, c in zip(axes.ravel(), cont):
        hi = np.percentile(np.concatenate([MY_FRAUD[c], NOON_FRAUD[c]]), 99)
        bins = np.linspace(0, hi, 30)
        ax.hist(MY_FRAUD[c].clip(upper=hi), bins=bins, density=True, alpha=0.55, color=MINE, label="Mine")
        ax.hist(NOON_FRAUD[c].clip(upper=hi), bins=bins, density=True, alpha=0.55, color=NOON, label="Noon")
        if show_real:
            ax.hist(REAL_FRAUD[c].clip(upper=hi), bins=bins, density=True, histtype="step", lw=2,
                    color="#0b0b0b", label="Real fraud")
        ax.set_title(c, fontsize=10); tidy(ax)
    axes[0, 0].legend(frameon=False); fig.tight_layout()
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    bin_cols = ["device_change", "location_mismatch", "discount_applied"]
    rates = pd.DataFrame({"Mine": MY_FRAUD[bin_cols].mean(), "Noon": NOON_FRAUD[bin_cols].mean()})
    if show_real:
        rates["Real fraud"] = REAL_FRAUD[bin_cols].mean()
    fig, ax = plt.subplots(figsize=(6, 3.5)); x = np.arange(len(rates)); n = rates.shape[1]; w = 0.8 / n
    for i, (col, colr) in enumerate(zip(rates.columns, [MINE, NOON, "#52514e"])):
        b = ax.bar(x + (i - (n - 1) / 2) * w, rates[col], w - 0.02, color=colr, label=col)
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, rates.index); ax.set_ylim(0, 1); ax.set_title("Share of fraud rows with flag = 1", fontsize=10)
    ax.legend(frameon=False, fontsize=8); tidy(ax); fig.tight_layout(); c1.pyplot(fig)

    ld = M["lie_detector"]; imp = pd.Series(ld["importances"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5)); ax.barh(imp.index, imp.values, color=MINE)
    ax.set_title(f"Lie detector: {ld['accuracy']:.2f} accuracy — what gives my fraud away", fontsize=10)
    tidy(ax); fig.tight_layout(); c2.pyplot(fig)

    st.markdown("**Fraud proportion in each dataset**")
    prop = pd.DataFrame(M["proportions"], index=["Rows", "Fraud fraction"]).T
    prop["Rows"] = prop["Rows"].astype(int).map("{:,}".format)
    prop["Fraud fraction"] = prop["Fraud fraction"].map("{:.2%}".format)
    st.dataframe(prop, use_container_width=True)
    st.caption("0.50 lie-detector accuracy would mean my fraud is indistinguishable from Noon's; 0.99 means very different.")


# ---------------------------------------------------------------- header
st.title("📊 Noon Fraud Lab — Results Dashboard")
st.caption("Seed → GAN → classifier, compared against Noon's GAN and scored on real referee data "
           "(20,000 orders, 65 frauds, 0.33% fraud rate).")

ref = M["referee"]
mine_after, noon_m, mine_before = ref["Mine — after (seed + my GAN)"], ref["Noon (real + Noon GAN)"], ref["Mine — before (seed only)"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("My final model — real fraud caught", f"{mine_after['recall']:.0%}",
          f"{(mine_after['recall'] - mine_before['recall']) * 100:+.0f} pts vs. before GAN")
k2.metric("Noon — real fraud caught", f"{noon_m['recall']:.0%}")
k3.metric("My false positives (real data)", f"{mine_after['cm'][0][1]:,}",
          f"{mine_after['cm'][0][1] - mine_before['cm'][0][1]:+,} vs. before GAN", delta_color="inverse")
k4.metric("Noon false positives", f"{noon_m['cm'][0][1]:,}")

tab1, tab2, tab3, tab4 = st.tabs(["Before vs. after GAN", "Real data: me vs. Noon",
                                  "My GAN vs. Noon's GAN", "Assumptions check"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.subheader("Phase 1 — before vs. after adding GAN fraud (my seed's held-out test set)")
    st.write("Same 1,000-row test set (80 fraud). **Before** = only 12 real fraud rows in training (extreme scarcity). "
             "**After** = the same 12 rows + 300 rows from my GAN trained on them. The full 320-row model is the ceiling.")
    seed_df = metrics_table(M["seed_test"])
    c1, c2 = st.columns([3, 2])
    c1.pyplot(grouped_metric_bars(seed_df, [GREY, MINE_LIGHT, MINE], "Seed test set — higher is better"))
    c2.pyplot(fp_bars(seed_df, [GREY, MINE_LIGHT, MINE], "False positives (lower is better)"))
    cols = st.columns(3)
    for col, (name, v), cmap in zip(cols, M["seed_test"].items(), ["Greys", "Blues", "Blues"]):
        col.pyplot(cm_plot(v["cm"], name, cmap))
    st.dataframe(seed_df.style.format({c: "{:.3f}" for c in ["Recall (fraud caught)", "Precision", "F1", "Accuracy"]}),
                 use_container_width=True)
    st.info("Trained on only 12 rows, my GAN blurred the fraud signal toward the average (lower order amounts, "
            "no card payments), so recall **fell** from 0.38 to 0.21. Accuracy barely moved — which is why "
            "accuracy alone is misleading here.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Phase 3 — real referee data: my classifier vs. Noon's")
    ref_df = metrics_table(ref).loc[["Mine — before (seed only)", "Mine — after (seed + my GAN)",
                                     "Noon real-only (reported)", "Noon (real + Noon GAN)"]]
    colors = [MINE_LIGHT, MINE, NOON_LIGHT, NOON]
    c1, c2 = st.columns([3, 2])
    c1.pyplot(grouped_metric_bars(ref_df, colors, "Real data — higher is better"))
    c2.pyplot(fp_bars(ref_df, colors, "Innocent orders flagged (lower is better)"))
    cols = st.columns(3)
    for col, name, cmap in zip(cols, ["Mine — before (seed only)", "Mine — after (seed + my GAN)", "Noon (real + Noon GAN)"],
                               ["Blues", "Blues", "Oranges"]):
        col.pyplot(cm_plot(ref[name]["cm"], name, cmap))
    st.dataframe(ref_df.style.format({c: "{:.3f}" for c in ["Recall (fraud caught)", "Precision", "F1"]}, na_rep="—"),
                 use_container_width=True)
    st.caption("'Noon real-only' is from Noon's benchmark (numbers only, no model), so it has no confusion matrix.")
    st.warning("My GAN lifted real-data recall from 0.46 to 0.60, but false alarms rose ten-fold (174 → 1,776, about 9% "
               "of legit orders). Noon's GAN learned from real fraud; mine learned from my assumptions.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Phase 2 — my synthetic fraud vs. Noon's (3,000 rows each)")
    if MY_FRAUD is None or NOON_FRAUD is None:
        st.error("Distribution charts need `my_gan_synthetic_fraud.csv` and `noon_synthetic_fraud.csv` "
                 "in a `data/` folder (or next to dashboard.py). Add them to the repo and reboot the app.")
    else:
        _phase2()
# ---------------------------------------------------------------- tab 4
with tab4:
    st.subheader("My Phase 1 assumptions vs. the real data (column means)")
    chk = pd.DataFrame(M["assumption_check"])
    st.dataframe(chk.style.format("{:.2f}"), use_container_width=True)
    col = st.selectbox("Compare one column", chk.index.tolist(), index=0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    pal = [MINE_LIGHT, "#52514e", MINE, MINE, NOON, "#0b0b0b"]
    b = ax.bar(chk.columns, chk.loc[col], color=pal)
    ax.bar_label(b, fmt="%.2f", fontsize=8); ax.set_title(f"{col} — mean by dataset", fontsize=10)
    plt.setp(ax.get_xticklabels(), fontsize=8); tidy(ax); fig.tight_layout(); st.pyplot(fig)
    st.markdown("- **Right:** device change and location mismatch are strong fraud flags.\n"
                "- **Wrong about legit:** real customers spend more (~100 AED), order more (~3/week) and have newer accounts.\n"
                "- **Wrong about fraud:** real fraud is a set of small shifts, not extreme archetypes "
                "(no 30–80 km deliveries or 5–20 prior refunds).")
