# dashboard.py — Noon Fraud Lab results dashboard
# Run locally:  streamlit run dashboard.py
# Reads precomputed results from ./data (built from the lab notebook), so it needs no model training.
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
MINE_LIGHT, MINE, NOON_LIGHT, NOON = "#86b6ef", "#2a78d6", "#f5b49a", "#eb6834"
GREY = "#9a9994"

st.set_page_config(page_title="Noon Fraud Lab — Results", page_icon="📊", layout="wide")


@st.cache_data
def load():
    m = json.loads((DATA / "metrics.json").read_text(encoding="utf-8"))
    mine = pd.read_csv(DATA / "my_gan_synthetic_fraud.csv")
    noon = pd.read_csv(DATA / "noon_synthetic_fraud.csv")
    real = pd.read_csv(DATA / "real_fraud_referee.csv")
    return m, mine, noon, real


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
    show_real = st.checkbox("Overlay the real fraud from the referee set (65 rows)", value=False)
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
