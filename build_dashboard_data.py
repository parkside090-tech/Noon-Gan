# build_dashboard_data.py — rebuilds ./data from the lab notebook (run from this folder, with the notebook,
# A_eval_helper.py, the seed CSV and Noon's files alongside). Usage: python build_dashboard_data.py noon_fraud_seed_dataset.ipynb
import os; os.makedirs("data", exist_ok=True)
import json, sys, contextlib, io
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.show=lambda *a,**k:None
import numpy as np, pandas as pd
nb=json.load(open(sys.argv[1] if len(sys.argv) > 1 else "noon_fraud_seed_dataset.ipynb")); ns={"__name__":"__main__"}
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code" or i in range(31,46): continue
    s="".join(c["source"])
    if s.strip().startswith("!"): continue
    with contextlib.redirect_stdout(io.StringIO()): exec(s, ns)
    plt.close("all")
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
g=ns.get
def m(y, p):
    cm=confusion_matrix(y,p); return dict(recall=recall_score(y,p), precision=precision_score(y,p,zero_division=0),
        f1=f1_score(y,p,zero_division=0), accuracy=float((cm[0,0]+cm[1,1])/cm.sum()), cm=cm.tolist())
out={}
# Phase 1 in-seed test (Part 2/4 scenarios), from metrics_df + recompute CMs
yt=g("y_test"); Xt=g("X_test")
seed_scen={}
for name,(Xtr,ytr) in g("scenarios").items():
    from sklearn.ensemble import RandomForestClassifier
    p=RandomForestClassifier(random_state=42).fit(Xtr,ytr).predict(Xt); seed_scen[name]=dict(m(yt,p), train_fraud=int(ytr.sum()), train_rows=int(len(ytr)))
out["seed_test"]=seed_scen
ref=g("referee"); _prep=g("_prep")
rt={}
for name,(mod,cols) in {"Mine — before (seed only)":(g("clf_before"),g("cols_before")),
                        "Mine — after (seed + my GAN)":(g("clf_after"),g("cols_after")),
                        "Noon (real + Noon GAN)":(g("NOON_M")["model"],g("NOON_M")["columns"])}.items():
    rt[name]=m(ref["is_fraud"], mod.predict(_prep(ref,columns=cols)))
rt["Noon real-only (reported)"]=dict(recall=0.14,precision=0.90,f1=0.24,accuracy=None,cm=None,false_positives=1)
out["referee"]=rt
out["lie_detector"]=dict(accuracy=float(g("ld_acc")), importances=g("ld_imp").sort_values(ascending=False).round(4).to_dict())
st,after,mine,noon=g("seed_train"),g("after_train"),g("my_fraud"),g("noon_fraud")
out["proportions"]={"My synthetic file":[len(mine),1.0],"Noon synthetic file":[len(noon),1.0],
  "My seed (train)":[len(st),float(st.is_fraud.mean())],"My training set after GAN":[len(after),float(after.is_fraud.mean())],
  "Real referee data":[len(ref),float(ref.is_fraud.mean())]}
NUM=g("NUM"); seed=g("seed")
out["assumption_check"]=pd.DataFrame({"My seed legit":seed[seed.is_fraud==0][NUM].mean(),"Real legit":ref[ref.is_fraud==0][NUM].mean(),
  "My seed fraud":seed[seed.is_fraud==1][NUM].mean(),"My GAN fraud":mine[NUM].mean(),"Noon GAN fraud":noon[NUM].mean(),
  "Real fraud":ref[ref.is_fraud==1][NUM].mean()}).round(3).to_dict()
json.dump(out, open("data/metrics.json","w"), indent=1, ensure_ascii=False)
mine.to_csv("data/my_gan_synthetic_fraud.csv", index=False)
noon.to_csv("data/noon_synthetic_fraud.csv", index=False)
ref[ref.is_fraud==1].to_csv("data/real_fraud_referee.csv", index=False)
print(json.dumps({k:{kk:(round(vv['recall'],3), vv.get('cm')) for kk,vv in v.items()} for k,v in out.items() if k in ('seed_test','referee')}, indent=1, ensure_ascii=False))
