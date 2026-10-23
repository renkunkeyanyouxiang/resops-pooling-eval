# -*- coding: utf-8 -*-
"""S4 Panel C, the definition used in manuscript v8.13.

Frozen membership: the released high_flow_n test days with the largest realized targets.
A hit on that set is a forecast above the lowest realized target in the set (Panel B).

Reconstructed membership: test days with realized target above the training-period q0.90
recomputed from the raw archive (ext03_threshold_audit.csv). A hit is a forecast above that quantile.

This is not the earlier 22_v87 comparison, which scored both day sets against the reconstructed quantile
and therefore reported hit-rate changes of about 0.1 pp.
"""
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"I:\stremflow")
EX = ROOT / "extensions" / "ext03_fair_pooling"
z = zipfile.ZipFile(ROOT / "水电入流预测_全部计算_results_20261001.zip")
rel = pd.read_parquet(io.BytesIO(z.read("hydro_project/completion_20261001/all_completion_forecasts_wide.parquet")))
rel = rel[rel.cohort.eq("primary_dual_catalog") & rel.task.eq("h1")]
met = pd.read_csv(io.BytesIO(z.read("hydro_project/completion_20261001/final_reservoir_metrics.csv")))
hfn = met[met.cohort.eq("primary_dual_catalog") & met.model.eq("baseline")].set_index(["DAM_ID", "task"]).high_flow_n
thr = pd.read_csv(EX / "ext03_threshold_audit.csv")
thr = thr[thr.task.eq("h1")].set_index("DAM_ID")
models = [("Local LightGBM A", "lgb_local_A"), ("Shared LightGBM A", "lgb_shared_A"), ("Locked baseline", "baseline")]
rows = []
for did, g in rel.groupby("DAM_ID"):
    g = g.sort_values("y", ascending=False)
    n = int(hfn.loc[(int(did), "h1")])
    hf = g.iloc[:n]
    t_s4 = float(hf.y.min())
    t = float(thr.loc[int(did), "thr"])
    rec = g[g.y > t]
    non = g[g.y <= t]
    recs = {}
    for name, col in models:
        recs[name] = dict(
            s4_rel=float((hf[col] - hf.y).mean() / hf.y.mean()),
            rec_rel=float((rec[col] - rec.y).mean() / rec.y.mean()) if len(rec) else np.nan,
            s4_hit=float((hf[col] > t_s4).mean()),
            rec_hit=float((rec[col] > t).mean()) if len(rec) else np.nan,
            rec_fa=float((non[col] > t).mean()) if len(non) else np.nan,
        )
    rows.append(recs)

out = []
for name, _ in models:
    s4_rel = 100 * np.mean([r[name]["s4_rel"] for r in rows])
    rec_rel = 100 * np.mean([r[name]["rec_rel"] for r in rows])
    s4_hit = 100 * np.mean([r[name]["s4_hit"] for r in rows])
    rec_hit = 100 * np.mean([r[name]["rec_hit"] for r in rows])
    rec_fa = 100 * np.mean([r[name]["rec_fa"] for r in rows])
    out.append(dict(
        task="h1", model=name, n_reservoirs=len(rows),
        frozen_fixed_count_rel_bias_pct=round(s4_rel, 2),
        reconstructed_q90_rel_bias_pct=round(rec_rel, 2),
        delta_rel_bias_pp=round(rec_rel - s4_rel, 2),
        frozen_fixed_count_hit_pct=round(s4_hit, 2),
        reconstructed_q90_hit_pct=round(rec_hit, 2),
        delta_hit_pp=round(rec_hit - s4_hit, 2),
        reconstructed_q90_false_alarm_pct=round(rec_fa, 2),
        definition="S4 Panel C / manuscript v8.13",
    ))
df = pd.DataFrame(out)
dest = EX / "ext03_s4_panelc.csv"
df.to_csv(dest, index=False)
print(df.to_string(index=False))
print("wrote", dest)
