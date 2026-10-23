# -*- coding: utf-8 -*-
"""Audit: recompute training-period q0.90 high-flow thresholds from RAW daily data
(methods 2.8 definition) for all 50 primary reservoirs x 4 targets; verify that the
number of test-period days above the threshold equals the frozen high_flow_n counts.
If exact, the v4.1 top-N reconstruction is validated as equivalent."""
from pathlib import Path
import io, json, zipfile
import numpy as np, pandas as pd

ROOT = Path(r'I:\stremflow')
EX = ROOT / 'extensions' / 'ext03_fair_pooling'
WORK = EX / 'work'

table = pd.read_csv(WORK / 'cohort50.csv').set_index('DAM_ID')
splits = {int(k): v for k, v in json.load(open(WORK / 'metadata' / 'splits.json')).items()}

z = zipfile.ZipFile(str(ROOT / '水电入流预测_全部计算_results_20261001.zip'))
rel = pd.read_parquet(io.BytesIO(z.read('hydro_project/completion_20261001/all_completion_forecasts_wide.parquet')))
rel = rel[rel.cohort.eq('primary_dual_catalog')]
met = pd.read_csv(io.BytesIO(z.read('hydro_project/completion_20261001/final_reservoir_metrics.csv')))
hfn = met[met.cohort.eq('primary_dual_catalog') & met.model.eq('baseline')].set_index(['DAM_ID', 'task']).high_flow_n

TASKS = {'h1': ('target_1', 1), 'h3': ('target_3', 3), 'h7': ('target_7', 7), 'v7': ('target_sum7', 7)}
def wy(idx):
    return idx.year + (idx.month >= 10).astype(int)

rows = []
for i in table.index:
    f = pd.read_parquet(WORK / 'prepared' / f'{i}.parquet')
    s = splits[int(i)]
    for task, (col, h) in TASKS.items():
        # realized target series on eligible origins, training years only
        ycol = f[col] / 7 if task == 'v7' else f[col]
        tr_mask = f.eligible_origin & ycol.notna() & f.wy.isin(s['train_years']) & np.isin(wy(f.index + pd.Timedelta(days=h)), s['train_years'])
        train_targets = ycol[tr_mask].dropna()
        thr = float(np.quantile(train_targets.to_numpy(), 0.9))
        if task == 'v7':
            thr *= 0.6048  # released v7 y is in MCM (mean flow * 0.6048)
        # test-period realized targets above threshold
        r_i = rel[(rel.DAM_ID == int(i)) & (rel.task == task)]
        y_test = r_i.set_index('origin').y.to_numpy()
        n_above = int((y_test > thr).sum())
        n_ge = int((y_test >= thr).sum())
        n_frozen = int(hfn.loc[(int(i), task)])
        rows.append(dict(DAM_ID=int(i), task=task, thr=round(thr, 3),
                         n_test_rows=len(y_test), n_above_thr=n_above, n_ge_thr=n_ge,
                         frozen_hf_n=n_frozen,
                         match=(n_above == n_frozen or n_ge == n_frozen),
                         diff=int(n_above - n_frozen)))
df = pd.DataFrame(rows)
ok = df['match'].all()
print('exact count matches:', df['match'].sum(), '/', len(df))
if not ok:
    print(df[~df['match']].to_string(index=False))
df.to_csv(EX / 'ext03_threshold_audit.csv', index=False)
print('saved ext03_threshold_audit.csv; ALL MATCH =', bool(ok))
