# -*- coding: utf-8 -*-
"""High-flow signed bias + event miss/false-alarm from released forecasts (no retraining)."""
import zipfile, io, sys, json
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

zp = r'I:\stremflow\水电入流预测_全部计算_results_20261001.zip'
z = zipfile.ZipFile(zp)
base = 'hydro_project/completion_20261001/'
df = pd.read_parquet(io.BytesIO(z.read(base + 'all_completion_forecasts_wide.parquet')))
df = df[df.cohort.eq('primary_dual_catalog')].copy()
m = pd.read_csv(io.BytesIO(z.read(base + 'final_reservoir_metrics.csv')))
huc = pd.read_csv(io.BytesIO(z.read(base + 'basin_huc4_huc2.csv')))
ids = sorted(df.DAM_ID.unique())
h4 = huc.set_index('DAM_ID').HUC4
print('records:', len(df), 'reservoirs:', len(ids))

# high-flow n per reservoir/task (take baseline model counts; verify identical across models)
ncheck = m[m.cohort.eq('primary_dual_catalog')].pivot_table(index=['DAM_ID','task'], columns='model', values='high_flow_n')
assert ncheck.nunique(axis=1).eq(1).all(), 'high_flow_n differs across models'
hfn = ncheck.iloc[:,0].rename('hf_n')

MODELS = ['baseline','lgb_local_A','lgb_shared_A','capacity_shared_A','ridge_local_A','ridge_shared_A']
rows_bias, rows_event = [], []
rng = np.random.default_rng(20261011)
NBOOT = 4000

for task in ['h1','h3','h7','v7']:
    sub = df[df.task.eq(task)]
    per = {}
    for mod in MODELS:
        recs = []
        for did, g in sub.groupby('DAM_ID'):
            g = g.sort_values('y', ascending=False).reset_index(drop=True)
            n = int(hfn.loc[(did, task)]) if (did, task) in hfn.index else 0
            if n == 0:
                continue
            hf = g.iloc[:n]
            bias = (hf[mod] - hf['y']).mean()
            ybar = hf['y'].mean()
            rec = dict(DAM_ID=did, HUC4=h4.loc[did], bias=bias, rel=bias/ybar if ybar else np.nan,
                       mae=(hf[mod]-hf['y']).abs().mean())
            if task == 'h1':
                thr = hf['y'].min()
                non = g.iloc[n:]
                rec['hit'] = float((hf[mod] > thr).mean())
                rec['fa'] = float((non[mod] > thr).mean()) if len(non) else np.nan
            recs.append(rec)
        per[mod] = pd.DataFrame(recs)

    def clust_ci(frame, col):
        vals = frame.groupby('HUC4')[col].mean().to_numpy()
        pt = frame[col].mean()
        boots = rng.choice(vals, size=(NBOOT, len(vals)), replace=True).mean(axis=1)
        return pt, np.percentile(boots, [2.5, 97.5])

    for mod in MODELS:
        f = per[mod]
        bp, bci = clust_ci(f, 'bias')
        rp, rci = clust_ci(f, 'rel')
        rows_bias.append(dict(task=task, model=mod, reservoirs=len(f),
            mean_bias=round(bp,3), ci_low=round(bci[0],3), ci_high=round(bci[1],3),
            rel_bias_pct=round(100*rp,1), rel_ci=f'[{100*rci[0]:.1f}, {100*rci[1]:.1f}]'))
        if task == 'h1':
            hp, hci = clust_ci(f, 'hit')
            fp_, fci = clust_ci(f, 'fa')
            rows_event.append(dict(model=mod, reservoirs=len(f),
                hit_rate=round(100*hp,1), hit_ci=f'[{100*hci[0]:.1f}, {100*hci[1]:.1f}]',
                miss_rate=round(100*(1-hp),1),
                false_alarm=round(100*fp_,1), fa_ci=f'[{100*fci[0]:.1f}, {100*fci[1]:.1f}]'))

out = dict(bias=rows_bias, event_h1=rows_event)
json.dump(out, open(r'I:\stremflow\_review_tmp\hf_diagnostics.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print()
print('=== signed high-flow bias (m3/s, equal reservoir weight, HUC4 cluster CI) ===')
for r in rows_bias:
    print(f"{r['task']:3s} {r['model']:18s} bias {r['mean_bias']:+8.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]  relative {r['rel_bias_pct']:+6.1f}% {r['rel_ci']}")
print()
print('=== h1 event detection (threshold: training q0.90, reconstructed from released counts) ===')
for r in rows_event:
    print(f"{r['model']:18s} hit {r['hit_rate']:5.1f}% {r['hit_ci']}  miss {r['miss_rate']:5.1f}%  false-alarm {r['false_alarm']:5.1f}% {r['fa_ci']}")
