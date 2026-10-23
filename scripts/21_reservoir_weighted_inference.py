# -*- coding: utf-8 -*-
"""Reservoir-equal-weight paired inference for the tuned-pooled vs frozen-local
contrast (the §2.8 primary metric), with HUC4-cluster-resampled reservoir-equal
bootstrap CI, alongside the existing HUC4-equal-weight inference. Also the full
local-tuned vs shared-tuned paired table for all four targets."""
from pathlib import Path
import io, json, zipfile, warnings
import numpy as np, pandas as pd
import sys, io as _io
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

ROOT = Path(r'I:\stremflow')
EX = ROOT / 'extensions' / 'ext03_fair_pooling'
SEED = 20261011
rng = np.random.default_rng(SEED)
NB = 4000
NSF = 20000

huc = pd.read_csv(ROOT / '_review_tmp' / 'audit' / 'completion_20261001' / 'basin_huc4_huc2.csv').set_index('DAM_ID')
p0 = pd.read_csv(EX / 'ext03_p0_results.csv')
lt = pd.read_csv(EX / 'ext03_local_tuned.csv')
z = zipfile.ZipFile(str(ROOT / '水电入流预测_全部计算_results_20261001.zip'))
rel = pd.read_parquet(io.BytesIO(z.read('hydro_project/completion_20261001/all_completion_forecasts_wide.parquet')))
rel = rel[rel.cohort.eq('primary_dual_catalog')]

def res_skill(col, task):
    out = {}
    for did, g in rel[rel.task == task].groupby('DAM_ID'):
        g = g.set_index('origin')
        base = np.abs(g.y - g.baseline).mean()
        out[int(did)] = 1 - np.abs(g.y - g[col]).mean() / base
    return pd.Series(out)

def res_eq_boot(delta, h4):
    """Cluster-resampled reservoir-equal-weight CI: sample HUC clusters with
    replacement, then compute the reservoir-equal-weight mean on the resampled rows."""
    d = pd.DataFrame({'v': delta, 'c': h4})
    clusters = d.c.unique()
    groups = {c: d[d.c == c].v.to_numpy() for c in clusters}
    boots = []
    for _ in range(NB):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        vals = np.concatenate([groups[c] for c in chosen])
        boots.append(vals.mean())
    return np.quantile(boots, [.025, .975])

def signflip_p(cv):
    obs = cv.mean()
    ge = 1
    for _ in range(NSF):
        s = rng.choice([-1., 1.], size=len(cv))
        if abs((s * cv).mean()) >= abs(obs) - 1e-12:
            ge += 1
    return ge / (NSF + 1)

def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    run = 0.
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(run, 1.)
    return adj

# ---- 1. Tuned pooled vs frozen local: both estimators with full inference ----
print('=' * 70)
print('Tuned pooled - Frozen local: TWO estimators with matched inference')
print('=' * 70)
rows = []
for task in ['h1', 'h3', 'h7', 'v7']:
    loc = res_skill('lgb_local_A', task)
    tuned = p0[(p0.task == task) & (p0.arm == 'shared_tuned')].set_index('DAM_ID').mae_skill
    common = tuned.index.intersection(loc.index)
    delta = (tuned.loc[common] - loc.loc[common]) * 100
    h4 = huc.loc[[int(x) for x in common], 'HUC4'].to_numpy()
    # reservoir-equal estimator + its own CI
    res_pt = delta.mean()
    res_ci = res_eq_boot(delta.to_numpy(), h4)
    # HUC4-equal estimator + its own CI and sign-flip
    grp = pd.DataFrame({'v': delta.to_numpy(), 'c': h4}).groupby('c').v.mean()
    grp_pt = grp.mean()
    grp_boot = rng.choice(grp.to_numpy(), size=(NB, len(grp)), replace=True).mean(axis=1)
    grp_ci = np.quantile(grp_boot, [.025, .975])
    sf = signflip_p(grp.to_numpy())
    rows.append(dict(task=task,
                     res_eq_pt=round(res_pt, 2), res_eq_ci=f'[{res_ci[0]:.2f}, {res_ci[1]:.2f}]',
                     huc_eq_pt=round(grp_pt, 2), huc_eq_ci=f'[{grp_ci[0]:.2f}, {grp_ci[1]:.2f}]',
                     signflip_p=round(sf, 4)))
    print(f'  {task}: res-eq {res_pt:+.2f} [{res_ci[0]:.2f},{res_ci[1]:.2f}]  '
          f'| huc-eq {grp_pt:+.2f} [{grp_ci[0]:.2f},{grp_ci[1]:.2f}]  signflip p={sf:.4f}')
pd.DataFrame(rows).to_csv(EX / 'ext03_tuned_vs_local_two_estimators.csv', index=False)

# ---- 2. Local-tuned vs Shared-tuned: all four targets with Holm ----
print('\n' + '=' * 70)
print('Local-tuned vs Shared-tuned: all four targets with Holm')
print('=' * 70)
rows2 = []
for task in ['h1', 'h3', 'h7', 'v7']:
    a = lt[(lt.task == task) & (lt.arm == 'local_tuned')].set_index('DAM_ID').mae_skill
    b = p0[(p0.task == task) & (p0.arm == 'shared_tuned')].set_index('DAM_ID').mae_skill
    common = a.index.intersection(b.index)
    delta = (a.loc[common] - b.loc[common]) * 100
    h4 = huc.loc[[int(x) for x in common], 'HUC4'].to_numpy()
    grp = pd.DataFrame({'v': delta.to_numpy(), 'c': h4}).groupby('c').v.mean()
    pt = grp.mean()
    grp_boot = rng.choice(grp.to_numpy(), size=(NB, len(grp)), replace=True).mean(axis=1)
    ci = np.quantile(grp_boot, [.025, .975])
    p = signflip_p(grp.to_numpy())
    rows2.append(dict(task=task, delta_pp=round(pt, 2), ci95=f'[{ci[0]:.2f}, {ci[1]:.2f}]', p_raw=round(p, 4)))
holm2 = holm([r['p_raw'] for r in rows2])
for i, r in enumerate(rows2):
    r['holm_p'] = f'{holm2[i]:.3f}'
for r in rows2:
    print(f"  {r['task']}: {r['delta_pp']:+.2f} {r['ci95']}  raw p={r['p_raw']:.4f}  Holm p={r['holm_p']}")
pd.DataFrame(rows2).to_csv(EX / 'ext03_local_vs_shared_tuned.csv', index=False)
print('\nsaved ext03_tuned_vs_local_two_estimators.csv and ext03_local_vs_shared_tuned.csv')
