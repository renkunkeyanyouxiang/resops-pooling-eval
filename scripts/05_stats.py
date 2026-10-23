# -*- coding: utf-8 -*-
"""ext03 statistical inference: paired shared-local skill differences with
HUC4 cluster-equal-weight multinomial bootstrap (study convention, seed 20261011),
cluster sign-flip tests, and Holm correction within declared families.

Combines ext03 arms with frozen local/reference arms. Primary families:
  F1 h1 A: shared_tuned - local_tuned_ref(local500)
  F2 h1 A: similar/random donors - full pooling (shared_tuned)
Everything else secondary/exploratory.
"""
from pathlib import Path
import io, json, zipfile
import numpy as np, pandas as pd

ROOT = Path(r'I:\stremflow')
EX = ROOT / 'extensions' / 'ext03_fair_pooling'
SEED = 20261011
NB = 4000
NSF = 20000
rng = np.random.default_rng(SEED)

# HUC4 clusters
huc = pd.read_csv(ROOT / '_review_tmp' / 'audit' / 'completion_20261001' / 'basin_huc4_huc2.csv').set_index('DAM_ID')

p0 = pd.read_csv(EX / 'ext03_p0_results.csv')
p1 = pd.read_csv(EX / 'ext03_p1_results.csv')

# frozen reference arms from released wide forecasts (already audited) -> per-reservoir skill
z = zipfile.ZipFile(str(ROOT / '水电入流预测_全部计算_results_20261001.zip'))
rel = pd.read_parquet(io.BytesIO(z.read('hydro_project/completion_20261001/all_completion_forecasts_wide.parquet')))
rel = rel[rel.cohort.eq('primary_dual_catalog')]

def frozen_skill_arm(model_col, task):
    out = []
    for did, g in rel[rel.task.eq(task)].groupby('DAM_ID'):
        g = g.set_index('origin')
        y = g.y.to_numpy()
        base = np.abs(y - g.baseline.to_numpy()).mean()
        out.append(dict(task=task, arm=model_col, DAM_ID=int(did),
                        mae_skill=1 - np.abs(y - g[model_col].to_numpy()).mean() / base,
                        HUC4=huc.loc[int(did), 'HUC4']))
    return pd.DataFrame(out)

# local reference per task (frozen lgb_local_A) + ridge frozen
ref = pd.concat([frozen_skill_arm('lgb_local_A', t) for t in ['h1', 'h3', 'h7', 'v7']])
ridge_s = pd.concat([frozen_skill_arm('ridge_shared_A', t) for t in ['h1', 'h3', 'h7', 'v7']])
ridge_l = pd.concat([frozen_skill_arm('ridge_local_A', t) for t in ['h1', 'h3', 'h7', 'v7']])
for d in (ref, ridge_s, ridge_l):
    d['arm'] = d['arm'].replace({'lgb_local_A': 'local500_ref', 'ridge_shared_A': 'ridge_shared_ref',
                                  'ridge_local_A': 'ridge_local_ref'})

allres = pd.concat([p0, p1, ref, ridge_s, ridge_l], ignore_index=True)

def cluster_stats(delta_by_dam, h4):
    """cluster-equal-weight multinomial bootstrap CI + cluster sign-flip p."""
    d = pd.DataFrame({'v': delta_by_dam, 'c': h4})
    agg = d.groupby('c').v.agg(['sum', 'count'])
    w = rng.multinomial(len(agg), np.full(len(agg), 1 / len(agg)), size=NB)
    draws = (w @ agg['sum'].to_numpy()) / (w @ agg['count'].to_numpy())
    lo, hi = np.quantile(draws, [.025, .975])
    # sign flip: flip whole clusters
    cv = (d.groupby('c').v.sum() / d.groupby('c').v.count()).to_numpy()
    obs = cv.mean()
    ge = 1
    for _ in range(NSF):
        s = rng.choice([-1., 1.], size=len(cv))
        if abs((s * cv).mean()) >= abs(obs) - 1e-12:
            ge += 1
    p = ge / (NSF + 1)
    return obs, lo, hi, p

def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    run = 0.
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(run, 1.)
    return adj

# -------- paired contrasts --------
def skill_pivot(df, arm):
    return df[df.arm.eq(arm)].set_index(['task', 'DAM_ID']).mae_skill

contrasts = [
    # (name, task, arm_minus, arm_base, family)
    ('shared_tuned-local', 'h1', 'shared_tuned', 'local500_ref', 'F1'),
    ('shared_orig500-local', 'h1', 'shared_orig500', 'local500_ref', 'F1'),
    ('similar10-shared_tuned', 'h1', 'similar10', 'shared_tuned', 'F2'),
    ('random10-shared_tuned', 'h1', 'random10', 'shared_tuned', 'F2'),
    ('similar25-shared_tuned', 'h1', 'similar25', 'shared_tuned', 'F2'),
    ('random25-shared_tuned', 'h1', 'random25', 'shared_tuned', 'F2'),
    ('lad_shared-ridge_shared', 'h1', 'lad_shared', 'ridge_shared_ref', 'F3'),
    ('huber_shared-ridge_shared', 'h1', 'huber_shared', 'ridge_shared_ref', 'F3'),
]
# also all 4 tasks for tuned-local as secondary
for t in ['h3', 'h7', 'v7']:
    contrasts.append((f'shared_tuned-local', t, 'shared_tuned', 'local500_ref', 'S1'))
for t in ['h3', 'h7', 'v7']:
    contrasts.append((f'lad_shared-ridge_shared', t, 'lad_shared', 'ridge_shared_ref', 'S2'))

rows = []
for name, task, a, b, fam in contrasts:
    try:
        pa = skill_pivot(allres, a).loc[task]
        pb = skill_pivot(allres, b).loc[task]
    except KeyError:
        continue
    common = pa.index.intersection(pb.index)
    pa, pb = pa.loc[common], pb.loc[common]
    delta = (pa - pb)
    h4 = huc.loc[[int(x) for x in common], 'HUC4'].to_numpy()
    obs, lo, hi, p = cluster_stats(delta.to_numpy() * 100, h4)
    rows.append(dict(family=fam, contrast=name, task=task, n=len(common),
                     delta_skill_pp=round(obs, 2), ci95=f'[{lo:.2f}, {hi:.2f}]',
                     signflip_p=round(p, 4)))
res = pd.DataFrame(rows)
for fam, g in res.groupby('family'):
    res.loc[g.index, 'holm_p'] = holm(g.signflip_p.to_numpy()).round(3)
res.to_csv(EX / 'ext03_contrasts.csv', index=False)
print(res.to_string(index=False))
