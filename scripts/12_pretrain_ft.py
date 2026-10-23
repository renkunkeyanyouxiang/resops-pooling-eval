# -*- coding: utf-8 -*-
"""E3: pretrain-then-finetune. Pooled tuned model (F1 grid) as init; fine-tune on each
target's own training rows (lr=0.01, max 300 rounds, early stop on target calibration year).
Primary contrast: finetune - local500 (frozen released), paired.
Crash-safe: chunked by (task, cutoff); merge-saves ext03_e3_results.csv.
Usage: python 12_pretrain_ft.py [h1|h3|h7|v7 ...] [2011-09-30|...]
"""
from pathlib import Path
import io, json, sys, time, zipfile, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')

ROOT = Path(r'I:\stremflow')
EX = ROOT / 'extensions' / 'ext03_fair_pooling'
WORK = EX / 'work'
SEED = 20261011
LAGS = [1, 2, 3, 7, 14, 30]
A = [f'q_lag{k}' for k in LAGS] + ['q_mean7', 'q_mean30', 'doy_sin', 'doy_cos']
TASKS = {'h1': ('target_1', 1, 1.), 'h3': ('target_3', 3, 1.),
         'h7': ('target_7', 7, 1.), 'v7': ('target_sum7', 7, .6048)}
GRID = [dict(num_leaves=nl, min_child_samples=mcl, learning_rate=.05, reg_lambda=1., n_estimators=1000)
        for nl in [15, 31, 63, 127] for mcl in [20, 50, 100, 200]]
BASE = dict(objective='regression_l1', max_bin=127, verbosity=-1, n_jobs=2,
            random_state=SEED, deterministic=True, force_col_wise=True)
FT_PARAMS = dict(objective='regression_l1', n_estimators=300, learning_rate=.01,
                 max_bin=127, verbosity=-1, n_jobs=2, random_state=SEED,
                 deterministic=True, force_col_wise=True)

def wy(idx):
    return idx.year + (idx.month >= 10).astype(int)

table = pd.read_csv(WORK / 'cohort50.csv').set_index('DAM_ID')
splits = {int(k): v for k, v in json.load(open(WORK / 'metadata' / 'splits.json')).items()}
DATA = {i: pd.read_parquet(WORK / 'prepared' / f'{i}.parquet') for i in table.index}
HUC = pd.read_csv(ROOT / '_review_tmp' / 'audit' / 'completion_20261001' / 'basin_huc4_huc2.csv').set_index('DAM_ID')
z = zipfile.ZipFile(str(ROOT / '水电入流预测_全部计算_results_20261001.zip'))
rel = pd.read_parquet(io.BytesIO(z.read('hydro_project/completion_20261001/all_completion_forecasts_wide.parquet')))
rel = rel[rel.cohort.eq('primary_dual_catalog')]

def tgt(f, task):
    col = f[TASKS[task][0]]
    return col / 7 if task == 'v7' else col

def fold_mask(f, s, col, h, kind):
    years = s['train_years'] if kind == 'train' else [s['cal_year']] if kind == 'cal' else s['test_years']
    return f.eligible_origin & f[col].notna() & f.wy.isin(years) & np.isin(wy(f.index + pd.Timedelta(days=h)), years)

def scale_for(i, cutoff=None):
    f = DATA[i]; s = splits[i]
    take = np.isin(f.wy, s['train_years'])
    if cutoff:
        take &= f.index <= pd.Timestamp(cutoff)
    a = f.loc[take, 'q_lag1'].dropna().to_numpy()
    med = float(np.median(a)); q = np.quantile(a, [.25, .75]); sc = float(q[1] - q[0])
    return med, (sc if sc > 1e-8 else max(float(np.std(a)), 1.))

def baseX(f, row, med, sc):
    x = f[A].copy()
    for c in x:
        x[c] = (x[c] - med) / sc
    x['log_capacity'] = np.log1p(max(float(row.CAP_MCM), 0))
    x['log_catchment'] = np.log1p(max(float(row.CATCH_SKM), 0))
    return x.astype('float32')

def localX(f, row, med, sc):
    x = f[A].copy()
    for c in x:
        x[c] = (x[c] - med) / sc
    # fine-tune uses the local (non-shared) feature space: no statics, matching local scheme
    return x.astype('float32')

def merge_save(rows):
    _part = pd.DataFrame(rows)
    _p = EX / 'ext03_e3_results.csv'
    if _p.exists():
        _old = pd.read_csv(_p)
        key_old = pd.MultiIndex.from_arrays([_old.task, _old.DAM_ID])
        key_new = pd.MultiIndex.from_arrays([_part.task, _part.DAM_ID])
        _part = pd.concat([_old[~key_old.isin(key_new)], _part], ignore_index=True)
    _part.to_csv(_p, index=False)
    print(f'[checkpoint] e3 rows: {len(_part)}', flush=True)

def main():
    _tasks = [a for a in sys.argv[1:] if a in TASKS] or list(TASKS)
    _cuts = [a for a in sys.argv[1:] if a.startswith('20')]
    cutoffs = sorted({s['cutoff'] for s in splits.values()})
    _run = [c for c in cutoffs if (not _cuts or c in _cuts)]
    for task in _tasks:
        col, h, factor = TASKS[task]
        rows = []
        for cutoff in _run:
            t0 = time.time()
            targets = [i for i, s in splits.items() if s['cutoff'] == cutoff]
            scs = {i: scale_for(i, cutoff) for i in table.index}
            xs, ys, vs, vy = [], [], [], []
            for i, r in table.iterrows():
                f = DATA[i]; s = splits[i]; med, sc = scs[i]
                tr = f.loc[fold_mask(f, s, col, h, 'train') & (f.index + pd.Timedelta(days=h) <= pd.Timestamp(cutoff))]
                xs.append(baseX(tr, r, med, sc)); ys.append((tgt(tr, task) - med) / sc)
                if i in targets:
                    ca = f.loc[fold_mask(f, s, col, h, 'cal')]
                    vs.append(baseX(ca, r, med, sc)); vy.append((tgt(ca, task) - med) / sc)
            X = pd.concat(xs, ignore_index=True); y = pd.concat(ys, ignore_index=True)
            V = pd.concat(vs, ignore_index=True); v = pd.concat(vy, ignore_index=True)
            best = None
            for g in GRID:
                p = dict(BASE); p.update(g)
                m = lgb.LGBMRegressor(**p)
                m.fit(X, y, eval_set=[(V, v)], eval_metric='l1',
                      callbacks=[lgb.early_stopping(30, verbose=False)])
                mae = float(np.abs(m.predict(V) - v.to_numpy()).mean())
                if best is None or mae < best[0]:
                    best = (mae, g, m.booster_)
            pt = dict(BASE); pt.update(best[1])
            pre = lgb.LGBMRegressor(**pt)
            pre.fit(X, y, eval_set=[(V, v)], eval_metric='l1',
                    callbacks=[lgb.early_stopping(30, verbose=False)])
            booster0 = pre.booster_
            for i in targets:
                r = table.loc[i]; f = DATA[i]; s = splits[i]
                med, sc = scale_for(i)  # local scheme scale (no cutoff truncation), matching local500
                tr = f.loc[fold_mask(f, s, col, h, 'train')]
                ca = f.loc[fold_mask(f, s, col, h, 'cal')]
                te = f.loc[fold_mask(f, s, col, h, 'test')]
                # fine-tune in the SHARED feature space (statics included) so init applies
                Xtr = baseX(tr, r, med, sc); Xca = baseX(ca, r, med, sc)
                ytr = (tgt(tr, task) - med) / sc; yca = (tgt(ca, task) - med) / sc
                ft = lgb.LGBMRegressor(**dict(FT_PARAMS, num_leaves=best[1]['num_leaves'],
                                               min_child_samples=best[1]['min_child_samples']))
                ft.fit(Xtr, ytr, init_model=booster0,
                       eval_set=[(Xca, yca)], eval_metric='l1',
                       callbacks=[lgb.early_stopping(30, verbose=False)])
                pred = (ft.predict(baseX(te, r, med, sc)) * sc + med) * factor
                yv = tgt(te, task).to_numpy() * factor
                rr = rel[(rel.DAM_ID == int(i)) & (rel.task == task)].set_index('origin')
                base_mae = float(np.abs(rr.y - rr.baseline).mean())
                mae_m = float(np.mean(np.abs(yv - pred)))
                rows.append(dict(task=task, arm='pretrain_ft', DAM_ID=int(i),
                                 mae_skill=1 - mae_m / base_mae, mae_model=mae_m,
                                 mae_baseline=base_mae, HUC4=HUC.loc[int(i), 'HUC4'],
                                 grid=str(best[1]), ft_best_iter=ft.best_iteration_))
            print(f'{task} cut{cutoff}: pretrain {best[1]} ({time.time()-t0:.0f}s)', flush=True)
        merge_save(rows)

if __name__ == '__main__':
    main()
