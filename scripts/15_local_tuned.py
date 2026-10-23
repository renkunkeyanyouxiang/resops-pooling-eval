# -*- coding: utf-8 -*-
"""P0 retraining: Local-tuned (same 16-grid calibration-only selection as shared-tuned).
Also: pretrain-ft local-only control (continue training local500 for same fine-tune budget).
Crash-safe: chunked by task; merge-saves ext03_local_tuned.csv.
Usage: python 15_local_tuned.py [h1|h3|h7|v7 ...]
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
FT = dict(objective='regression_l1', n_estimators=300, learning_rate=.01,
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

def scale_for(i):
    f = DATA[i]; s = splits[i]
    take = np.isin(f.wy, s['train_years'])
    a = f.loc[take, 'q_lag1'].dropna().to_numpy()
    med = float(np.median(a)); q = np.quantile(a, [.25, .75]); sc = float(q[1] - q[0])
    return med, (sc if sc > 1e-8 else max(float(np.std(a)), 1.))

def Xm(f, med, sc):
    x = f[A].copy()
    for c in x:
        x[c] = (x[c] - med) / sc
    return x.astype('float32')

def merge_save(rows):
    _part = pd.DataFrame(rows)
    _p = EX / 'ext03_local_tuned.csv'
    if _p.exists():
        _old = pd.read_csv(_p)
        key_old = pd.MultiIndex.from_arrays([_old.task, _old.DAM_ID])
        key_new = pd.MultiIndex.from_arrays([_part.task, _part.DAM_ID])
        _part = pd.concat([_old[~key_old.isin(key_new)], _part], ignore_index=True)
    _part.to_csv(_p, index=False)
    print(f'[checkpoint] local_tuned rows: {len(_part)}', flush=True)

def main():
    _tasks = [a for a in sys.argv[1:] if a in TASKS] or ['h1']
    for task in _tasks:
        col, h, factor = TASKS[task]
        t0 = time.time()
        rows = []
        for i, r in table.iterrows():
            f = DATA[i]; s = splits[i]; med, sc = scale_for(i)
            tr = f.loc[fold_mask(f, s, col, h, 'train')]
            ca = f.loc[fold_mask(f, s, col, h, 'cal')]
            te = f.loc[fold_mask(f, s, col, h, 'test')]
            Xtr, Xca, Xte = Xm(tr, med, sc), Xm(ca, med, sc), Xm(te, med, sc)
            ytr, yca = (tgt(tr, task) - med) / sc, (tgt(ca, task) - med) / sc

            # --- local-tuned: same 16-grid calibration selection ---
            best = None
            for g in GRID:
                p = dict(BASE); p.update(g)
                m = lgb.LGBMRegressor(**p)
                m.fit(Xtr, ytr, eval_set=[(Xca, yca)], eval_metric='l1',
                      callbacks=[lgb.early_stopping(30, verbose=False)])
                mae = float(np.abs(m.predict(Xca) - yca).mean())
                if best is None or mae < best[0]:
                    best = (mae, g, m)
            pred_tuned = (best[2].predict(Xte) * sc + med) * factor

            # --- local-ft-only control: continue local500 for 300 rounds at lr=0.01 ---
            p500 = dict(BASE); p500.update(dict(num_leaves=31, min_child_samples=50,
                                                 learning_rate=.05, n_estimators=500))
            m500 = lgb.LGBMRegressor(**p500)
            m500.fit(Xtr, ytr, eval_set=[(Xca, yca)], eval_metric='l1',
                     callbacks=[lgb.early_stopping(30, verbose=False)])
            ftl = lgb.LGBMRegressor(**dict(FT, num_leaves=31, min_child_samples=50))
            ftl.fit(Xtr, ytr, init_model=m500.booster_, eval_set=[(Xca, yca)],
                    eval_metric='l1', callbacks=[lgb.early_stopping(30, verbose=False)])
            pred_ftl = (ftl.predict(Xte) * sc + med) * factor

            rr = rel[(rel.DAM_ID == int(i)) & (rel.task == task)].set_index('origin')
            base_mae = float(np.abs(rr.y - rr.baseline).mean())
            y = tgt(te, task).to_numpy() * factor
            for arm, ph in [('local_tuned', pred_tuned), ('local_ft_only', pred_ftl)]:
                mae_m = float(np.mean(np.abs(y - ph)))
                rows.append(dict(task=task, arm=arm, DAM_ID=int(i), n=len(te),
                                 mae_skill=1 - mae_m / base_mae, mae_model=mae_m,
                                 mae_baseline=base_mae, HUC4=HUC.loc[int(i), 'HUC4']))
        print(f'{task} done ({time.time()-t0:.0f}s)', flush=True)
        merge_save(rows)

if __name__ == '__main__':
    main()
