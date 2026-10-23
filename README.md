# ResOpsUS pooling-evaluation pipeline

Code for the manuscript *"When does cross-reservoir pooling help or harm reported
inflow forecasting? Evidence from 50 U.S. reservoirs"* (Computers & Geosciences).

The pipeline compares per-reservoir and pooled LightGBM / Ridge models for
short-term forecasts of agency-reported reservoir inflow (ResOpsUS, 50
dual-catalog hydropurpose reservoirs) under a frozen, leakage-audited protocol
with locked persistence and climatology baselines and cluster-aware inference.

## Repository structure

```
quick_test.py            Quick test: verifies headline statistics from bundled outputs
requirements.txt         Pinned dependencies (Python 3.12)
scripts/                 Analysis scripts (05-24 series, see below)
outputs/                 Released result files (CSV) supporting the manuscript numbers
sources/                 Cited source files (Table 1/2 notes, figure captions):
                         final_reservoir_metrics.csv, final_clustered_model_summary.csv,
                         final_clustered_comparisons.csv, cohort65.csv, metadata/splits65.json,
                         ext02_aux_increment_audit.csv
protocols/               ext01-ext03 protocols logged before execution
```

## Requirements

Python 3.12 with `lightgbm==4.7.0`, `numpy==2.5.3`, `pandas==3.0.6`:

```
pip install -r requirements.txt
```

## Quick test

```
python quick_test.py
```

The test reads bundled output files and asserts the headline paired statistic
of the fair-tuning experiment (1-day tuned-pooled vs. frozen-local difference of
-0.39 pp, 95% CI [-2.76, 1.36], Holm p = 0.861). Expected output: `PASS`.
No training is performed; the test runs in seconds.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/05_stats.py` | Clustered inference for the ext03 contrasts (Table S9) |
| `scripts/07_threshold_audit.py` | High-flow threshold reconstruction audit |
| `scripts/12_pretrain_ft.py` | Pretrain-fine-tune experiment (retaining; not needed for checks) |
| `scripts/15_local_tuned.py` | Fairness-control local tuning (retaining) |
| `scripts/20_drift_audit.py` | Rerun drift audit (0.32 / +0.45 pp quantities) |
| `scripts/21_reservoir_weighted_inference.py` | Reservoir-equal-weight inference |
| `scripts/23_s4_panelc.py` | Table S4 Panel C sensitivity recomputation |
| `scripts/24_estimator_sources.py` | Estimator-source reconciliation for Table S9 |
| `scripts/hf_diagnostics.py` | High-flow signed-bias and event-rate diagnostics |

## Mapping published numbers to files

| Published number | Authoritative file |
|---|---|
| 1-day tuned-pooled vs frozen-local, -0.39 pp [-2.76, 1.36], Holm 0.861 | `outputs/ext03_tuned_vs_local_two_estimators.csv` |
| Table S9 HUC4-equal intervals | `outputs/ext03_contrasts.csv` (from `scripts/05_stats.py`) |
| Pretrain-fine-tune vs matched local continuation (Fig. 6) | `outputs/ext03_pretrain_vs_local_ft.csv` |
| High-flow Panel C hit rates | `outputs/ext03_s4_panelc.csv` |
| Drift quantities 0.32 pp / +0.45 pp | `outputs/ext03_drift_audit.csv` |
| Table 1 and Figs. 2-4 (frozen primary results) | `sources/final_clustered_model_summary.csv`, `sources/final_reservoir_metrics.csv` |
| Table 2 (paired differences, HUC4 cluster) | `sources/final_clustered_comparisons.csv` |
| Fig. 1 (cohort design) | `sources/cohort65.csv`, `sources/metadata/splits65.json` |
| Table 1 auxiliary-feature increment | `sources/ext02_aux_increment_audit.csv` |

`outputs/superseded/` holds a superseded sensitivity file kept for provenance only.

## Data

The input datasets are public: ResOpsUS (Steyaert et al., 2022), Global Dam Watch
v1.0 (Lehner et al., 2024), and the U.S. National Inventory of Dams. The frozen
forecast archive (`results_20261001.zip`, 142 MB) exceeds the repository
file-size limit and is deposited at [ZENODO DOI].

## License

MIT (see LICENSE).
