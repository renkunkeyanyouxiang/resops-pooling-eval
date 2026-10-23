# ext03 — Fair-pooling extension (pre-registration draft, protocol to be logged before execution)

Status: **executed** (phase 1 on 2026-10-11; phase 2 on 2026-10-12). See EXECUTION RECORD below. The design text above the execution record is retained as the logged protocol.
Created 2026-10-11 for manuscript "When does cross-reservoir pooling help or harm reported inflow forecasting? Evidence from 50 U.S. reservoirs".
Principle: the 2026-09-29 frozen benchmark and its primary endpoint (1-day, feature set A, Holm-corrected sign-flip) remain the Original benchmark. ext03 is a separate Protocol v2 extension; it cannot retroactively change the frozen primary result and is labelled post-batch/supplementary in the manuscript.

## Rationale

In the Original benchmark local and pooled LightGBM share one hyperparameter set (31 leaves, min 50/leaf, lr 0.05, ≤500 rounds). Pooled fits see ~50× more and far more heterogeneous rows, so the observed −2.53 pp (A) / −3.37 pp (B) deficits cannot distinguish pooling itself from (i) pooled-capacity misspecification, (ii) inability to retain local identity, or (iii) loss-function mismatch. ext03 answers which of these survive a fair comparison.

## Fixed inputs and non-reselection

- Cohort, splits (per-reservoir water years, cutoffs 2011/2014/2015/2016), features A/B, targets h1/h3/h7/V7, issue-day alignment, paired issue-day sets, donor temporal constraints (cutoff + donor's own frozen years), seeds (20261001), inference (4000 HUC4/HUC2 cluster bootstrap; 20,000 sign-flip; Holm within pre-declared families) — all inherited unchanged.
- Hyperparameter candidates are selected with training/calibration data only; test rows are never touched in selection.
- All new models adopt only on the same calibration-MAE rule as the existing capacity arm; adoption decisions logged.

## Arms

| Arm | Training | Hyperparameters | Purpose |
|---|---|---|---|
| local500 / shared500 (existing) | per-reservoir / pooled | frozen local set | Original benchmark |
| local-tuned | per-reservoir | same fixed tuning budget | equal tuning opportunity baseline |
| shared-tuned (P0) | pooled (50 primary; 65 sensitivity) | calibration-only search below | capacity misspecification |
| shared-identity (P1) | pooled + seen-reservoir one-hot | tuned; identity used at inference only for training-seen reservoirs | retain local differences; never used for unseen-reservoir claims |
| pretrain-finetune (P1) | pooled pretrain, then target calibration/training data finetune with fixed early budget | tuned | share + local adaptation |
| similar-k / random-k (P1) | k most similar reservoirs / k size-matched random reservoirs, similarity computed from training-period features only | tuned | hydrological-similarity vs sample-size explanation |
| l1-linear / huber-linear (P1) | identical imputation/standardization scheme as pooled Ridge; L1 (MAE) or Huber loss | — | isolate L2 loss as Ridge-failure cause |
| lag+1/lag+2 sensitivity (P2) | identical to A/B with latest usable observation shifted 1/2 days | frozen | information-delay robustness (retrospective; not operational) |

## shared-tuned search space (fixed budget, time-ordered inner validation)

- num_leaves ∈ {15, 31, 63, 127}
- min_data_in_leaf ∈ {20, 50, 100, 200}
- learning_rate ∈ {0.03, 0.05, 0.10}
- reg_lambda ∈ {0, 1, 10}
- max rounds ∈ {500, 1000, 2000} with calibration early stopping
- Search: at most 24 fixed pseudo-random draws (seeded), identical budget for local-tuned and shared-tuned; inner validation uses the calibration water year only. CPU-time matched comparison reported separately (same wall-clock budget arms) — the two fairness notions are never conflated.

## Pre-declared primary questions (multiplicity capped)

1. Q1 (1-day, A): Δ = skill(shared-tuned) − skill(local-tuned), HUC4 cluster bootstrap; Holm family = {h1} single test for the tuned pair; original shared500-vs-local500 comparison stays in its own frozen family.
2. Q2 (1-day, A): Δ = skill(similar-k best arm) − skill(shared-tuned), testing whether similarity selection beats full pooling; random-k is the required negative control.
Both one-degree-of-freedom primary tests; everything else secondary/exploratory with named families. Equivalence claims require a pre-set margin of ±1 pp MAE skill tested with two one-sided inference, not p > 0.05.

## Reporting

- Per-reservoir ΔMAE in m³/s and scaled by each reservoir's typical inflow, in addition to percentage-point skill.
- Full table in the Supplementary-material pattern of Table S1: delta, unadjusted CI, raw p, Holm p, correction family.
- Any arm that wins only at >1000 rounds or only in set B is reported as such.
- If shared-tuned still underperforms, that is the result: pooling under short-lead history-only inputs is harmful beyond configuration; if it closes the gap, the Original claim is reframed as configuration sensitivity. Neither outcome is pre-required.

---

## EXECUTION RECORD (2026-10-11, after this protocol was logged)

Executed as specified with two recorded deviations:
1. Tuning grid: 16-combo factorial over (num_leaves x min_child_samples) with lr=0.05, reg_lambda=1, max 1000 rounds + calibration early stopping (subset of the logged grid; capacity axis fully covered).
2. LAD control fitted by sparse LP (scipy highs) on a deterministic 50,000-row subsample of pooled training rows (12+1 parameters, seeded per task-cutoff); identical median-impute + train-standardization preprocessing as frozen Ridge. An IRLS implementation was discarded after failing an independent solver cross-check (09_p1_lp_check.py).

Pipeline validation: rerunning the frozen local/shared 500-round configuration reproduces the released equal-weight skills within 0.32 pp on all four targets (02_validate_pipeline.py).

Primary results (ext03_contrasts.csv; HUC4 cluster-equal-weight bootstrap, 20,000 cluster sign-flip, Holm within family):
- F1 shared_tuned - local: h1 -0.21 pp [-2.73, 1.39], Holm p=0.864 (original-config contrast -2.28 pp [-4.71, -0.20], Holm p=0.105); secondary h3 +0.87, h7 +1.42, v7 +1.92 (Holm p >= 0.136).
- F2 similar10 - shared_tuned: -18.71 pp; random10 - shared_tuned: -28.60 pp (similar beats random by ~9.9 pp, Holm p<0.001); similar25 -11.60 vs random25 -13.50.
- F3 lad_shared - ridge_shared: +29.85 pp at h1 (Holm p<0.001); +24.2/+16.6/+26.6 at h3/h7/v7.

Status: post-batch extension analyses; the frozen primary endpoint is unchanged. Raw outputs: ext03_p0_results.csv (1,200 rows), ext03_p1_results.csv (200 rows), ext03_p2_lag_results.csv, ext03_contrasts.csv, ext03_threshold_audit.csv.

---

## PHASE-2 PRE-REGISTRATION (logged 2026-10-11, before execution; results not yet computed)

Three further arms complete the reviewer-requested controls. All inherit the frozen splits, features A, scales, donor temporal constraints, and the crash-safe (task, cutoff) execution pattern; inference as in phase 1.

E1 Leave-one-HUC4-out (and leave-one-HUC2-out supplement): for each held-out HUC4 group g (28 groups; 10 HUC2), targets are the primary reservoirs in g; donors are all primary reservoirs outside g, labels restricted to donors' own frozen training years and predating each target's cutoff. Calibration uses donor calibration-year rows only (LOBO convention); target labels never enter fitting, calibration, or scale estimation (per-reservoir scales are cutoff-truncated training statistics of the target's own history, as in the frozen shared scheme; the target's input features at issue time are its own observed history). Frozen 500-round configuration for comparability with the LOBO analysis. Reference: target persistence. Reporting: equal-reservoir mean skill, bootstrap over held-out group means (4,000 draws), plus the paired contrast (LOH4O - full-pool LOBO-comparable rerun) is NOT computed; instead LOH4O is compared descriptively with the frozen LOBO skills of Section 3.4, since donor pools and calibration rules differ.

E2 Identity encoding (seen-reservoir only): pooled model with one-hot reservoir identity added to feature set A + statics; tuned under the same 16-point capacity grid with pooled calibration selection; identity active at inference only for reservoirs seen in training. Not a zero-sample claim. Primary contrast: identity - tuned-without-identity, paired, HUC4 cluster bootstrap + sign-flip + Holm within the four-target family.

E3 Pretrain-then-finetune: pooled tuned model (re-fitted per task-cutoff as in F1) as init; fine-tuned on each target reservoir's own training rows with learning rate 0.01, max 300 rounds, early stopping (30) on the target's calibration year. Primary contrast: finetune - local500 (frozen), paired, same inference.

Declared interpretation limits: all phase-2 arms are post-batch extensions; the frozen primary endpoint is unchanged; no equivalence claims; E1's HUC2 variant is reported with wide uncertainty given 10 units.
