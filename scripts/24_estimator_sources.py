# -*- coding: utf-8 -*-
"""Join the published HUC4 intervals with the later reservoir-equal rerun.

Published Table S9 HUC4-equal intervals come from ext03_contrasts.csv, produced by
05_stats.py: point = unweighted mean of HUC4 means; interval = multinomial cluster
resample, then (sum of reservoir deltas) / (count of reservoirs) inside the resample.
Seed 20261011, 4000 draws, 20000 cluster sign-flips.

ext03_tuned_vs_local_two_estimators.csv later recomputed the HUC4 interval as the
unweighted mean of resampled cluster means. Same point estimates, different interval.
Those rerun intervals are not the numbers in Table S9.
"""
from pathlib import Path
import pandas as pd

EX = Path(r"I:\stremflow\extensions\ext03_fair_pooling")
contr = pd.read_csv(EX / "ext03_contrasts.csv")
contr = contr[contr.contrast.eq("shared_tuned-local")][
    ["task", "delta_skill_pp", "ci95", "signflip_p", "holm_p"]
].rename(columns={
    "delta_skill_pp": "huc4_point_published",
    "ci95": "huc4_ci_published_05_stats",
    "signflip_p": "huc4_signflip_published",
    "holm_p": "huc4_holm_published",
})
two = pd.read_csv(EX / "ext03_tuned_vs_local_two_estimators.csv")
m = two.merge(contr, on="task", how="left")
m["huc4_point_matches"] = (m["huc_eq_pt"] - m["huc4_point_published"]).abs() < 0.02
m["published_source"] = "ext03_contrasts.csv via 05_stats.py"
m["rerun_ci_source"] = "ext03_tuned_vs_local_two_estimators.csv; unweighted cluster-mean bootstrap; not used in Table S9"
m.to_csv(EX / "ext03_estimator_sources.csv", index=False)
print(m[["task", "res_eq_pt", "res_eq_ci", "huc4_point_published", "huc4_ci_published_05_stats", "huc_eq_ci", "huc4_point_matches"]].to_string(index=False))
