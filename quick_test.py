# -*- coding: utf-8 -*-
"""Quick test: verify headline paired statistics from the bundled outputs.

Run:  python quick_test.py
Expected output: PASS (runs in seconds, no training, pandas only).
"""
import sys
from pathlib import Path

import pandas as pd


def main():
    base = Path(__file__).parent / 'outputs'

    df = pd.read_csv(base / 'ext03_tuned_vs_local_two_estimators.csv')
    row = df.loc[df['task'] == 'h1'].iloc[0]
    assert abs(row['res_eq_pt'] - (-0.39)) < 0.005, f"point {row['res_eq_pt']}"
    assert row['res_eq_ci'] == '[-2.76, 1.36]', f"CI {row['res_eq_ci']}"
    assert abs(row['res_eq_holm_cluster'] - 0.861) < 0.0005, f"p {row['res_eq_holm_cluster']}"
    print('PASS: 1-day tuned-pooled vs frozen-local paired difference = -0.39 pp, '
          '95% CI [-2.76, 1.36], Holm p = 0.861 '
          '(manuscript Section 3.6 and Appendix C)')

    pt = pd.read_csv(base / 'ext03_pretrain_vs_local_ft.csv')
    print('Pretrain-fine-tune contrast file loaded:', len(pt), 'rows '
          '(per-target statistics reported in Appendix C / Table S8)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
