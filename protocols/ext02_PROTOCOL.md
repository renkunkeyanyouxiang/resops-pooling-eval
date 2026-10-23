# ext02：审计整改计算（DOR-ΔS 回归 + BOR 直接检验 + 辅助增量审计）

日期：2026-10-04。性质：**外部审计后的补充再分析（post-hoc）**，不是预登记实验；只读冻结结果
`hydro_project/completion_20261001/`（final_reservoir_metrics.csv、cohort65.csv、cohort_split65.csv、
basin_huc4_huc2.csv、final_clustered_comparisons.csv），不重训任何模型，随机种子 20261004。

## 背景

审稿发现 v6 稿件 §3.5 声称检验了"共享收益与 DOR 的调整后关联"，但冻结文件
final_dor_adjusted_associations.csv 的响应变量是**各模型的绝对技能**（analyze_completion.py:194
`.skill.reindex(ids)`），并非配对差 ΔS = skill(shared) − skill(local)。本扩展补齐三项：

1. **ΔS-on-DOR 回归**（ext02_dor_delta_regression.csv）：3 个 A 族配对
   （lgb_A_500、capacity_A、ridge_A）× 4 目标 × 2 规格（attributes_agency；
   attributes_agency_cutoff）× 2 聚类层（HUC4/HUC2），设计与冻结 dor_analysis 完全一致
   （标准化 log1p DOR/CAP/CATCH + 机构哑变量 [+ 截止组哑变量]，聚类稳健标准误，Holm 在
   聚类层内跨 3×4×2=24 检验）。
2. **BOR vs 非 BOR 直接检验**（ext02_bor_direct_test.csv）：h1 四个构型 + h3 A500 的
   分层均值差，HUC4/HUC2 聚类 bootstrap 95% CI（4000 次），名义置换 p（10,000 次，库标签
   置换，忽略空间聚簇，保守性不足，仅作描述），Welch t。口径与 §3.6(iv) 一致：
   inflow_source == 'BOR'（n=17）；396 Palisades 编码 BOR_PN 不在层内，
   含入（n=18）的敏感性另报。
3. **辅助增量审计**（ext02_aux_increment_audit.csv）：local/shared 的 B−A 逐库配对差均值、
   中位数、逐目标负值数与四目标全负库数。

## 关键结果（写入稿件 §3.1/§3.4/§3.5/§3.6(iv)）

- ΔS 回归：Spearman(ΔS, DOR) ∈ [−0.11, +0.41]；原始 p<0.05 仅 3 例且全在 HUC2
  （lgb h1 agency β=−0.085 p=0.026；capacity h1 agency β=−0.074 p=0.020；
  capacity h7 agency_cutoff β=+0.015 p=0.037——方向跨时距不一致）；**Holm 后全部不显著
  （最小 Holm p=0.481）**。
- BOR 直接检验 h1：A500 差 −5.98pp [−10.99, −1.43]（HUC4），置换 p=0.001，Welch p=0.008；
  B500 −6.11 [−10.87, −1.78]；A-capacity −6.34 [−10.88, −2.21]；B-capacity −5.99 [−10.67, −1.91]；
  **HUC2 区间均含零**（[−9.56,+1.57] 等，仅 10 簇）；h3 A500 −0.66 [−2.12,+1.00] 不显著；
  含 BOR_PN（n=18）敏感性：−5.57/−5.99pp，区间仍不含零。
- 辅助增量：local B−A 均值 +0.98/+0.92/+1.22/+1.27 pp（中位数 0.43–0.59）；
  四目标全负 3 库；shared500 B−A 均值 +0.15/−0.03/+0.36/+0.05 pp，四目标全负 6 库。
- LOBO h7（冻结文件复核，供 §3.4 措辞）：HUC4 区间 [0.52, 14.95]pp 排零；HUC4 符号翻转
  原始 p=0.063；reservoir 级 p=0.0019；HUC2 区间 [−0.94, 18.29] 含零；Holm p=0.252。

## 结论强度边界

- ΔS 回归的空结果是"在所检验的规格下未获稳健关联"，不是"证明不存在边界"。
- BOR 异质性在库级与 HUC4 级直接检验中成立，HUC2 级不能分辨；置换 p 在空间聚簇下名义化。
  全文标注为探索性异质性。

## 第二轮外审要求的四点核验（2026-10-04，audit_checks.py 独立复算全部 OK）

1. **响应变量确为共享减逐库技能**：`delta(sm, lm, task) = skill(shared cfg) − skill(local cfg)`，
   逐库配对后进入回归；audit_checks.py check1/check4 从原始 CSV 独立重算分层均值与
   Spearman(ΔS, DOR)（h1 A500 ρ=−0.107326），与输出 CSV 逐位一致。
2. **Holm 校正覆盖范围**：在每个聚类层级（HUC4、HUC2）内分别执行，跨
   3 族(lgb_A_500/capacity_A/ridge_A) × 4 目标 × 2 规格 = 24 检验；两层级各 24 行
   （check3 重推 Holm 列与 CSV 一致，n_tests=24）。聚类层级之间不合并。
3. **BOR 区间的重采样方式**：HUC4/HUC2 簇有放回整簇抽样（4000 次，种子 20261004），
   每次把抽中簇的成员库拼接后分别计算两层的均值差；若某次抽样中任一层为空则该次
   记 NaN 并从分位数中剔除。簇组成：HUC4 28 簇中 11 簇含 BOR，NaN 抽样 0/4000；
   HUC2 10 簇中 4 簇含 BOR，NaN 抽样 19–28/4000（0.5–0.7%，见 CSV
   boot_nan_draws_huc4/huc2 列）。
4. **容量扩展比较的参照系**：capacity 比较始终为 capacity_shared_X − capacity_local_X
   （A、B 各自配对），非 500 轮 local 参照（见 run_ext02.py cfgs 与 FAMILIES 定义）。

独立复核（audit_checks.py，2026-10-04）：check1 四构型分层均值、check2 lgb_A_500/h1/
attributes_agency/HUC4 的 beta 与聚类稳健 se/p（lstsq 路径 vs 原 pinv 路径）、
check3 Holm 重推、check4 Spearman 重算——**全部逐位一致（ALL OK）**。
版本溯源：本目录为冻结计算发布 20261001 之外的**新增分析**（post-hoc），只读其输出，
与 ext01（2026-10-03）同属 extensions/ 归档；两者与冻结发布在正文 Data availability 中分别标注。
