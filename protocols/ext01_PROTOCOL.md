# 扩展分析协议 ext01：共同测试时段（WY2018）共享-逐库配对差再分析

登记日期：2026-10-03（执行前登记）
性质：事后补充的稳健性分析（exploratory robustness），不改变主终点、不新增模型拟合、不重选基线。
动机：冻结设计中 50 座主样本测试时段不完全相同（截止 2011/2014/2015/2016 四组）。审稿可能质疑"短提前期共享劣势由测试时段差异驱动"。本分析把所有库限制到同一目标水文年，检验配对差方向与区间是否稳定。

## 预登记规则

1. **数据**：仅使用已保存的 `completion_20261001/all_completion_forecasts_wide.parquet`（逐库逐起报日全部模型预测、真值、训练优选基线、持续性）。不读取原始数据，不训练模型。
2. **子样本**：主队列（cohort=primary_dual_catalog）中训练截止晚于 2011-09-30 的库（即 2014/2015/2016 三组，测试期均覆盖 WY2018），预期 48 座；以 `cohort_split65.csv` 的 cutoff 字段判定。
3. **行过滤**：water_year == 2018（目标日落在 2018 年 10 月 1 日–2019 年 9 月 30 日的水文年）。四任务 h1/h3/h7/v7 全做。宽表中每行天然对齐全部模型，无需重对齐。
4. **统计量**：逐库计算
   delta_mae = mean( |y−pred_model| − |y−pred_ref| )，
   delta_skill = −delta_mae / mean|y−baseline|（baseline=训练优选基线列；零目标族用 persistence）。
   比较对（与主分析同名）：
   - lgb_shared_A vs lgb_local_A
   - capacity_shared_A vs capacity_local_A
   - lgb_shared_B vs lgb_local_B
   - capacity_shared_B vs capacity_local_B
   - （次要）lobo_lgb_A vs persistence、lobo_lgb_B vs persistence
5. **推断**：库级等权均值；HUC4 聚簇自助 4000 次（多项式簇权重复现主分析 cluster_ci 实现）；簇级符号翻转（≤16 簇精确枚举，否则 20000 次蒙特卡洛）；同一比较对内四目标 Holm 校正。随机种子 20261001。
6. **报告**：无论方向如何完整报告 mean、95%CI、signflip p、Holm p、库数、行数；同时报告 WY2018 上 lgb_local_A、capacity_shared_A、lgb_shared_A 的平均技能供背景。
7. **有效性自检（先于正式结果）**：去掉 WY2018 过滤、用全部测试行重算，应复现 `final_clustered_comparisons.csv` 的 headline（h1 lgb_shared_A−lgb_local_A mean ≈ −0.02529）；聚簇自助含随机性，允许 CI/p 与冻结值有小差异，均值必须一致到 1e-6。若自检失败，停止并排查，不发布结果。
8. **解释边界**：WY2018 单一年份；结果外推到其他年份受该年水文特性限制；48 库子样本不等于 50 库全样本。本分析结论只用于回应"测试时段不同"的替代解释，不产生新的主终点声明。

## 输出

- `ext01_common_wy2018_results.csv`：逐比较对×任务的统计
- `ext01_reproduction_check.txt`：自检输出
- `ext01_run.log`：运行日志
