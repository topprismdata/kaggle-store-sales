# M5 文献研究矩阵（公开出处）

> 记录公开论文中可溯源的结论；本仓库尚未复跑官方比赛和论文实验。DOI 用于寻找出版物原文，并不表示所有全文已被逐页精读。

| 主题 | 文献 | 来源链接 | 对本研究的作用 | 重要边界 |
|---|---|---|---|---|
| 赛题与数据 | Makridakis et al., *M5 Competition: Background and Implementation* (2022) | https://doi.org/10.1016/j.ijforecast.2021.07.007 | 数据、12 层、任务时间 | 预测竞赛，不是经营利润评估 |
| 点预测总论 | Makridakis et al., *M5 Accuracy: Results, Findings and Conclusions* (2022) | https://doi.org/10.1016/j.ijforecast.2021.11.013 | 评分、冠军、基线与跨层结果 | WRMSSE 不是经济效用 |
| 概率总论 | Makridakis et al., *M5 Uncertainty: Results, Findings and Conclusions* (2022) | https://doi.org/10.1016/j.ijforecast.2021.10.009 | 九分位点、WSPL、概率竞赛 | 有限分位点不等于完整联合分布 |
| 冠军工程 | In & Jung, *Simple Averaging of Direct and Recursive Forecasts* (2022) | https://doi.org/10.1016/j.ijforecast.2021.11.007 | 不同数据池、直接/递归与融合 | 训练规模和结果不能机械迁移 |
| 概率冠军 | Lainder & Wolfinger (2022), M5 Uncertainty winning solution | https://doi.org/10.1016/j.ijforecast.2021.12.003 | 梯度提升树、验证与数据增强 | 需要重建时间信息边界 |
| 计数/白盒 | de Rezende et al., *A White-boxed ISSM Approach* (2022) | https://doi.org/10.1016/j.ijforecast.2021.11.006 | 状态空间、负二项与 Monte Carlo 轨迹 | 联合依赖与独立假设应检验 |
| 分位评价 | Chen et al., *Evaluating Quantile Forecasts in the M5 Uncertainty Competition* (2022) | https://doi.org/10.1016/j.ijforecast.2022.03.004 | 分位点和层级的评价结构 | 不能盲目聚合分位点 |
| 概率预测库存价值 | Spiliotis et al., *Product sales probabilistic forecasting* (2021) | https://doi.org/10.1016/j.ijpe.2021.108237 | 比较概率方法与库存相关评价 | 库存评价基于特定政策/假设 |
| 库存成本 | Theodorou et al., *Forecast accuracy and inventory performance* (2025) | https://doi.org/10.1016/j.ejor.2024.12.033 | 多预测方法、order-up-to 策略与成本情景 | 模拟结果不是实际利润 |
| 稳健性 | Ma & Fildes, *The performance of the global bottom-up approach…: A robustness check* (2022) | https://doi.org/10.1016/j.ijforecast.2021.09.002 | 多时间窗口及排名稳定性 | 实验窗口和所测方法有限 |
| 指标批判 | Hewamalage et al., *A Look at the Evaluation Setup of the M5 Forecasting Competition* (2021) | https://arxiv.org/abs/2108.03588 | 分层、加权、尺度化与排名 | 工作论文；不是官方结论 |
| 概率诊断 | Gneiting et al., *Probabilistic forecasts, calibration and sharpness* (2007) | https://doi.org/10.1111/j.1467-9868.2007.00587.x | 校准、锐度、适当评分 | 理论本身不保证任何模型达标 |
| 分位数诊断 | Gneiting et al., *Model diagnostics and forecast evaluation for quantiles* (2023) | https://doi.org/10.1146/annurev-statistics-032921-020240 | 条件覆盖、评分与可靠性图 | 分组样本量要足够 |
| 概率协调 | Panagiotelis et al., *Probabilistic Forecast Reconciliation* (2023) | https://doi.org/10.1016/j.ejor.2022.07.040 | 联合/层级一致性框架 | 需定义有效的联合分布 |
| 服饰外推边界 | Sousa et al., *Predicting demand for new products in fashion retailing using censored data* (2025) | https://doi.org/10.1016/j.eswa.2024.125313 | 缺货截断、冷启动需求恢复 | 非 M5；仅用于说明外推难题 |

## 官方资源

- M5 Accuracy：https://www.kaggle.com/competitions/m5-forecasting-accuracy
- M5 Uncertainty：https://www.kaggle.com/competitions/m5-forecasting-uncertainty
- 官方方法与代码：https://github.com/Mcompetitions/M5-methods
- M5 特刊综述：https://pmc.ncbi.nlm.nih.gov/articles/PMC9232271/

## 下一步精读记录的统一模板

逐篇记录：研究问题；数据/时间切分；统计目标；基线；核心方法；真实复算结果；局限；可迁移假设；证据等级；复现资源；未解决问题。

**当前状态**：文献索引与研究归纳；并非逐篇全文复核完成清单。
