# 证据登记表（2026-11-09）

| 断言 | 证据类别 | 状态 | 边界 |
|---|---|---|---|
| M5 包含 12 个聚合层、42,840 条序列 | 官方竞赛论文 | SOURCE_REPORTED | 非本仓库独立复算 |
| Accuracy 预测 28 天，并用 WRMSSE | 官方规则与论文 | SOURCE_REPORTED | 非经营收益 |
| Uncertainty 每个预测日提交九个分位数，预测跨度 F1–F28 | Kaggle 官方提交说明 | SOURCE_REPORTED | 每日边际预测，不等于完整联合分布 |
| 冠军方案组合多个 LightGBM 模型 | 冠军方法论文 | SOURCE_REPORTED | 非本仓库复跑 |
| 竞赛方法排名会随测试窗口变化 | 独立稳健性论文 | SOURCE_REPORTED | 适用其研究窗口 |
| 更低预测误差未必带来最低库存成本 | 库存模拟论文 | SOURCE_REPORTED | SIMULATED 而非实际经营实验 |
| 概率预测必须检验校准与锐度 | 概率预测方法学论文 | METHODOLOGY | 必须按分组和时间检验 |
| 跨区域分位数可直接相加 | 数学上一般不成立 | REJECTED | 需联合情景 |
| 本仓库复现了冠军成绩 | 无实验报告 | NOT_VERIFIED | 禁止宣称 |
| 本仓库证明了利润增长 | 无经营干预 | NOT_VERIFIED | 禁止宣称 |

## 证据等级

- **SOURCE_REPORTED**：论文原作者已报告；按原结论引用，尚未独立复算。
- **METHODOLOGY**：已有理论支持的方法学，仍需实际检验。
- **DESIGNED**：已设计但未执行的实验或假设。
- **INDEPENDENTLY_REPRODUCED**：需附程序、数据哈希、运行日志、复算结果后才可使用。
- **SIMULATED**：仅在明确假设条件下计算的库存价值。
- **CAUSAL_FIELD_EVIDENCE**：需有真实可执行策略的可信前瞻性对照。

**当前材料均不达到 INDEPENDENTLY_REPRODUCED 或 CAUSAL_FIELD_EVIDENCE。**
| companion `Estimate WRMSSE.R` 对零尺度是否有显式 fallback | 当前检查的官方 companion R 脚本 | NOT_SUPPORTED_BY_INSPECTED_CODE | 未发现 all-zero/zero-scale 专门分支；不得声称官方指定 epsilon/zero 约定 |
