# M5研究

> **M5 Research — Hierarchical Forecasting, Probabilistic Prediction & Inventory Decision Value**  
> 公开研究资料｜中文｜版本 v1.0｜2026-11-09

本项目独立研究 [M5 Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) 及其后续学术工作，探索 **零售预测 → 概率预测 → 库存决策价值** 的完整证据链。

**研究命题**：预测排行榜上的优势是否可重复、能否跨聚合层级成立、是否能够在明确的库存政策与经济成本假设下转化为更好的决策？

## 研究内容

| 板块 | 核心问题 | 材料 |
|---|---|---|
| 竞赛设计 | 预测目标、时间协议、数据、12 层级 | [研究综述](docs/research-overview.md) |
| 领先方法 | 模型组合、全局学习、层级建模 | [论文矩阵](docs/literature-matrix.md) |
| 概率预测 | 九分位点、WSPL、校准、锐度、联合需求 | [概率预测专题](docs/probabilistic-forecasting.md) |
| 评价与稳健性 | WRMSSE、时间外检验、排名稳定性 | [评价协议](docs/evaluation-protocol.md) |
| 库存与价值 | 成本非对称、order-up-to、策略后悔值 | [决策价值专题](docs/inventory-decision-value.md) |
| 实验复现 | 可证伪假设、E1–E4、验收闸门 | [实验计划](docs/experiment-plan.md) |
| 证据边界 | 已有论文结果 ≠ 自行复现 ≠ 实际运营收益 | [证据清单](docs/evidence-register.md) |

## 演示文稿

**[M5研究 — HTML Deck（44 页）](deck/index.html)**

下载 `deck/index.html` 用浏览器打开，支持 ← → / 空格翻页、讲稿切换、打印。文稿完全本地运行；论文链接需要联网打开。

## 最重要的研究发现

1. M5 不只有 Accuracy 赛道，还设置了 Uncertainty 赛道；后者预测九个分位数，按 WSPL 评分。
2. 竞赛冠军使用的是有组织的全局训练、直接/递归预测与模型组合，不应简化为“单个 LightGBM 获胜”。
3. **总排名不等于底层决策粒度排名**；单次排行榜也不保证多时间窗口稳定。
4. 概率预测并非简单输出几个不同数字：必须检查条件覆盖、Pinball/WSPL、尾部可靠性、时间/层级依赖。
5. 分开预测多个门店的 P90，不代表这些 P90 可以相加为总体 P90。
6. 预测误差较低 **不必然** 带来较低库存成本；结论依赖库存政策、缺货与持有成本、补货周期及提前期。
7. M5 公共数据不能直接恢复全部缺货期间的潜在需求或真实库存决策反事实；库存收益实验属于**假设下模拟**。

## 数据与代码状态

- **公开资料梳理：** 已形成文献综述、演示文稿、实验协议。
- **M5 原始数据：** 未在仓库再分发。须通过官方 Kaggle 条款自行获取。
- **竞赛冠军代码：** 未搬运。使用 [官方方法仓库](https://github.com/Mcompetitions/M5-methods) 原始链接。
- **独立复现实验：** *尚未运行或通过评分一致性验收*。本仓库目前是研究资料与预注册实验方案，而非成绩复现仓库。
- **经营收益：** 无真实业务收益实验记录，也无可宣称的利润提升。

## 使用方式

1. 阅读 [研究综述](docs/research-overview.md)，了解 M5 的任务契约与证据边界。
2. 浏览 [HTML Deck](deck/index.html)，快速理解竞赛与概率预测。
3. 使用 [论文矩阵](docs/literature-matrix.md) 逐篇精读一手论文。
4. 依据 [评价协议](docs/evaluation-protocol.md) 和 [实验计划](docs/experiment-plan.md) 开展可复现实验。
5. 在产生独立运行结果后，才更新 [证据清单](docs/evidence-register.md) 中的状态。

## 发布与许可说明

本仓库仅整理研究者原创的分析文字、图示与实验设计，并链接论文和公开资源；**不再分发竞赛数据、论文 PDF、第三方代码或任何非公开材料**。引用论文及官方项目时请遵循原作者权利、竞赛条款及相应许可。HTML 的交互示例数据为教学假设数据。

> **研究纪律：** 文章作者报告的结果 / 独立复现结果 / 仿真经营结果 / 真实因果收益，必须分开标记、分开验收。
