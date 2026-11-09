# M5 Hands-on Reproduction — 执行计划 V1.0

**状态：READY FOR TEAM EXECUTION（待 Owner 确认执行环境与 Kaggle 数据权限）**  
**计划周期：4 周；第一周为强制启动门**  
**目标：亲自复现 M5 关键预测任务，建立从数据、评分、概率分布到库存模拟的可重复实验链。**

---

## 0. 执行决策

同意开展 hands-on reproduction，但要明确它不是重新参加已结束的排行榜竞赛，也不是一开始就复刻冠军的 220 个模型。

第一阶段的目标是让研究团队真正理解并跑通：

1. M5 的原始数据和时间边界；
2. 28 天点预测；
3. 12 层级 WRMSSE 的正确计算；
4. 强基线与全局机器学习模型；
5. 按层级、商品和时间窗口解释结果。

之后再进入冠军方法、九分位概率预测、库存决策模拟。

**第一周只做 Accuracy 赛道。** 不同时下载和开发 Uncertainty，不在评分器尚未核验时追求模型分数，也不做库存收益结论。

## 1. 官方数据与任务契约

### 1.1 竞赛入口

- M5 Accuracy：<https://www.kaggle.com/competitions/m5-forecasting-accuracy>
- M5 Uncertainty：<https://www.kaggle.com/competitions/m5-forecasting-uncertainty>
- 官方方法、指南、获奖代码与成绩：<https://github.com/Mcompetitions/M5-methods>
- 官方 M5 方法仓库补充资料目录：<https://drive.google.com/drive/folders/1D6EWdVSaOtrP1LEFh1REjI3vej6iUS_4?usp=sharing>

Kaggle 页面显示 Accuracy 数据包约 450 MB，Uncertainty 数据包约 515 MB；Accuracy 首周只需下载其官方五个文件。下载前需要在 Kaggle 页面登录并接受竞赛规则。竞赛已经结束，本项目进行离线研究，不需要也不应把“提交排行榜”作为第一周目标。

### 1.2 Accuracy 数据文件

| 文件 | 用途 | 首周处理 |
|---|---|---|
| `sales_train_validation.csv` | 30,490 条 item-store 序列，历史至 d_1913 | 训练输入 |
| `sales_train_evaluation.csv` | 同一底层序列，历史至 d_1941，含 d_1914–1941 验证实际值 | 仅用于验证标签及历史回测 |
| `calendar.csv` | 日期、周、星期、事件、SNAP | 特征与未来已知日历 |
| `sell_prices.csv` | 商品×门店×周售价 | 价格特征；核验未来价格可见性 |
| `sample_submission.csv` | 官方预测输出格式 | 格式校验，不作为标签 |

数据基础：3,049 个商品、10 个门店、3 个州、30,490 条底层 item-store 序列；官方层级评分覆盖 12 个层级、42,840 条序列。原始销售列是**已实现销售单位**，不是库存，也不等同于未受缺货截断的潜在需求。

### 1.3 时间边界（不可修改）

| 区间 | 日序号 | 作用 |
|---|---|---|
| 首轮训练 | d_1–d_1913 | 官方 Accuracy 训练可见历史 |
| 首轮验证 | d_1914–d_1941 | 28 天公开验证窗口 |
| 官方私有评价 | d_1942–d_1969 | 竞赛最终测试区间；实际标签不在公开训练文件中 |
| 日历 | 覆盖预测日所需日期 | 未来日历特征 |
| 售价 | 按周记录 | 使用前核对相应预测起点的可见性契约 |

首轮官方验证严格模拟：只使用 d_1–d_1913 的销量训练，预测 d_1914–d_1941。不得把 `sales_train_evaluation.csv` 中的验证销量混进训练特征。

补充稳健性回测建议先冻结六个连续 28 天窗口：预测起点 d_1773、d_1801、d_1829、d_1857、d_1885、d_1913。它们是本项目的历史滚动回测设计，并非官方 M5 排行榜切分。每个起点只读取起点之前的销量；未来日历/售价仅在该信息在预测时可见且数据确实提供时使用。若售价的发布时间/可见性无法证明，另做不含未来售价的敏感性版本。

## 2. 机器与软件策略

### 2.1 计算分工

- **规范实验环境**：团队 Linux 计算机；记录 CPU、RAM、GPU、驱动、CUDA、OS。
- **MacBook Pro M1 Max 64 GB**：数据探索、报告审阅、轻量 CPU 基线和结果可视化；不把它伪装成 CUDA 环境。
- **首轮模型**：LightGBM CPU 优先。M5 数据规模不构成强制 GPU 的理由；LightGBM 的 GPU 后端与 CUDA 不是可直接互换的概念。只有当某个候选模型明确支持并验证 GPU 后端时，才单独登记 GPU 实验。
- 所有模型必须使用同一数据切分、信息集、指标实现和预测粒度；硬件不同需记录环境，不能将不同预算结果直接比较。

### 2.2 版本基线

- Python 3.11（当前 Kaggle CLI 文档要求 Python 3.11+）。
- 使用独立虚拟环境；推荐 `uv` 或 `venv`。
- 首轮依赖：`kaggle`、`numpy`、`pandas`、`polars`、`pyarrow`、`scipy`、`scikit-learn`、`lightgbm`、`statsforecast`、`matplotlib`、`pyyaml`、`pytest`、`ruff`。
- 首次安装成功后立即锁定精确版本并提交 lockfile；不要长期使用无版本边界的 requirements。
- 数据审计可用 Polars/Arrow；模型输入输出优先 Parquet，原始 CSV 保持只读。

## 3. Kaggle 数据获取

先由数据管理员在浏览器中接受 Accuracy 竞赛规则，再在有权限的执行机上认证。

当前 Kaggle CLI 文档支持 OAuth `kaggle auth login`、`KAGGLE_API_TOKEN`、`~/.kaggle/access_token`，并保留 legacy 凭据方式。优先 OAuth；不要把 token、`kaggle.json` 或任何认证文件放进仓库、Notebook 输出或日志。

首周下载命令见 [第一周 Runbook](week1-runbook.md)。下载后记录文件大小与 SHA-256。**原始 CSV、解压包、token、派生大文件和模型权重不得提交 GitHub。**

## 4. 代码目录与职责

建议在 `M5研究/` 下建立：

```text
M5研究/
├── README.md
├── deck/
├── docs/
├── configs/
│   └── accuracy_v1.yaml
├── src/m5research/
│   ├── io.py                 # 原始文件读取与数据契约
│   ├── hierarchy.py          # 12 层级映射、聚合矩阵
│   ├── features.py           # 仅基于预测起点前数据的特征
│   ├── metrics/
│   │   └── wrmsse.py         # 官方协议实现/适配器
│   ├── models/
│   │   ├── seasonal_naive.py
│   │   └── global_lgbm.py
│   ├── forecast.py           # 28 天滚动预测
│   └── reporting.py          # 分层诊断和图表
├── scripts/
│   ├── 00_preflight.py
│   ├── 01_audit_data.py
│   ├── 02_verify_hierarchy.py
│   ├── 03_run_baselines.py
│   └── 04_score_wrmsse.py
├── tests/
│   ├── test_data_contract.py
│   ├── test_hierarchy.py
│   ├── test_wrmsse.py
│   └── test_no_future_leakage.py
├── data/                     # gitignored
├── artifacts/                # gitignored
└── runs/                     # gitignored
```

文件职责不可混用：`io.py` 不做模型特征；`hierarchy.py` 不负责评分；`wrmsse.py` 不自行猜测权重；`features.py` 必须显式接收 forecast origin；`forecast.py` 保存每个预测起点的输入时间戳。

## 5. 四周实施路线

| 周次 | 工作包 | 主要交付 | 退出条件 |
|---|---|---|---|
| W1 | 数据、层级、官方评分、基线 | 审计报告、评分器核验、Seasonal Naive、Global LightGBM 首跑 | G0–G4 通过 |
| W2 | Accuracy 深入与冠军方法 | 直接/递归、分组 partial pooling、组合方法、六窗口稳健性 | 有按层级和窗口的可解释比较 |
| W3 | Uncertainty 概率预测 | 九分位点、WSPL、校准/锐度/尾部、分位交叉与概率路径 | 概率输出通过格式和可靠性检查 |
| W4 | 库存决策模拟 | 固定政策、成本敏感性、预测与策略增益拆解 | 输出 SIMULATED 决策结果和边界 |

W2 不承诺第一时间复刻冠军完整 220 模型；先复刻其方法结构与关键对照，再决定是否投入完整规模。W3 必须先读取 Uncertainty 官方 sample submission 和 guide，不能照搬 Accuracy 提交格式。

## 6. W1 实验定义

### W1-A：数据审计

- 确认 30,490 个唯一 item-store ID，无重复；
- 核验 3,049 item、10 store、3 state；
- 核验 `sales_train_validation` 与 `sales_train_evaluation` 的 d_1–d_1913 完全一致；
- 核验销售值为非负整数，缺失值与零销量分别统计；
- 核验 calendar 日期映射、事件字段、SNAP、未来日期覆盖；
- 核验 sell_prices 主键唯一性、缺失和预测窗口价格覆盖；
- 输出各店/州/品类/部门的销量稀疏度、零比例、销售活跃度、价格变化与分布。

### W1-B：层级与评分器

- 从官方 M5 guide、官方方法仓库及官方 R 评价代码建立参考实现。
- 底层预测必须为 30,490 × 28；再按官方层级映射聚合为 42,840 条序列。
- 逐层核验 12 个序列数、聚合和、尺度项、权重及权重归一化。
- WRMSSE 以官方口径为准；先用官方参考预测/已知评分交叉核验，再对本地预测打分。
- 如果无法获得官方参考代码/文件，状态记为 `SCORER_NOT_VERIFIED`，不得宣布评分器通过或发布模型排名。

### W1-C：基线

1. `SeasonalNaive7`：用最近可见的同星期销量重复预测 28 天。
2. `SeasonalMean28`：最近 28 天对应星期均值的简单基线。
3. `GlobalLightGBMRecursive`：一个跨底层序列共享的单步模型，递归滚动 28 天。

Global LightGBM 首轮目标训练样本使用每条序列最近 365 个可训练目标日；特征包括滞后销量（7、14、28、56、364 天，实际可用时才生成）、滞后滚动统计、星期/月份/事件/SNAP、静态商品和门店类别、预测起点可用的价格信息。所有 rolling 统计先 shift，防止当前目标泄漏。此模型是本研究强基线，不冒称 M5 冠军 DRFAM。

先做 3,000 条分层抽样序列 smoke test，再跑全量 30,490 序列。抽样按州、门店、品类、销量活跃度分层，固定抽样种子；抽样成绩不能与全量正式成绩混报。

### W1-D：诊断报告

必须报告：
- overall WRMSSE 与 12 个层级 WRMSSE；
- 各州、门店、品类、部门、需求活跃度分组结果；
- 预测与实际时间曲线（高销量、间歇、零销量、新上市/长尾样例）；
- 训练耗时、预测耗时、内存峰值；
- 模型间排名及相对基线增益；
- 错误样例和下一轮问题清单。

## 7. 统一运行记录

每个 run 至少记录：

```json
{
  "run_id": "M5A_W1_...",
  "git_commit": "...",
  "data_manifest_sha256": "...",
  "config_sha256": "...",
  "environment_lock_sha256": "...",
  "forecast_origin": 1913,
  "horizon": 28,
  "seed": 20261109,
  "model": "...",
  "metric_version": "...",
  "hardware": "...",
  "started_at": "...",
  "finished_at": "...",
  "status": "..."
}
```

每个预测文件必须关联唯一 run_id。不得覆盖历史 run；重跑创建新 run_id。

## 8. W1 验收门禁

| Gate | 通过条件 | 不通过时 |
|---|---|---|
| G0 Data Provenance | 来源、规则接受人、文件名/大小/hash、下载日期登记 | 停止训练 |
| G1 Data Contract | 关键行列、主键、日期、重复、缺失检查通过 | 停止评分 |
| G2 Hierarchy | 12 层映射数、底层加总与官方结构一致 | 停止 WRMSSE |
| G3 Scorer | 官方 reference 对照通过，权重/尺度项可追溯 | 仅可做内部临时指标，不得叫官方分数 |
| G4 Baseline | 28 天全量预测、无负值/NaN/Inf、无未来泄漏 | 修复后重跑 |
| G5 Report | 全层级和分组结果、运行成本、失败样例完整 | 不进入 W2 |

## 9. Owner 与团队职责

- **Owner（用户）**：确认目标与资源、亲自查看数据分布和预测图、参与结果解释与 Go/No-Go。
- **实施团队**：下载、环境、代码、训练、测试、日志与复算。
- **研究设计/评审**：任务契约、实验协议、强基线选择、指标解释、结果审查和证据边界。
- **独立复核人**：对评分器、时间泄漏与关键结论进行复算/检查。

本计划不假设助手能访问团队 CUDA 主机，也不声称已运行任何 M5 模型。

## 10. W1 后的决策

- **GO**：G0–G5 全通过，进入冠军结构复现。
- **RESEARCH_ONLY**：数据/指标可运行，但官方评分复现或数据边界尚未解决；先修复。
- **NO_GO**：数据权限、标签时间契约或评分实现无法证明正确；暂停模型比较。

不得因四周排期而降低验收标准。
