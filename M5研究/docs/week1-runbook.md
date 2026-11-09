# M5 Hands-on — 第一周执行 Runbook

> 本 Runbook 可以直接交给研发执行。所有命令默认从仓库根目录运行。  
> 首周只做 M5 Accuracy；Uncertainty 数据和概率模型放到 W3。

## A. Owner 开始前需要完成的两件事

1. 登录 Kaggle，打开 [M5 Accuracy 数据页](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)，点击接受竞赛规则。
2. 指定一个数据保管人和一台规范运行机。建议团队 Linux 主机负责规范运行；Mac 可用于审阅与轻量探索。

不要把 Kaggle token 发到聊天、GitHub、Notebook 或命令行日志中。当前 Kaggle CLI 推荐 OAuth `kaggle auth login`；也支持 API token 环境变量和 `~/.kaggle/access_token`。具体参照 [Kaggle CLI 官方文档](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)。

## B. 环境安装与认证

### B1. Python 环境

要求 Python 3.11+（当前官方 Kaggle CLI 文档要求）。

使用 venv 的示例：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install kaggle numpy pandas polars pyarrow scipy scikit-learn lightgbm statsforecast matplotlib pyyaml pytest ruff
```

如果使用 uv，可用 `uv venv --python 3.11` 建立隔离环境，依赖经首次验证后写入 lockfile。

### B2. Kaggle 登录

```bash
kaggle --version
kaggle auth login
kaggle competitions files m5-forecasting-accuracy
```

若在无浏览器的服务器登录，使用 `kaggle auth login --no-launch-browser`，按终端给出的授权链接完成认证。也可以按官方文档设置 `KAGGLE_API_TOKEN` 或 token 文件。认证文件权限应限制为当前用户可读写。

## C. 下载数据

```bash
mkdir -p M5研究/data/raw/m5_accuracy
kaggle competitions download m5-forecasting-accuracy \
  --path M5研究/data/raw/m5_accuracy \
  --unzip
```

如果项目工作目录本身就是 `M5研究/`，则改用：

```bash
mkdir -p data/raw/m5_accuracy
kaggle competitions download m5-forecasting-accuracy \
  --path data/raw/m5_accuracy \
  --unzip
```

预期核心文件：

```text
calendar.csv
sales_train_validation.csv
sales_train_evaluation.csv
sell_prices.csv
sample_submission.csv
```

官方页面列出以上五个文件，Accuracy 数据包约 450.47 MB。若下载包中多出 README 或系统文件，先登记，不要静默删除。

## D. 下载后第一时间留存数据指纹

在项目根目录执行：

```bash
find "M5研究/data/raw/m5_accuracy" -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256
```

Linux 也可用 `sha256sum`。将结果、文件字节数、下载时间、Kaggle competition slug、数据授权人写入内部运行记录；**只提交不含凭据和原始数据的 manifest**。

原始 CSV、压缩包、Kaggle token、模型权重和大型派生文件不得提交到 GitHub。仓库的 `.gitignore` 已忽略数据和运行目录，但执行者仍须检查 `git status --ignored`。

## E. 数据审计任务（Day 2）

### E1. 结构检查

必须核验：

| 对象 | 预期检查 |
|---|---|
| `sales_train_validation.csv` | 30,490 行；6 个 ID/层级字段；d_1–d_1913 |
| `sales_train_evaluation.csv` | 30,490 行；相同底层 ID；d_1–d_1941 |
| `calendar.csv` | 日期 ID 覆盖 d_1–d_1969；日期唯一 |
| `sell_prices.csv` | store/item/week 主键；价格非负；重复主键统计 |
| `sample_submission.csv` | 列名和行 ID 严格匹配官方模板 |

如官方文件的列顺序或版本与预期不同，以下载文件本身和官方 guide 为准，记录差异后再处理；不得通过删列“修复”。

### E2. 数据一致性

- item-store ID 无重复，30,490 条；
- item 3,049、store 10、state 3；
- validation 文件和 evaluation 文件在 d_1–d_1913 的销量完全一致；
- 销量缺失、负值、非整数分别统计；
- 零销量比例、首次非零日期、连续零长度分布；
- 价格表缺失与“该周无该 item-store 价格记录”分别统计；
- 日历日期、事件、SNAP 与销量日期正确连接。

### E3. 必须产出

- `data_audit.md`；
- `data_manifest.json`（文件名、大小、SHA-256、获取日期、来源）；
- `sales_profile.parquet` 或等价摘要（不可提交原始数据）；
- 分层稀疏度与价格覆盖图；
- 任何异常的 issue 清单。

## F. 层级和官方评分器（Day 3）

1. 按 M5 官方 guide 构建 12 层级映射；底层 30,490 条序列，聚合后 42,840 条。
2. 检查每条底层序列在每个层级中恰好归属正确组，聚合后总销量守恒。
3. 先取官方 M5-methods 仓库的评分参考资料与 R 代码（仓库 README 链接的 companion Drive 包含评价代码、benchmark forecasts 和权重文件）。
4. 以官方实现为 reference；Python 实现必须逐层、逐序列与 reference 交叉验证。
5. 若 reference 代码/权重暂时无法取得，记录 `SCORER_NOT_VERIFIED`，可以开发模型但不准对外报告“官方 WRMSSE”。

必须记录权重计算时点、RMSSE scaling denominator、首个非零销售前的处理、12 层级权重与归一化方式。不得仅凭公式印象自行重写。

## G. 基线模型（Day 4）

### G1. SeasonalNaive7

对每个底层 item-store 序列，使用预测起点前最近的同星期销量循环预测未来 28 天。不得读取目标窗口真实销量。

### G2. SeasonalMean28

以预测起点前最近 28 天的对应星期均值形成简单基线。若序列过短或历史全零，按预注册回退规则处理并单独标记。

### G3. GlobalLightGBMRecursive

先对分层抽样的 3,000 条序列做 smoke test，再扩至 30,490 条全量。建议首轮仅用最近 365 个目标日训练全局一步模型，保留足够历史生成 lag 特征。

候选特征：lag 7/14/28/56/364、shift 后 rolling mean/std、星期/月/事件/SNAP、item/store/category/dept、预测起点可见的价格及价格变化。

约束：
- rolling 特征必须先 shift；
- 每个 run 显式传入 forecast origin；
- 递归预测时只将模型先前预测回填，不得读取未来真实销量；
- future price 必须遵守官方信息可见性；无法确认则做不含 future price 的对照；
- 预测值 finite、non-negative；
- LightGBM 首轮固定 CPU 后端，GPU 后端另做实验，不与 CPU run 混合调参。

## H. 评分与结果审阅（Day 5）

输出：
- Overall WRMSSE；
- Level 1–12 的 WRMSSE；
- 按州、门店、品类、部门、销量活跃度分组的误差；
- 高销量、间歇销量、零销量和价格变化样例图；
- 训练耗时、预测耗时、内存峰值；
- 配置、随机种子、Git SHA、数据 hash、预测文件 hash；
- 对每个失败案例写明原因假设和待验证问题。

不得只交一个 overall 分数；不得将验证窗口结果描述为官方私有测试成绩。

## I. 第一周验收清单

- [ ] Kaggle 规则已接受，数据下载权限正常；
- [ ] 五个数据文件完整，SHA-256 已登记；
- [ ] 原始数据未提交 GitHub；
- [ ] 数据结构与 ID 唯一性通过；
- [ ] 两个 sales 文件共有 d_1–d_1913 完全一致；
- [ ] 日历/售价连接与未来可见性已审计；
- [ ] 12 层级和 42,840 条聚合序列通过核验；
- [ ] WRMSSE 与官方 reference 对照通过；
- [ ] SeasonalNaive7 全量预测成功；
- [ ] GlobalLightGBM smoke test 和全量 run 成功（若资源不允许，明确记录阻塞，不能伪报通过）；
- [ ] 结果含 12 层级分解、分组诊断和 run manifest；
- [ ] 独立复核人能从 manifest 重跑一个 run。

## J. 第一周结束时必须回答的五个问题

1. M5 的总体 WRMSSE 是由哪些层级贡献的？
2. 最细 item-store 层级和高聚合层级的误差差距有多大？
3. 哪些序列的误差主要来自零销量、间歇需求、促销/事件或价格变化？
4. Global LightGBM 相对简单季节基线的收益是否跨店、跨品类稳定？
5. 我们当前的预测目标究竟是销量，还是潜在需求？数据能够支持哪一种解释？

答不出这五题，不进入冠军复现。
