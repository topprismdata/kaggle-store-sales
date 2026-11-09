# M5 Hands-on — 第一周执行 Runbook V1.1

> 本 Runbook 可以直接交给研发执行。所有命令默认从仓库根目录运行。  
> 首周只做 M5 Accuracy；Uncertainty 数据和概率模型放到 W3。

## A. Owner 开始前需要完成的三件事

1. **Kaggle 规则接受人**：由 Owner 指定一名已登录 Kaggle 并接受 M5 Accuracy 竞赛规则的团队成员；姓名/账号和接受日期写入 G0 manifest。
2. **数据保管人**：由 Owner 指定一名研发负责人，负责下载、只读保管、SHA-256 清单和访问控制。可以与规则接受人为同一人，但必须分别登记角色。
3. **运行环境**：团队 Linux 仍为规范实验环境；Owner 的 Windows 11 工作站登记为开发/冒烟环境。Windows run 与 Linux canonical run 的硬件字段、配置、结果目录分开，不能合并成绩。

当前规则接受人和数据保管人尚未命名，故 G0 仍为 BLOCKED。不要把 Kaggle token 发到聊天、GitHub、Notebook 或命令行日志中。Kaggle CLI 推荐 OAuth `kaggle auth login`；参见 [Kaggle CLI 官方文档](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)。

## B. 环境安装与认证

### B1. 统一版本

项目环境固定 Python **3.11.15 x64**。Owner 机器上的 Python 3.13.13 保留给其他工作，不作为本项目首轮规范版本。依赖直接版本固定在 `m5/requirements-py311.txt`；这是精确的顶层依赖 pin，尚非包含所有传递依赖哈希的完整 lockfile。Linux 环境登记后需生成其平台专属完整 lock。

### B2. Windows PowerShell（开发/冒烟环境）

项目路径使用 ASCII 目录 `m5/`，原始数据放 E 盘，不放 D 盘。启动 Python 前设置 UTF-8：

```powershell
$env:PYTHONUTF8 = "1"
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r m5\\requirements-py311.txt
python m5\\scripts\\00_preflight.py
```

### B3. Linux（规范实验环境）

```bash
export PYTHONUTF8=1
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r m5/requirements-py311.txt
python m5/scripts/00_preflight.py
```

Windows 和 Linux 必须各自保存环境报告与锁文件。首轮 LightGBM 固定 CPU 后端，配置中固定 `deterministic=true`、`force_row_wise=true`、`num_threads` 和所有 seed；GPU 不作为 W1 的必要条件。

### B4. Kaggle 登录

```powershell
kaggle --version
kaggle auth login
kaggle competitions files m5-forecasting-accuracy
```

无浏览器服务器使用 `kaggle auth login --no-launch-browser`。认证文件仅保存在执行用户目录，权限限制为当前用户可读写。

## C. 下载数据

### C1. Windows（Owner 开发机）

```powershell
New-Item -ItemType Directory -Force E:\\m5-data\\raw\\m5_accuracy
kaggle competitions download m5-forecasting-accuracy --path E:\\m5-data\\raw\\m5_accuracy --unzip
```

### C2. Linux（规范机）

```bash
mkdir -p /srv/m5-data/raw/m5_accuracy
kaggle competitions download m5-forecasting-accuracy --path /srv/m5-data/raw/m5_accuracy --unzip
```

预期核心文件：

```text
calendar.csv
sales_train_validation.csv
sales_train_evaluation.csv
sell_prices.csv
sample_submission.csv
```

Accuracy 原始数据由 Kaggle 按竞赛规则提供。下载后不要把原始文件、压缩包或凭据上传 GitHub。

## D. 下载后第一时间留存数据指纹

**Windows PowerShell：**

```powershell
python m5\\scripts\\fingerprint.py --input E:\\m5-data\\raw\\m5_accuracy --output m5\\runs\\manifests\\accuracy-data.json
```

**Linux：**

```bash
python m5/scripts/fingerprint.py --input /srv/m5-data/raw/m5_accuracy --output m5/runs/manifests/accuracy-data.json
```

脚本使用 Python 标准库，逐文件计算 SHA-256 和字节数，跨 Windows/Linux 可用，不依赖 Bash heredoc。将文件清单、下载时间、Kaggle competition slug、规则接受人和数据保管人写入 G0 运行记录。只提交不含原始数据和凭据的 manifest。

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
3. **Companion Drive 已在本会话连接的 Google Drive 中验证可列目录/下载**：已看到 `M5-Competitors-Guide.pdf`、`Estimate WRMSSE.R`、`Estimate weights.R`、`weights_validation.csv`、`weights_evaluation.csv`、`M5_Evaluation_Accuracy.R`、Accuracy template、24 个 Accuracy benchmark forecasts、50 个 Accuracy submissions，以及 validation 目录中的 point/probabilistic benchmark R 脚本。Day 3 负责人必须先下载并登记文件 SHA-256。
4. 以官方实现为 reference；Python 实现必须逐层、逐序列与 reference 交叉验证。
5. 若 reference 代码/权重暂时无法取得，记录 `SCORER_NOT_VERIFIED`，可以开发模型但不准对外报告“官方 WRMSSE”。

必须记录权重计算时点、RMSSE scaling denominator、首个非零销售前的处理、12 层级权重与归一化方式。不得仅凭公式印象自行重写。\n\n**全零/零尺度门禁**：官方 guide 规定 RMSSE scale 从首个非零销售后计算；但本轮检查的 companion `Estimate WRMSSE.R` 在计算 `mean(diff(insample)^2)` 时没有显式的全零/零尺度 fallback。因此不能说官方 reference 已规定给全零序列赋 0 或 epsilon。Python scorer 必须对 all-zero history、active-history constant/zero-scale 显式抛出 `UndefinedRMSSEScaleError`；若真实数据触发，保持 `SCORER_NOT_VERIFIED`，先与官方评价脚本和官方数据分布核实，禁止静默跳过或加 epsilon。

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
- [ ] WRMSSE 与官方 reference 对照通过；\n- [ ] all-zero history、single-active-observation、constant-active-history 的零尺度单元测试通过；若正式数据触发零尺度，维持 `SCORER_NOT_VERIFIED` 并提交异常清单；
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
