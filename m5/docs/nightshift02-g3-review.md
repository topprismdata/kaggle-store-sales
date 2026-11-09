# M5 Hands-on / 夜班报告②：评审裁决与证据登记

**状态**：LOCAL_G3_PARITY_REPORTED / CODE_AND_LOG_REVIEW_PENDING  
**日期**：2026-11-09；**责任签署**：PENDING_OWNER  
**此文档不宣称助手已运行本机代码，也不将报告的结果冒充独立复现结果。**

## 1. 两类证据严格区分

### 来自本机工程师的执行报告（ENGINEER_REPORTED）

- Windows preflight 6/6、pytest 12/12（上一轮）。
- 数据文件五件齐全；新增 `01b_content_audit.py` 报告跨文件历史销量逐格一致、无负值/非整数/NaN、零销量约 68.2%；12 层聚合划分、价格连接和日历覆盖检查通过。
- 自算验证轮 `Dollar_Sales` 的最大相对差 4.4e-16；依据官方 **12 位舍入的 weight 值**，42,840 个聚合键的最大绝对差为 0。这里不称未经舍入的运算“bit-exact”。
- 评价轮 sNaive，逐层与官方成绩表匹配，报告最大差 1.5e-10；总体约 0.847017。
- 42,840 条训练序列均有有效 RMSSE scale，未触发零尺度分支。
- 验证轮基线：SeasonalNaive7 overall WRMSSE 0.8697；SeasonalMean28 0.7607；原始预测及运行元数据保存在本机 `m5/runs/`。

### 已独立核对的官方文献/文件事实（SOURCE_VERIFIED）

- 官方 Companion `Scores and Ranks.xlsx` 中 **sNaive 评价轮**：L1=0.560047211808532、L12=1.1764190820399、12层 Avg=**0.8470173528438624**；不能与验证轮数值混为同一窗口。
- Dataset 的 `weights_validation.csv` 为 42,840 条五列文件；另有 validation 子目录下同名的四列较低精度版本。两者 42,840 键相同，最大权重绝对差约 5e-8。
- 官方权重和 R 评分脚本来源目录已核对并登记于 [reference/README.md](../reference/README.md)。

## 2. 正式 G0–G5 裁决

| Gate | 当前证据级别 | 裁决 |
|---|---|---|
| G0 | 已报告下载+ SHA-256 manifest；规则接受人、保管人待 Owner 登记 | PENDING_OWNER，禁止宣称已签署 |
| G1 | 工程师报告内容审计通过，但新脚本及日志尚未上传 | ENGINEER_REPORTED_PASS |
| G2 | 工程师报告 12 层划分与守恒通过；源代码仍不可见 | ENGINEER_REPORTED_PASS |
| G3 | 工程师报告逐键 weight 一致及官方 sNaive 12/12 层评价轮复现 | LOCAL_PARITY_ACCEPTED_FOR_CONTINUATION；独立代码审查未完 |
| G4 | 验证轮两个简单基线已报告；预测文件/manifest 尚不可审 | BASELINES_REPORTED |
| G5 | 完整实验报告及独立复核尚未提交 | OPEN |

**研究实施授权**：允许继续开发 GlobalLightGBM、补齐 scorer 测试，并启动低成本六窗口的 Naive 基线回放，均标 `PROVISIONAL_LOCAL`。未审计源码与证据前，不将 G3 设置为 `INDEPENDENTLY_VERIFIED_PASS`。

## 3. 必须收到的代码文件（本地尚不可读）

- `m5/scripts/01b_content_audit.py`
- `m5/scripts/06_g3_weights_crosscheck.py`
- `m5/scripts/03_run_baselines.py`（由 stub 实现）
- `m5/scripts/04_score_wrmsse.py`（由 stub 实现）
- 所有新 `src/m5research/` 模块和 `tests/` 变动，避免只评入口文件而漏掉核心逻辑。

现在主分支 `research/m5-study-v1` 上前两个路径不存在，后两个仍是占位实现。未得到代码，不能声称已完成 diff 审查/提交。建议工程师将 **仅源码/测试** 推送到独立 `research/m5-w1-g3-engineer` 分支，或者发送统一 patch / 原文件；由研究评审复核后再写回研究分支。不得推送 `runs/` 原始预测、凭据和 Kaggle 数据。

## 4. 应提交的精简证据包（允许入库）

提交 `m5/evidence/w1/` 下经过脱敏、可独立验证的摘要：

| 建议名称 | 最小内容 |
|---|---|
| `data-audit-summary.json` | 每项核验 PASS/FAIL、序列数量、零值比例、销量矩阵相等/异常计数、日历主键、聚合守恒差异 |
| `g3-weights-parity.json` | 官方参考文件 SHA-256、逐键 42,840 检查、Dollar_Sales 最大差、12 位权重最大差、分层和 |
| `g3-snaive-parity.json` | 评价轮标识、12 个官方层分数、12 个复算层分数、最大差、overall、软件/code SHA |
| `baselines-validation-summary.json` | 验证轮、两模型各 12 层分数、overall、误差分组、预测文件 SHA、运行配置 hash |
| `run-manifest-index.json` | run_id、git_sha、data/reference SHA、硬件环境 ID、训练 cutoff、预测时点、运行结果摘要 |

存放于本地的 `data_manifest.json` 与 `reference_sha256.json` 原版不一定可公开：先剔除用户名、绝对本机路径、Kaggle 认证片段和所有密钥。不要将完整销售矩阵、官方权重CSV、评测源 R 文件和大预测文件提交 GitHub。可只公开经过审计的文件名、SHA-256、大小和引用。

## 5. 需要额外关注的风险（独立评审意见）

1. 报告“价格连接 13 周全覆盖”应明确为**价格可用性检查**，不能把它误称为 WRMSSE 权重窗口。官方验证轮权重使用训练结束前 28 天金额；评价轮也为对应训练截止前 28 天。
2. 验证轮与评价轮的原始基线预测严格隔离；评价轮可作为事后 scorer 复算证据，但不能反向优化验证轮模型规则。
3. 官方分数匹配到 1e-10 很强，但仍需独立审查 **ID 对齐、预测生成逻辑、是否使用测试窗口标签、NaN/Inf 与零权重处理**。
4. 全部 42,840 条 scale 有效是工程师报告的经验事实。保留 synthetic all-zero / constant-history 单元测试。
5. SeasonalMean28 相对 SeasonalNaive7 下降 0.1090 WRMSSE（约 12.5%），是单个验证窗口观察，需六窗口稳健性确认；不能借此证明统计显著、经营价值或所有层级一致受益。

## 6. 六窗口回测执行

冻结 origin：1773、1801、1829、1857、1885、1913；各预测 28 天。对每窗口输出：

- Overall + Level 1–12 WRMSSE；
- 训练期定义的活跃度分组、零销售、间歇、长尾与高销量分组；
- 同一 horizon 的模型间配对差与排名反转；
- 是否由少数大权重序列主导；
- data/weights/scales 的正确 origin 计算或严格注明采用官方静态权重的偏差。
  
**特别注意**：前五个滚动窗口不可直接使用官方 d1913 对应的 `weights_validation.csv` 作为“每个窗口的官方评分”。每个 origin 需根据当时历史重算权重和 scale，或将静态权重结果明确标注为非官方口径的敏感性评价。否则六窗口结果不可作为严谨的时间外稳健性证据。

## 7. LightGBM 口径

继承已冻结的 [W1 G3 与 LightGBM 契约](w1-g3-and-lgbm-contract-v1.1.md)。允许开发并自测后提交 diff；**全量运行须先有 smoke 性能、无泄漏证据与资源预算检查**。训练目标不默改：每系列最近 365 个可训练目标日；lags 7/14/28/56/364；shift 后滚动；已知日历/事件/SNAP；静态 item/store 属性；未能证明已知的未来价格不进入主实验；CPU deterministic=true、force_row_wise=true、冻结线程/seed。

## 8. 结论

G3 在本地具有相当强的数值复现证据，可放行继续研究，但尚缺少**可读取的源码、独立检验日志与脱敏证据**。评审人不得将工程师自测结果写成自己已独立复算的结果。只有拿到文件并审查后，才对正式 `G3_INDEPENDENT_PASS` 签字。
