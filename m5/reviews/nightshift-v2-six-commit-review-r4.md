# M5 Hands-on — R4 六提交补丁审计

**Decision: CHANGES_REQUESTED / DO_NOT_MERGE / G3_LOCAL_PARITY_STRONGLY_SUPPORTED**

**输入：** 对话上传 `m5-w1-g3-review-v2.patch`  
**SHA-256：** `5d9ba2b0bdebeac2f6ca5809e37a02462659d2beed18235749f6bf8742b68f99`  
**大小：** 150,896 B  
**提交顺序（经实读核验）：** `21a31cb` → `ef54be1` → `3fce96c` → `02b2da9` → `0b383ba` → `22a3027`

## 1. 评审范围与方法

- 六份 `git format-patch` 均在隔离 Git 工作区顺序通过 `git apply --check` 与 `git apply`。
- 在从补丁重建的源码基础上，结合规范分支原有基础模块/测试契约，运行 `pytest`：**20/20 PASS**；隔离环境 Python 3.13 / numpy 2.3.5 / pandas 2.2.3 / LightGBM 4.6.0，与工程师 Windows 3.11.15 / LightGBM 4.7.0 不同。因此这不是规范机的完整锁环境重跑。
- 新增独立合成反例：多序列日历行对齐、未来价格扰动、正权重无效 scale、极端零销量 qcut、日期窗口边界、错误 Dollar_Sales 的交叉校验、冻结哈希一致性。
- 复核证据 JSON 的六窗口 WRMSSE、12 层平均和 G3 全精度比较。
- 本环境没有官方 M5 原始销售矩阵及执行者完整模型日志；所有真实数据训练成绩仍列 `ENGINEER_REPORTED / ARTIFACT_INTERNALLY_CHECKED`，不能写为独立重跑。

## 2. 四项旧 P0 的逐项结论

| 旧问题 | v2 代码位置与实证 | 裁决 |
|---|---|---|
| 日历/event/SNAP 行顺序错位 | `models/global_lgbm.py:135,138,140` 改为 `np.tile`，`test_feature_alignment.py` 多序列逐行用例通过 | **FIX VERIFIED**（但测试 fixture 另有下述负索引缺陷） |
| 未来价格进入推理 | `global_lgbm.py:26-36,111,217` 对训练和推断统一裁至 origin week；独立替换所有 origin 之后的价格，价格敏感 dummy booster 的 28 日预测最大变化 **0** | **FIX VERIFIED**（针对“origin 周之后”价格泄漏） |
| fixed-round 仍丢弃尾部 28 天 | `03b_global_lgbm.py:120-128` fixed 分支直接 `train_rows=np.arange(train_x.shape[0])`，对应 365 个目标日/序列 | **FIX VERIFIED BY CODE**（完整正式训练尚未由审查环境复跑） |
| `np.nansum` 抹去 undefined | `scoring.py:203+` 对正权重 NaN 改为抛错；**但 `04_score_wrmsse.py:120-128` 抢先执行 `scores[~scoreable]=0.0`，依然掩盖正权重零/NaN scale** | **P0 STILL OPEN** |

特别说明：独立最小反例令 `Level1` 的 `w=1`、`scale=NaN`，实际传到汇总器的 `score=0.0` 并返回成功；不触发新 `wrmsse_from_levels()` 的 ValueError。

## 3. 新发现的 G3 P0：校验差异不阻断 PASS

`04_score_wrmsse.py:211-242` 的 `_crosscheck_weights()` 即使 `Dollar_Sales` 相差巨大，仍无条件返回 `{"status":"OK", "dollar_sales_max_rel_diff":...}`；`main()` 又无差异门槛而写 `status:"PASS"`。

独立注入错误官方 Dollar_Sales 的合成 12 层样本：`max_rel_diff=0.98998998998999`，函数仍返回 `status=OK`。

**整改：** 将逐键 `Dollar_Sales`、权重、层级总和和 reference SHA 的阈值检查作为**硬中止门禁**；异常立即失败，不给出 PASS。请增加负向单测证明 1%/100% 注入误差会被阻止。原始 M5 数据报告差异 4.32e-16 是好的证据，但不能替代防伪门禁测试。

## 4. P1 及实验治理发现

1. **价格审计窗口仍不正确**：`01b_content_audit.py:85` 仍只有 `d_num > origin-28`，缺少 `d_num <= origin`；所谓“13周覆盖”包含未来周。尽管新增逐周检查，输入窗口错误导致该检查并不检验正确的 28 天权重窗口。需限定两端并明确上市前缺价的合适业务定义。`E_event_columns_present=True` 仍是常量，不是实际验证；层级 `groupby().size().sum()` 不等于销量数值守恒。
2. **分层 smoke 采样未修**：`models/global_lgbm.py:273` 对高度零销量 `qcut(..., labels=[0,1,2], duplicates="drop")` 可报 `ValueError: Bin labels must be one fewer than the number of bin edges`，独立合成复现。
3. **特征测试存在假阴性**：`test_feature_alignment.py:11-12,70-71` 使用 `origin=380, targets=15..379`，前段 `lag_364` 下标为负。首行 `t=15` 的 `t-364=-349` 被 NumPy 解析为 `sales[:,51]`，即**未来日期**；测试断言也使用同一负下标，因而会放行错误。实际本次 M5 origins 都大于 364+365，当前正式回测无此问题；应将 fixture origin 提高到 >=730，并新增早期历史 lag 缺失显式处理用例。
4. **所谓 confirmatory 哈希不一致**：`03b:79-87,169` 的冻结 SHA 是 `params + fixed_rounds`，运行 manifest 的 `config_hash` 仅对 `params` 求 SHA，**两个值不同**；也未冻结数据、代码、origin、特征窗口，`--phase confirmatory` 仍允许已经开封的 origin=1913。需同一配置对象与相同哈希、预先冻结 untouched holdout 或禁止再称 d1913 为 confirmatory。
5. **轮次/行索引完整性不足**：`04:65-83` 允许 `--round evaluation --origin 1913` 自相矛盾组合；`04:165-172` 直接按 CSV 行位置读取 actuals，不按真实 ID 交叉核验。 `_weights_from_official()` 在建字典时没有拒绝重复键和缺失键的显式门禁。
6. **六窗口汇总缺少真实性验证**：`07_six_window_summary.py:54-68` 只使用硬编码 run ID 路径，没有读回评分文件中的 origin、model、round、data/config SHA 做一致性校验；`windows_with_mean28_first_of_simple` 逻辑也把含 LGBM 的总体排名当作“两个简单基线比较”。
7. **早期实验的胜出模型选择已受污染**：旧 Poisson/Tweedie/轮数设计在错位特征上选出，严格证据不成立；新的 d1913 仍 `POST_SELECTION_EVALUATION`，不能以简单重命名恢复未开封身份。
8. **发布安全**：`m5/evidence/` 多份 JSON 保留了本机绝对 `E:\\m5-data\\...` 路径。未见 token 原文，但应在公开上传前去除主机路径，改为相对路径/文件类型及 SHA。M5 原始数据、官方权重 CSV/R 文件、令牌和预测大文件绝不可提交。

## 5. 数值证据验收（内部计算一致；非重新训练）

官方评价轮 sNaive：`0.8470173528438624`；v2 的 `g3_snaive_reproduction_full_precision.json` 均值为 `0.8470173528345969`，最大 12 层差 `1.4909784518124525e-10`，已复核内部数值一致。

六个自定义回测窗口的 Naive7 与 Mean28 整体结果、12 层均值均自洽，Mean28 **6/6** 胜 Naive7（这不是独立新市场概率结论）。

**修复前/修复后 Tweedie LightGBM：**

| Origin | 修复前（问题代码） | 修复后 | Mean28 | 修复后相对 Mean28 |
|---:|---:|---:|---:|---:|
| 1857 | 0.93579 | **0.8184498373** | 0.7175785836 | **仍差约 14.1%** |
| 1913 | 0.91638 | **0.5854685359** | 0.7606716964 | **低约 23.0%（POST_SELECTION）** |

复核 `six_window_baseline_summary.json`：o1857 修复版在 12 层中仅 3 层胜 Mean28，o1913 则 12 层全部胜。由此有**显著的窗口异质性迹象（描述性用语，不是显著性检验）**。任何“普遍优于强基线”结论均不成立。

**因果归因边界：** P0 batch 同时改变了 calendar row-order、未来价格处理和 fixed-1300 训练样本规模。修复前后差异不能单独归因于“calendar 修正”，需要分别消融 `C calendar`、`P price`、`R refit` 并提供交互项，同时封闭未开封测试集。

**机能实测：** 工程师报告 RSS 峰值 4.76GB、train 129.6s、infer 13.9s；未得到 psutil 采样原件，登记为 `ENGINEER_REPORTED`。

## 6. 精确整改出口

**P0 blocking**

- `04` 在清零任何 unscoreable 行之前检查 `(~isfinite(scale) | (scale<=0)) & (weight>0)`；加模拟非零权重 NaN/0 scale 必报错回归测试。
- `_crosscheck_weights` 将异常差异改为直接失败；阈值/舍入方法冻结，禁止始终输出 `status OK`。

**P1 required before release**

- `01b` 正确封闭 origin 前 28 日窗口、检查事件/SNAP 真正存在，按 12 层数字销量守恒。
- `stratified_sample_mask` 使用零值单独桶和可退化的活动量分层，保证多零分布不崩。
- 修复 `test_feature_alignment` 负下标“自证正确”问题。
- Phase 判据冻结包括目标函数、轮次、origin、数据版本、源码 SHA，区分探索 / 后选择评估 / 未开封确认。
- `04` 拒绝 round-origin 矛盾并核验 actuals/train/id 对齐；`07` 核验分窗 origin/model/评分口径与输入文件 SHA。
- 六窗口与修复前后报告保留每层、分组、预测文件和哈希；分开验证点预测与决策收益，不开展未经预注册的事后模型挑选。

**决策：** G3 本地数值复现证据强；独立完整运行未取得。 `20/20` 单测通过、部分 P0 确实关闭，但**两处评分器硬门禁仍失效**。暂不合并代码、暂不称 `G3_INDEPENDENT_PASS` 或 W2 正式对照模型完成。

> 本审查分支仅收录源代码审计报告，不在发现 blocker 时把整个未修复补丁标记为已批准部署。
