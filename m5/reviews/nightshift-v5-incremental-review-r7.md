# M5 Hands-on｜v5 增量补丁 R7 独立复审

**裁决：P0_REFERENCE_INTEGRITY_CLOSED / P1_PARTIAL / W2_EXPLORATORY_ONLY / DO_NOT_MERGE_FULL_PATCH**

## 1. 输入与测试环境

- **输入**：用户上传 `m5-w1-r6-fixes.patch`，增量于 v4 commit `846341a`。
- **字节数**：12,396；**SHA-256**：`63e25804f5861829ef512f09f60c3d8ed16e05552f1b4b8df63222523e99ce64`（本轮独立读取、验证与申报一致）。
- **提交**：一个，`c29225f0347e3373393b77986aaaa810a4602a1c`；仅修改六个路径，**没有新增测试文件**。
- **复现工作区**：先前已应用八提交的隔离 v4 worktree，`git apply --check` 与 `git apply` 成功。
- **验证**：`compileall` PASS；原测试套件 **22/22 PASS**。该运行环境不是用户 Windows Python 3.11.15/LightGBM 4.7.0，不能称为用户本机一致性复测。
- **增加独立入口级合成故障注入**：`m5_r7_acceptance.py` 在 04 的 `main()` 上测试正常参考及六种恶意参考输入。所有 7 种预期通过；脚本仅使用合成参考、monkeypatch 的小型维度，不假装完成 42,840 真序列的整套实际复算。
- 当前没有在评审端重新取得完整 Kaggle 数据 / 执行真实 LightGBM；原始成绩仍记 `ENGINEER_REPORTED / EVIDENCE_ARITHMETIC_VERIFIED`。

## 2. R6 指明的 reference P0：CLOSED

`m5/scripts/04_score_wrmsse.py:104–111` 在评分前先构造预期完整键集，并调用 `_validate_official_weights(official, expected_keys)`。

该函数（196–267）核对行数、唯一聚合键、完整键集、有限非负权重、每层总和≈1 和 `Dollar_Sales` 推导的每行权重。随后旧的自算 Dollar_Sales 交叉核验照常运行，异常会强制中止。独立入口故障注入结果：

| 合成输入 | 04.main() |
|---|---|
| 正确参考 | PASS |
| `weight×2`，`Dollar_Sales` 未改 | SystemExit: Level1 权重和=2 |
| `weight=NaN` | SystemExit |
| `weight=-1` | SystemExit |
| 重复聚合键 | SystemExit |
| 丢失聚合键 | SystemExit |
| `Dollar_Sales`×99.9 | SystemExit: 自算金额与参考不一致 |

**P0_REFERENCE_WEIGHT_CORRUPTION_FALSE_PASS = CLOSED**（就上述实际测试场景）。

**边界提示**：`Dollar_Sales` 列未列为强制列。`_validate_official_weights` 会接受一个只有 normalized `weight` 的参考文件；后续 `_crosscheck_weights()` 先直接访问缺失的 `Dollar_Sales`，导致非结构化异常。虽然不会正常返回 PASS，但正式 official 模式应显式要求高精度五列参考并给出可审计错误。回归应将这些负向场景纳入版本库；本增量 commit 未新增相关测试。

## 3. R6 P1 逐项裁决

| 项目 | R7 事实 | 裁决 |
|---|---|---|
| `01b` 28天窗口 | 代码已限定 `origin-28 < d <= origin`；报告记录周 11609–11613 共五周，缺失计数0 | **CLOSED**（源码时间边界）；实际原始数据未由评审环境复读 |
| 价格“官方数据止于11613” | v5 删除虚假的文件边界注记，明确收回原结论 | **CLOSED**（错误表述撤回） |
| 高零销量 `qcut` 异常 | 无标签 `qcut` + `codes` + 退化回退，独立 70% 零/all-zero 不再报错 | **CLOSED AS NO-CRASH** |
| 3,000条分层 smoke 契约 | 合成完整 30,490 条、30 store-category groups、70% 零时只选 **1,020** 条（全零也1,020），不是冻结目标约3,000 | **P1 OPEN**（需要重分配采样配额并输出抽样数量/分层分布审计） |
| `07` simple-only 获胜窗口 | 现只比较 Naive7/Mean28，产物 6/6 且独立算术一致 | **CLOSED** |
| `01b-C` 数值守恒 | 只检查 `groupby(...).size().sum()==N`；此为分组计数，不证明每日聚合的数值和 | **P1 OPEN** |
| `01b-E` 事件字段 | `E_event_columns_present = True` 仍硬编码；未验证 event_name_1/2 等字段真的存在或有效 | **P1 OPEN** |
| `04` canonical ID 对齐 | `_assert_ids_aligned(train, actual)` 相互对齐，但未检查它们与 `load_id_frame()` 的 canonical 底层排序；双文件同换序仍可能假通过 | **P1 OPEN** |
| `07` 模型及哈希归属 | 仅核对 origin + prediction 路径包含 run_id，未核验 score 的 model/round/scale_source/data/config hash | **P1 OPEN** |
| `03b` confirmatory 身份 | hash 在 `params+fixed_rounds` 内一致；未冻结源码 SHA/数据/预测起点，且 o1913 已开封但 CLI 仍允许声明 confirmatory | **P1 OPEN** |

补充：当价格文件真的缺少某预测起点周时，`01b:123` 的 `file_last_week` 仍由**截断后的**价格表计算，不能推断真实源文件末周。现在 d1913 报告无缺价时该分支不执行，但其他滚动 origin 仍有误判风险。建议用未过滤的源表取得末周、并区分有销量却无价与本就无销量。

## 4. 六窗口研究裁决（证据算术独立核对）

| origin | Naive7 | Mean28 | Mean28 优势 |
|---:|---:|---:|---|
| 1773 | 1.411421 | 1.204080 | 0.207341 |
| 1801 | 1.112062 | 1.066091 | 0.045970 |
| 1829 | 1.021062 | 0.851218 | 0.169844 |
| 1857 | 0.888939 | 0.717579 | 0.171360 |
| 1885 | 0.922781 | 0.658369 | 0.264413 |
| 1913 | 0.869701 | 0.760672 | 0.109029 |

六窗 Mean28 **6/6 总体胜出**。总体 WRMSSE 的六窗算术平均：Naive7 **1.0376609296**，Mean28 **0.8763347492**；平均差 **0.1613261804**、相对 Naive7 平均降低约 **15.55%**。分层计数：L1–L5 各为5/6，L6–L12 各为6/6。这是限定于六个自定义历史窗口的**描述性证据**，不是六次独立自然实验，不支持总体泛化概率或库存收益判断。

`six_window_baseline_summary.json` 的 simple-only 计数字段现已更新为 6；但未附 13 份完整原始运行输出供独立复算，也未重算源 CSV。

## 5. 建议的最小整改交付（不要求重新上传八提交）

1. 在 `04` official 模式要求 `Dollar_Sales` 五列、严格三方 canonical ID 顺序比对，增加 official-key/weight 损坏的**持久化负向测试**。
2. 在 `01b` 和 `04` 调用正式的 `aggregate_to_levels` 并逐层验证**逐日**合计守恒；真正检查事件/SNAP 列是否存在与合法。
3. Smoke 抽样严格按目标 `min(3000,N)` 分配 quotas；对全零和大量 ties 的样本保底足量，不允许以一个 1020 样本冒充 3000 smoke；固定 seed，登记各层实际抽样量。
4. `07` 检查运行 manifest 与 score 的 origin、model、round、scale_source、data/config/forecast SHA，拒绝缺失或不一致；不要只通过路径名字符串判断。
5. `03b` 对 post-selection d1913 强制标签与 confirmatory 阻断；未来确认性测试需冻结数据与代码快照及未开封测试窗口。

## 6. R7 总结

**技术门槛结论**：`REFERENCE_P0_CLOSED / 22_SCaffold_TESTS_PASS / SIX_WINDOW_DESCRIPTIVE_ACCEPTED`。**发布门槛结论**：`G1_NUMERIC_CONSERVATION_AND_EVENT_CHECK_OPEN / 3000_SMOKE_CONTRACT_OPEN / CANONICAL_ID_AND_RUN_PROVENANCE_OPEN / CONFIRMATORY_NOT_SEALED / DO_NOT_MERGE_ALL / W2_DESIGN_GO_ONLY`。

允许开始 W2 **理论方案、预注册和探索性**方法研究；不允许把当前两个 LGBM 窗口宣称为确认性胜出，不允许 G3=INDEPENDENT_PASS 或把完整补丁直接合并到规范研究分支。G0 责任人签署仍待 Owner。
