# M5 Hands-on — v6 增量 R8 独立复审

**裁决：PARTIAL_ACCEPT / R7_P1_NOT_ALL_CLOSED / CHANGES_REQUESTED / DO_NOT_MERGE / W2_DESIGN_GO_ONLY**

## 1. 证据来源与完整性

- **输入**：本会话上传 `m5-w1-r7-fixes.patch`，本地实际路径 `/mnt/data/m5-w1-r7-fixes.patch`。
- **SHA-256**：`74636fd16ccc7431dc6e71aac7ba5901146cd1c857beae6c154d07a40237db2b`；大小 **18,989 B**，与工程师申报一致。
- **Commit**：`6ecdead2ec5112e890baa5a86fba4612e4e00385`；增量修改六个路径（01b/03b/04/global_lgbm/scoring/test_guards）。
- 在上一轮 v5 (`c29225f`) 源码隔离快照上 `git apply --check` 与 `git apply` 均通过；`compileall` PASS；该快照 `pytest m5/tests` **28/28 PASS**。
- 与本机工程师 Windows Python 3.11.15 的运行环境不同；本轮未取得完整 M5 原始 CSV，也未重新训练 11M 行 LightGBM。所有真实数据分数仍属 `ENGINEER_REPORTED`，历史工件仅做内部算术核验。
- 新增独立负向测试脚本及 JSON 结果由评审方保存在会话附件中，未直接推送原始 Kaggle 文件/官方 R 权重/大预测文件。

## 2. 逐项裁决（不以 28/28 替代负向测试）

| R7 项 | v6 实际修改 | 独立反例 / 审查观察 | 结果 |
|---|---|---|---|
| **逐日数值守恒** | `scoring.py::assert_aggregation_conservation`；`04` 在实际评分前调用 | 对 Level12 的某日销量+1，**正确抛 `ValueError`**；但删除整个 Level12 后，传入其余11层仍**正常返回**，因为函数只遍历 `aggregated.items()`，不检验12层完整性；并且当前 `04` 只调用于训练历史矩阵，不检查 actual/forecast | **CORE PASS / COMPLETENESS P1 OPEN** |
| **3,000分层 smoke** | `stratified_sample_mask` 自适应配额 + 补足，返回 mask 与 counts；03b 有下限2700 | 平衡6k样本能取3,000；但**极不均衡3,500样本**（一个store×cat cell占3473，其他9格各3，全部零销量），补足分支直接 `KeyError: ('s0','c1',0)` | **P1 FAIL** |
| **事件/SNAP真实性** | `01b` 检查 event_name 前缀、非空计数及已有 SNAP 列的二进制值 | 独立生成合法日期覆盖、含`event_name_1`但全部空、**仅有`snap_CA`且全部NaN、缺snap_TX/WI** 的测试数据，直接运行 `01b.main()` 返回 **exit=0, status=PASS**；`if not snap_cols` 只拒绝零列，`dropna` 使全空列被当作二进制合法 | **P1 FAIL** |
| **Canonical ID 三方对齐** | 04 `_assert_ids_aligned(train,actual)` 增加 `(item_id,store_id)` 唯一性；注册 `official_train_file_order` | 将 train 和 actual **同时按相同方式换序**，`_assert_ids_aligned` 仍放行；`load_id_frame(data_dir)` 的原官方顺序没有被比较，且 forecast.csv 不带ID。组矩阵仍可能按原顺序聚合错配 | **P1 FAIL** |
| **确认性窗口门禁** | `OPENED_ORIGINS={1857,1913}`，已列开封窗口拒绝 confirmatory；post_selection 标注 | 1857/1913 阻断语义可从代码成立；但其他历史窗口1773/1801/1829/1885的结果也已查看，仍可声明 confirmatory；冻结哈希只覆盖 params+rounds，**未绑定数据 SHA、代码 SHA 和 origin** | **PARTIAL; P1 OPEN** |
| **持久化测试** | 新增 `m5/tests/test_guards.py`，总测试22→28 | conservation 正反用例执行真正 guard；但 `test_weight_integrity_rejects_doubled_weights` 与 `test_weight_integrity_rejects_duplicate_keys` **没有调用 `_validate_official_weights`**，只是对手工造的错误数据做算术断言；`test_event_field_check_is_real_not_hardcoded` 也未调用 `01b.main()` | **COUNT PASS / COVERAGE WEAK** |
| **07 身份及哈希关联** | v6 没有修改07 | 合成编造14个score.json，写入对应origin/预测路径、但所有 `model=INTENTIONALLY_WRONG_MODEL`、`round=INTENTIONALLY_WRONG_ROUND`、`scale_source=FAKE`，直接运行 `07.main()` 仍返回 **PASS** | **P1 OPEN（遗留缺陷）** |

### 补足抽样的确切 bug

`global_lgbm.py:315` 把 `counts` 的键存成 `"store|cat|activity"` **字符串**，但 `global_lgbm.py:320` 使用 groupby `key`（**元组**）判断缺额，`global_lgbm.py:329` 又执行 `counts[key] += ...`，一旦进入补足即 `KeyError`。大部分平衡数据不会触发补足，因此原合成测试28/28无法发现。

建议统一使用元组作为内部 key，最终 JSON 序列化时才转换字符串，并每次确切核对 `mask.sum()==min(target_total,n_series)`、`sum(counts.values())==mask.sum()`、索引不重、固定 seed 同结果。对严重不平衡/退化分层也必须测试。

### ID 对齐的正确契约

**“官方文件顺序”是可接受的 canonical 定义，不必强行按 item/store 字典序排列。** 但必须建立**可信的基准序列**：`load_id_frame(data_dir)` 读出的官方行顺序与选定 `train_file`、`actual_file` 逐一比较；预测文件若无ID则必须绑定该顺序的 SHA / row_mapping，并在 scorer 中交叉检查。仅验证 train==actual 或二者键唯一，不构成三方对齐。

### 聚合数值守恒的正确边界

新 guard 确实对**现存每层的逐日总量**进行核算，测试显示被改坏的日会失败，核心功能值得保留。但若缺层会假放行，说明还需 `set(aggregated)==set(LEVEL_NAMES)==set(groups)`、每层行数符合 group 数量，以及每层 partitions 覆盖所有底层索引恰好一次。建议对 train、actual、forecast 三种矩阵各执行一次，防止聚合/ID错误被仅检验训练矩阵掩盖。

## 3. 六窗口研究结果与适用范围（以前提交工件复算）

| origin | Naive7 | Mean28 | Mean28在12层的胜出数 |
|---:|---:|---:|---:|
| 1773 | 1.411421 | 1.204080 | 12/12 |
| 1801 | 1.112062 | 1.066091 | 7/12 |
| 1829 | 1.021062 | 0.851218 | 12/12 |
| 1857 | 0.888939 | 0.717579 | 12/12 |
| 1885 | 0.922781 | 0.658369 | 12/12 |
| 1913 | 0.869701 | 0.760672 | 12/12 |

六窗口均采用各自历史回测起点，提交工件中 Mean28 总体 **6/6** 优于 Naive7；六窗算术平均 WRMSSE Naive7 **1.0376609296**、Mean28 **0.8763347492**，相对均值低 **15.55%**。L1–L5 各胜5/6，L6–L12 各胜6/6。只接受 `OBSERVED_IN_SIX_CUSTOM_WINDOWS / GENERALIZATION_NOT_CONFIRMED`；这不是独立自然实验、未由评审环境复跑450MB原始M5数据，也不支持库存价值因果主张。

修复后 LGBM：o1857=0.81845（逊于Mean28 0.71758）；o1913=0.58547（优于Mean28 0.76067，但该窗口已开封，属 `POST_SELECTION_EVALUATION`），不能宣称跨窗稳定优势，更不能直接把三项修复联动效果归因于单一 calendar bug。

## 4. R8 退出与 W2 决策

- `R8_ACCEPTED`：补丁真实性、28项现有测试、逐日不守恒异常捕获、平衡样本 smoke、事件列已从硬编码变成真正读取、1857/1913 已开封拦截、原先 reference-weight P0 保持关闭。
- `R8_BLOCKED_P1`：极端不均衡 smoke `KeyError`、不完整事件/SNAP仍PASS、canonical第三锚点缺失、聚合守恒12层完整性、07 model/round/hash真实性、确认性源码+数据+origin冻结、弱负向测试。
- **不批准整包合并 `research/m5-study-v1`，不签署 G3_INDEPENDENT_PASS，不放行 W2 confirmatory。**
- **W2 设计与探索性研究可继续**：预注册强基线 Mean28 与 LGBM比较；观察节日错位、四周均值平滑、层级差异、C/P/R三组修改分离消融；所有已开封窗口按描述性/探索性分类，确认性需要真正未使用的时间窗口或新数据。
- 研究负责人/Owner 仍需登记 G0 的 Kaggle 规则接受人与数据保管人。

**本轮总结：`R8 PATCH VALID / 28_TESTS_PASS / CORE_NUMERIC_CONSERVATION_PASS / 3000_SMOKE_UNBALANCED_FAIL / EVENT_SNAP_INCOMPLETE_FALSE_PASS / CANONICAL_ID_TRIPLE_CHECK_FAIL / 07_ARTIFACT_IDENTITY_FALSE_PASS / CHANGES_REQUESTED`。**
