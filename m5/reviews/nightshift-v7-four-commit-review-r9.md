# M5 Hands-on｜v7 四提交补丁 R9 独立验收

**结论：CHANGES_REQUESTED / G3_FINAL_SIGNOFF_NO-GO / W2_DESIGN_GO_ONLY / DO_NOT_MERGE**

## 0. 材料完整性和复核边界

- 上传附件：`m5-w1-r8-fixes.patch`，实际 83,576 字节；SHA-256 **`5327cc237308b3b57e1a18a27eacc7a3412a75b3943842685cee41901734a0ab`**。
- **四个提交**：`c29225f`、`6ecdead`、`45d5252`、`dbaddad`。本附件包含前两轮 v5/v6 改动，**不是四个全新 R8 commit**。
- 在此前 v6 隔离工作区上依序应用 **`45d5252`、`dbaddad`**：两次 `git apply --check` 与 `git apply` 均成功；前两提交已存在，未重复套用。
- `python -m compileall` PASS；`pytest m5/tests` **32/32 PASS**。
- 验收环境 Python 3.13.5 / numpy 2.3.5 / pandas 2.2.3 / LightGBM 4.6.0，**不是**工程师 Windows Python 3.11.15 / LightGBM 4.7.0；本轮无完整 M5 原始销售矩阵，未重训 11M 行 LightGBM，原始实测成绩仍是 `ENGINEER_REPORTED`。
- 额外执行：真实 CLI 入口合成数据测试（03、03b）、不均衡抽样、日历完整性负向、12 层删除负向、07 工件身份伪造、04 明确传入错误行序哈希。

## 1. 旧 R8 四项整改的有效部分

| 范围 | 独立复核 | 裁决 |
|---|---|---|
| **极不均衡 smoke** | 3,500 个序列，一组 3,473，另外九组各 3，全部零销量：**抽得恰好 3,000**、计数合计 3,000、同 seed 可重现 | **FIX VERIFIED** |
| **event/SNAP 缺失/全空** | 构造 1,969 行日历、仅 `snap_CA` 且全空、事件全空：`01b.main()` **exit 2 / BLOCKED** | **FIX VERIFIED**（需补齐全部事件 schema 的确切列定义） |
| **12 层守恒完整性** | 删除 `Level12` 后调用 `assert_aggregation_conservation`：**ValueError: 11 levels; expected 12** | **FIX VERIFIED**（已覆盖完整性+数值） |
| **07 伪造 model_label** | `model_label` 不匹配配置映射可拒绝；完整 score 工件必须含 `canonical_id_order_sha256` | **PARTIAL**；其余身份声明尚未核验 |

## 2. 新发现的两个 P0：预测生成脚本完全无法正常运行

### P0-A：`03_run_baselines.py` 返回值协议错位

- `m5/scripts/03_run_baselines.py:53`: `sales = _load_sales(train_file, args.origin)`。
- 同文件 `:98-109`：`_load_sales()` 改成返回 **`(sales_matrix, id_frame)` 元组**，但 `:57` 仍将 `sales` 传给 `seasonal_naive_7()`。
- 独立调用真实 CLI，使用仅两条序列的 740 日合成输入：**exit=1**，`TypeError: tuple indices must be integers or slices, not tuple`。
- 后续 `:78` 的 `id_order` 还没有赋值，即使修复第一个错误，保存 manifest 时也会再遇 `NameError`。
- **修复**：`sales, id_frame = _load_sales(...)`、`id_order = id_order_sha256(id_frame)`。需要通过两系列 synthetic CLI 真正生成 28 日预测 CSV 与合法 manifest 的双模型端到端测试，测试不得仅调用 `_load_sales`。

### P0-B：`03b_global_lgbm.py` 的 full 模式引用未赋值 `sample_mask`

- `m5/scripts/03b_global_lgbm.py:130-140`：只在 `--mode smoke` 分支赋予 `sample_mask`；`--mode full` 分支仅赋 `n_used`，后续却无条件调用 `build_feature_table(..., sample_mask)`。
- 独立调用真实 CLI，740 日 × 2 series 最小合成输入、`--mode full --fixed-rounds 1`：**exit=1**，`UnboundLocalError: cannot access local variable 'sample_mask'`。
- **修复**：分支前 `sample_mask = None`；固定轮数模式不要无条件创建空 `dval`，并在 synthetic full 流程完成至少一轮 LightGBM 的真正 CLI 集成测试，验证保存 forecast + manifest（不必跑 1,300 轮）。

**结论：当前代码不符合“READY_FOR_TEAM_EXECUTION”，既无法现场再生简单基线，也无法现场 full 运行 LightGBM。之前两轮的数值工件不能代表新代码可正常复跑。**

## 3. 评分器与汇总仍存在关键假通过

### P0-C：显式预测行序哈希不被验证

- `04_score_wrmsse.py:68-70` 提供 `--forecast-id-order-sha256`，宣称绑定预测行序。
- `:113-124` **仅在参数为 `None` 时执行 file-order 兼容协商**；显式传入任意不相等的哈希，不与 `id_order_sha256(id_frame)` 比较，就将预测视为 canonical。最终 `:194` 又把标准 canonical hash 写进成功报告，可能**替错误预测证明身份**。
- 独立合成模拟 `04.main()` 入口（用 monkeypatch 提供简化 12 层矩阵，**只隔离身份控制而非验证实际 42,840 序列**）：标准哈希为 `aaaa...`，显式预测哈希为 `bbbb...`，返回 **exit 0 / status PASS**。
- **修复**：若声明了预测行序 SHA，必须严格与 expected canonical SHA 相等，或使用可信的明确 row-map 转换后再核验；如果未声明则不得静默猜测并提升为官方 G3 parity。在 score 中记录声明 hash、实际 hash、映射 SHA 和校验 verdict，并和 run manifest + forecast SHA 对照。

### P1：07 只校验 model_label，未校验 round/scale_source/hash

- `07_six_window_summary.py:65-87` 已校验 `origin`、`model_label`、共同 `canonical_id_order_sha256` 和 predictions 路径包含 run_id。
- **独立伪造试验**：为全部 14 个所需 run 提供正确 model_label / origin / 同一个任意 canonical hash / 路径，却填写 `round=INTENTIONALLY_INVALID`、`scale_source=INTENTIONALLY_INVALID`、虚假的 data/config SHA。实际 `07.main()` **exit 0 / PASS**，接受全部六窗口。
- 此流程没有反查 `run_manifest.json`、预测文件真实 SHA 或原始 G0 数据 SHA；统一的假的 ID hash 也能自洽通过。
- **修复**：绑定每个 `score.json` 与其 `run_manifest.json` 的 model、origin、round、data_manifest SHA、config SHA、canonical ID-order SHA、forecast 文件 SHA；规定缺失即失败，不只检查路径包含字符串。

### P1：确认性门禁尚未完全冻结

- `03b` 只拒绝 opened origin `{1857,1913}`，但六窗口 `1773,1801,1829,1857,1885,1913` 的指标全部已被分析，`1941` 官方评价分数也已公开使用。
- `--phase confirmatory` 的参数哈希仅对 params+fixed_rounds；未把数据、代码和 origin 的 hash 纳入同一冻结凭证。把其他已经暴露的历史 origin 称为未开封确认性窗口欠严谨。
- **修复**：封存真正未利用的时间点/外部新样本，固定来源 SHA、git commit、预测起点、分组规则及统计方法。以往窗口仅为探索/回顾性检验。

### P1：来源与文档登记口径不一

- `load_id_frame()` 实际采用**按 ID 组合字段排序**的 canonical 顺序，但 `04` 的 report 字段 `canonical_id_order="official_train_file_order ..."` 仍声称采用官方源文件顺序。`_load_wide_canonical` 确实按 sorted ID 重排，实际计算可保持一致，但元数据文字错误且忽略显式预测 SHA 的风险需修复。

## 4. 六窗口证据和 G3 边界

v7 附件只携带更新的 `m5/evidence/six_window_baseline_summary.json`，并未提供声称重刷的全部 14 份 run score / 预测文件的原件。汇总 JSON 已复算整体与 12 层平均一致（浮点精度内）：

| Origin | SeasonalNaive7 | SeasonalMean28 | LGBM (post-fix, only two origins) |
|---:|---:|---:|---:|
| 1773 | 1.411421 | 1.204080 | — |
| 1801 | 1.112062 | 1.066091 | — |
| 1829 | 1.021062 | 0.851218 | — |
| 1857 | 0.888939 | 0.717579 | 0.818450 |
| 1885 | 0.922781 | 0.658369 | — |
| 1913 | 0.869701 | 0.760672 | 0.585469 (POST_SELECTION) |

**裁决**：Mean28 在六个自定义历史回测窗口整体 `6/6` 胜过 Naive7、平均 WRMSSE 约 `0.876335` vs `1.037661`，属于**本地提交证据的描述性一致性**；不能由此推出未来普遍优势或库存经济价值。W2 的理论与探索性设计可启动，严格确认性比较仍未开放。

本轮没有重新运行 450 MB 完整原始 M5 数据，无法签署 `G3_INDEPENDENT_PASS`；用户报告的真实数据分数如 `0.8470173528345969` 仍保留工程师来源并有先前的官方 `Scores and Ranks` 参考交叉校验。

## 5. R9 退出清单（优先级从高到低）

1. **P0 入口修复**：`03_run_baselines` 正确拆包并计算 ID hash；`03b --mode full` 初始化 sample_mask；加真实 CLI 端到端合成集成测试。
2. **P0 行序声明硬门禁**：`04` 显式预测 SHA 错误必须在评分之前拒绝；不明来源的 legacy forecast 应要求可追溯原始排序及 SHA，不得仅猜测。
3. **P1 07 工件链**：核验 round/model、data/config SHA、run_manifest SHA、预测文件 SHA，拒绝缺失/自相矛盾，不要仅靠标签和路径。
4. **P1 实验封存**：确认性 origin、数据/代码/配置、预注册指标统一冻结；已查看的六窗口不再当未开封测试。
5. 保存 v7 完整的安全增量工程证据，用户自行保管完整原始 Kaggle 数据与官方第三方评分资料；公开仓库只上传必要源码和脱敏报告。
6. G0 的规则接受人和数据保管人仍需 Owner 正式签署。

**最后裁决：`PATCH_VERIFIED / 32_OF_32_UNIT_TESTS_PASS / R8_FOUR_FIXES_SUBSTANTIALLY_IMPLEMENTED / THREE_P0_EXECUTION_AND_ID_INTEGRITY_BLOCKERS / DO_NOT_MERGE / W2_DESIGN_GO / CONFIRMATORY_NO_GO`。**
