# M5 Hands-on — R10 v8 增量独立复审

**裁决：CLI P0 CLOSED / EXPLICIT_FORECAST_ORDER_HASH P0 CLOSED / ARTIFACT_PROVENANCE GATE FAIL / G3_FINAL_INDEPENDENT_SIGNOFF = NO-GO / W2_DESIGN_GO_ONLY**

## 1. 材料、工作区和验收范围

- 用户上传 `m5-w1-r9-fixes.patch`，**14,713 B**，SHA-256 **`9504f7cb5f74ea3798cf193ba08a962b7118b22b0d1fc25902fbd3c0c630d63a`**。一个提交，`6de0b39da7839504d5664900f38bd7b71a3dcbd2`，修改 03/03b/04/07 和新增 `test_cli_integration.py`。
- 实际将补丁 `git apply --check`、`git apply` 应用于 R9 的隔离工作区；全部成功。恢复仓库 `research/m5-study-v1` 原有配置文件 `m5/configs/accuracy_v1.yaml` 后，`python -m compileall` 通过，`pytest m5/tests` **35/35 PASS**。
- 最初在仅由 patch 重建的工作区运行有两项 `FileNotFoundError: configs/accuracy_v1.yaml`；这是评审隔离工作区**缺少基线仓库文件**，并非工程师新补丁的缺陷；已恢复规范仓库版本后消除，不能将首次失败用于否定 03/03b 的真实修复。
- 评审 Python 3.13.5，并非规范 Windows 3.11.15；未重新读取450MB M5原始销售数据，也未复训 30,490 系列 / 1300 轮 LightGBM。本报告不把工程师的 0.869700702 / 0.8470173528345969 / 0.58720 等真实数据报告冒充本方复算。

## 2. R9 三项 P0：实际关闭

| R9 问题 | v8 改动 | R10 证据 | 裁决 |
|---|---|---|---|
| 03 基线 CLI 元组 TypeError | `sales, id_frame = _load_sales(...)`，计算 `id_order` | 新增的真实 CLI 集成测试以 4 合成系列 × 740 日，Naive7 和 Mean28 各生成 28日 forecast 与 manifest，均 PASS | **CLOSED** |
| 03b full 模式 `sample_mask` UnboundLocalError | 分支前 `sample_mask=None` | 使用合成数据与 5 轮 LightGBM，完整训练、推理、落盘 forecast/manifest，PASS | **CLOSED** |
| 04 显式错误行序 SHA 假通过 | 区分 `canonical_hash`、`file_hash`；不匹配直接 SystemExit | 直接调用 04.main() 并提供 2-row 非平凡行序 fixture：错误哈希 **SystemExit**；canonical 保留 [100,200]；file 转为 [200,100] | **CLOSED FOR EXPLICIT CLAIMS** |

完整 35 项套件的合成 CLI 用例不证明完整 M5 数据复跑；只是编排入口与基础身份映射通过。

## 3. 07 工件身份链：新的阻断问题（独立运行入口反例）

### P0-A 合法的 LightGBM 工件被拒绝（模型名称体系不一致）

- `03b_global_lgbm.py:184` manifest 写 `model="global_lgbm_recursive"`。
- `07_six_window_summary.py` 的 RUN_IDS 键为 `lgbm_tweedie1300`。
- 新增检查是 `if model not in str(man_model): raise SystemExit`；`"lgbm_tweedie1300" not in "global_lgbm_recursive"` 为 true。
- **独立真实 07.main() 复验**：建立含 14 份合法结构 score + run_manifest 的完整合成 run 集（LightGBM manifest 使用**当前03b真正生成的** `global_lgbm_recursive`），汇总在 `M5A_W1_BT_lgbm_t1300fix_o1857` 直接 SystemExit。
- **整改**：manifest 分离 `model_family="global_lgbm_recursive"` 与 `arm_id="lgbm_tweedie1300"`，使 07 显式验证双方字段，不再用字符串子串判断。

### P0-B 可以伪造 round / scale_source / data/config hash 并输出 PASS

- 为相同14个 run 创建同源 score + manifest，形式上的 `model_label`、origin、ID哈希、预测路径正确，但 `score.round="FORGED"`、`scale_source="FORGED"`、`config_sha256="THIS_IS_FAKE"`，manifest 里 `round/scale_source` 也故意错误。
- 将 manifest 模型名称人为改成 `lgbm_tweedie1300` 以绕开上面的模型名称冲突（仅用于测试另一项缺陷）。
- **独立执行 07.main() 结果：exit code 0，`status=PASS`，六窗口汇总落盘**。07 只验证 manifest 配置/数据/id hash 非空，没有对来源哈希实际文件或 score 跨文件一致性进行验证。
- 将 o1885 的 `data_manifest_sha256` 单独改成另一个非空 SHA 后也正常 PASS。
- 尚未看到 `forecast.csv` 的真实 SHA-256 参与绑定：仅对 `forecast_file` 路径做字符串等价检查；预测内容可被修改而不触发汇总检验。
- **整改**：每个 run 把 `run_manifest_sha256`、`forecast_sha256`、`data_manifest_sha256`、配置/code SHA、origin、model_family、arm_id、round/scale_source 绑定到同一可核验结构；由 07 逐项读取并校验真实文件哈希。模型/round 参数和评价口径按预注册映射检查，而不是只要字段非空就算真。

### P1：损坏的 manifest 被默认为 legacy，无声重排

- `04_score_wrmsse.py:129–134` 在 `run_manifest.json` 存在但 JSON 解析出错时 `except Exception: declared=None`，从而把损坏元数据静默视为旧版无哈希预测。
- **独立 04.main() 复验**：manifest 包含 `{BROKEN_JSON`，没有显式哈希时照样进入 `forecast_row_order=file` 分支，继续评分。
- **整改**：若 manifest 存在但损坏，应中止；若根本不存在且明确允许 legacy，要求显式 `--legacy-file-order`，并标注 `LEGACY_ASSUMED_NOT_VERIFIED`，不应默认升级 G3。

### P1：effective forecast hash 未作为证据落盘

- `04.report["forecast_id_order_sha256"]` 写的是原始 CLI 参数值，而非从 manifest 自动协商后的 **effective declared hash**。因此自动协商成功时报告仍可能是 null；`07` 只比较规范 id hash，并未检查实际的预测行序声明及映射映像。
- 应存 `forecast_declared_hash`, `forecast_resolved_row_order`, `canonical_hash`, `source_manifest_sha256`, `row_mapping_sha256`, `forecast_content_sha256` 六元证据。

## 4. 既往 R5–R10 注入验收账本

| 研究问题 | 最新经独立合成负向测试结论 |
|---|---|
| 正权重无效 RMSSE 被零填充 | **CLOSED**（v3：正权重 NaN/0 中止） |
| 自算 Dollar_Sales 与官方差异仍 OK | **CLOSED**（v4：大误差中止） |
| 官方权重×2/NaN/负值、重复/缺失键未中止 | **CLOSED**（v5：入口级故障注入） |
| 12层逐日守恒、完整性 | **CLOSED for synthetic guarded paths**（v6–v7） |
| 不均衡零销量 smoke `KeyError` | **CLOSED**（v7：3473+9×3选3000） |
| event/SNAP 缺列/全空仍PASS | **CLOSED**（v7） |
| 03/03b 真实 CLI TypeError / UnboundLocalError | **CLOSED**（v8） |
| 04 错误显式行序 SHA 仍 PASS | **CLOSED**（v8） |
| 07 合法 LightGBM manifest 被误拒；伪造轮次/来源/hash被接受 | **OPEN / BLOCKING**（v8） |
| 04 manifest JSON 损坏时退回 legacy | **OPEN P1** |
| 确认性封存与新的未开封时间样本 | **OPEN governance gate** |

## 5. G3 签署与研究结论

**G3 数值证据（工程师报告）**：评价轮 sNaive `0.8470173528345969`，官方 `0.8470173528438624`，整体差约 `9.27×10^-12`；12层最大差 `1.49×10^-10`。过去提交的 42,840 键的验证轮 Dollar_Sales/weight 自算也报告精确匹配。本轮未取得完整原始销量数据独立重算，这些是 **`LOCAL_PARITY_STRONGLY_SUPPORTED`**，不能写作 `INDEPENDENT_RAW_DATA_REPLAYED`。

**六窗口（来自之前提交的摘要工件，经算术检查）**：Mean28 在六个指定历史窗口 overall `6/6` 优于 Naive7，描述性研究结果成立；并不能作为六个独立样本来推断未来泛化概率。修复后 LGBM o1857=0.81845、o1913 新规范排序后报告=0.58720；o1913 `POST_SELECTION_EVALUATION`，不能计入未开封的确认性成功。

**G3 独立最终签署**：**NO-GO**。虽然 scorer 对正权重尺度、参考金额和 weight、显式行序的主要防线已通过独立合成故障注入，但仍存在对损坏 manifest 的静默回退；07 的工件身份链既会拒绝合法 run，又会接受伪造 round/hash；更关键的是本评审没有独立读取 M5 原始矩阵逐键运行官方两轮完整评分。应输出范围限定为 `SCORER_DEFENSIVE_CHECKS_SUBSTANTIALLY_VERIFIED / LOCAL_NUMERIC_PARITY_STRONGLY_SUPPORTED / PROVENANCE_CHAIN_NOT_VERIFIED / INDEPENDENT_G3_SIGNOFF_PENDING`。

**W2**：设计/探索 GO；严格确认性 NO-GO（已查看六个历史 origin；freeze 尚未绑定未开封外样本、数据/code/config SHA 与评估方案）。

## 6. 修复出口（最小增量）

1. 03b 与 07 使用稳定的 `model_family` 和 `arm_id` 两个字段，增加 **合法真实 manifest 14-run 集成测试**。
2. 07 按原始文件内容计算 SHA 并比较；把模型、轮次、scale source、origin、权重方案、配置/data ID、预测文件哈希同 run score 严格关联。增加伪造 round/hash 的入口级负向测试。
3. 04 manifest 损坏不能回退成 legacy；可选 legacy 必须显式授权且另列证据等级。记录 resolved hash 和 row mapping。
4. 保留既往成绩资产，明确新 r006=0.58720 相对 r005=0.58547 的行序敏感观察需要固定随机性与特征行序的配对复检；**不可将差异解释为真实模型能力提升或退化**。
5. 完成 G0 Owner 关于数据许可和责任人的必要签署；在工具可访问的规范原始数据上独立重跑验证/评价两轮的 per-level、per-key 和各项哈希，再申请最终独立签署。

**结论：`PATCH_HASH_PASS / 35_35_TESTS_PASS / CLI_R9_P0_CLOSED / 04_EXPLICIT_ORDER_P0_CLOSED / 07_MODEL_NAMESPACE_AND_PROVENANCE_FAIL / G3_FINAL_NO_GO / W2_DESIGN_GO`。**
