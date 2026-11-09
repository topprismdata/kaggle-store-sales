# M5 Hands-on｜R11 独立复审：v9 两提交补丁

**最终裁决：CHANGES_REQUESTED / G3_INDEPENDENT_FINAL_NO-GO / W2_DESIGN_GO**  
**本轮性质：来源与复现审计；不是完整 M5 原始数据重训。**

## 0. 交付校验和方法

- 实际附件：`m5-w1-r10-fixes.patch`，**34,790 字节**，SHA-256 `f6f24079c6ad52dee7ced344cb3ea4b4fe80be309f0b3870de022ce45c521418`，与工程师一致。
- 补丁内容共两提交：`6de0b39da7839504d5664900f38bd7b71a3dcbd2`（R9 CLI，已应用于 R10 基线）；`4fe98a0c8a977c7ba19852bbcf4f84b91fa39ce5`（本轮新增来源门禁）。
- 在上轮 R10 独立代码快照上**仅应用第二提交**：`git apply --check`、`git apply` 成功；`compileall` PASS，`pytest m5/tests` **35/35 PASS**（隔离环境而非 Windows Python3.11.15）。
- 运行额外 11 个独立 14-run 合成工件案例，直接调用 `07_six_window_summary.main()`；另复测 `04_score_wrmsse.main()` 的 explicit SHA/文件行序/损坏 manifest。
- 真实 M5 数据、14 份原始 run 的 forecast/score/manifest 文件、完整 42,840 序列独立评分均不在本附件中；这部分依然是工程师报告，**不能称作独立完整 raw-data replay**。
- 本地复现脚本：`/mnt/data/m5_r11_independent_checks.py`；实际结果：`/mnt/data/m5_r11_independent_results.json`；裁决快照：`/mnt/data/m5_r11_acceptance_result.json`。

## 1. 已关闭的 R10 具体缺陷

1. **合法 LGBM 工件误拒**：`03b` 新增 `model_family=global_lgbm_recursive`、`arm_id=tweedie_fixed1300`，`07` 精确匹配两字段；独立构造 14 份具有这两个字段且 config/data/forecast 哈希正确的工件集，**07 正常返回 PASS**。旧的 model 字符串包含规则不再存在。
2. **forecast 内容未核验**：`03/03b` 写入 `forecast_sha256`，07 重新读实际 `forecast.csv` 计算哈希；修改预测文件任一内容，**SystemExit**。已关闭。
3. **存在但损坏的 run_manifest JSON 静默回退 legacy**：`04` 现在 SystemExit；独立入口级合成测试证实。显式错误行序 SHA 继续中止，canonical/file 两种正确声明的行序置换行为保持通过。
4. **已存在且被修改的 config/data manifest 文件**：07 能拒绝与运行 manifest SHA 不一致的实际文件。该项属 **present-and-corrupted** 的完整性验证，不等于任何情况下 mandatory-present。

## 2. 仍然失败的来源门禁（独立复现）

以 **14 份本轮合法基线工件**为同一 fixture，仅改变一个输入：

| 输入扰动 | 07 实测 | 正确的 hard gate |
|---|---|---|
| 预测 CSV 内容改动 | SystemExit | 已满足 |
| data manifest 内容改动 | SystemExit | 已满足 |
| 配置文件内容改动 | SystemExit | 已满足 |
| 伪造 LightGBM arm_id | SystemExit | 已满足 |
| **将 `score.json.round` 改为 `FORGED_EVALUATION`** | **PASS** | 应拒绝 |
| **将 `score.json.scale_source` 改为 `WRONG_SCALE`** | **PASS** | 应拒绝 |
| **将 score 的 overall/per_level 全改 0** | **PASS** | 应拒绝或标为未独立核验 |
| **删除 `runs/data_manifest.json`** | **PASS** | 应拒绝 |
| **删除 `configs/accuracy_v1.yaml`** | **PASS** | 应拒绝 |
| 删除 score 里的 canonical ID hash | SystemExit | 已满足 |

根因：
- `07:140–153` 只在 `if dm_path.is_file():`、`if cfg_path.is_file():` **成立时**比较哈希，文件缺失时直接跳过，破坏“缺失即中止”的宣称。
- `07` 不读/校验 score 的 `round`、`scale_source`、`config_sha256`、`data_manifest_sha256`（甚至通常没有这两个字段）。`forecast.csv` 正确不代表 `score.json` 的 WRMSSE 没被事后修改。
- `04` 输出的评分报告没有可验证的 score→预测内容 SHA/源 run_manifest SHA/实际权重来源 SHA / 源码 SHA 绑定。07 也不复算 WRMSSE，所以仅有 forecast hash 不能为 score 数值背书。
- 07 只用 `run_id in predictions_path` 字符串子串检验 score 的预测路径，不要求与被哈希的同一个 `forecast.csv` 文件严格相等。

**验收出口**：

- 对评分汇总前必需的 data manifest 和 config 文件实行 `if not is_file(): SystemExit`，而非仅当文件存在才比较。
- 定义每个 run 的 `score_contract`：`origin`、`round=custom_window`（或符合预注册的轮次）、`scale_source`、`model_label`、`canonical_id_order_sha256`、`forecast_sha256`、`data_manifest_sha256`、`config_sha256`、`source_manifest_sha256`、`reference_weights_sha256`（如使用官方参考）明确来自哪份文件。07 要比对 report ↔ manifest ↔ 实际文件。
- 将 score 文件本身的 SHA-256 纳入不可变 run 索引或签名的 evidence manifest，保证修改 score 不可能仍声称与原运行完全一致。只能检查一致性的字段不足以证明任意数值真实来自预测；**完整独立 G3 仍需用原始数据重算评分**。
- `07` 要对 `round` 和 `scale_source` 做与每个 run 预注册契约对应的硬门禁，并比较 `predictions` 的规范路径与被哈希文件本身。
- 增加上述真实 `07.main()` 正反双向测试到版本库，而不是只依赖 35 项现存单元测试。

## 3. 发现不一致的模型实验工件：新的证据清理阻塞

`m5/evidence/six_window_baseline_summary.json` 被 `4fe98a0` 更新：

| 窗口 | 汇总中 LGBM | 同一补丁仍保留的对应 evidence JSON | 汇总 run ID |
|---:|---:|---:|---|
| o1857 | **0.80743584815725** | `bt_lgbm_t1300fix_o1857_score.json` = **0.8184498373211491** | `M5A_W1_BT_lgbm_t1300fix_o1857` |
| o1913 | **0.5872038872033637** | `lgbm_t1300fix_r005_score_post_selection.json` = **0.5854685358907349** | `M5A_W1_lgbm_t1300fix_r005` |

o1857 的分数相对先前变化约 **-0.011014**，但本轮文字报告并未给出该项重新运行来源。o1913 当前汇总采用工程师报告的 **r006=0.58720** 数值，却仍将 `run_id` 标记成 **r005**；而工程师明确要求旧 r005 作为探索记录保留。

两者未必是数学错误，可能来自重新评分、行序规范化或新的预测，但**现有补丁无法将新分数与旧 run ID 关联且证明 run 不被覆盖**。这与 `never_overwrite_run` 的审计契约相冲突，必须补充新 run ID、完整预测/score manifest 和保存原始记录的 diff，绝不能无证据将旧 run 改写成新分数。

本轮附件并未实际包含宣称“14 份真实工件全部通过”的 14 组 `run_manifest.json`、`score.json`、`forecast.csv`（历史发布政策可以仅提交脱敏摘要，但不可声称审查方见到了未上传内容）。

## 4. R11 数值与研究方法学裁决

- 简单基线的六窗口观察不变：Mean28 在 six custom origins 的 overall **6/6** 优于 Naive7；L1–L5 每层 5/6，L6–L12 每层 6/6。是描述性结论，不能宣称新时期普遍优势或库存经济效用。
- LightGBM 顺序改动观察：0.58547 与 0.58720 的差约 **+0.001735**，但若同时有运行版本、训练行序或处理变化，不能未经单因素冻结消融就认定因果机制。o1913 明确 `POST_SELECTION_EVALUATION`。
- G3 官方 sNaive 评价轮 `0.8470173528345969`（工程师报告）与官方 `0.8470173528438624` 极接近，基础 scorer 的权重/零尺度/行序防线此前已通过大量合成负向测试。现在可签署 `SOURCE-LEVEL SCORER GUARDS PARTIALLY ACCEPTED`，不可签署 `G3_INDEPENDENT_PASS`：缺完整真实源数据重放，且 score 工件身份链仍可假通过。
- G0 Owner 规则接受人与数据保管人尚未签署，不因此把 G3 数据科学发现作废，但发布和治理出口仍不得宣称 CLOSED。

**状态：`R11_PATCH_SHA_PASS / 35_TESTS_PASS / VALID_14_SYNTHETIC_PASSES / FORECAST_HASH_TAMPER_REJECTED / CONFIG_DATA_TAMPER_REJECTED / SCORE_TAMPER_ACCEPTED_FAIL / MISSING_SOURCE_FILES_ACCEPTED_FAIL / RUN_ID_SCORE_CONFLICT / G3_FINAL_NO_GO / W2_DESIGN_GO`。**
