# M5 W1｜R12 v10 六提交补丁独立复审

**结论：CHANGES_REQUESTED / G3_INDEPENDENT_SIGNOFF_NO-GO / W2_DESIGN_GO**

## 输入与方法
- 对话附件 `m5-w1-r11-fixes.patch`：**118,593 B**；SHA256 **`f1942ecdf1753fa036189e86f6ee7f2114ac82fbb5306ead365f456decd2a0bb`**；6 commits：`6ecdead` → `45d5252` → `dbaddad` → `6de0b39` → `4fe98a0` → `648ef8e`。
- 以已包含前5提交的 R11 独立快照为基线，只应用 `648ef8e`：`git apply --check` / `git apply` PASS；compileall PASS；pytest **35/35 PASS**（Python 3.13 隔离环境，不是用户 Windows 锁环境）。
- 用 14 份模拟 run 的 score/manifest/forecast，直接执行 `07_six_window_summary.main()` 做15种输入扰动；真实 M5 源数据及完整14套真实预测文件不在本次附件中，**未声称独立重新训练或全量评分**。

## R11 修复关闭项
1. **缺文件假通过 CLOSED**：删除 `runs/data_manifest.json` 或 `configs/accuracy_v1.yaml` 时07现在 SystemExit。
2. **已有文件篡改可拒绝**：改 forecast/data manifest/config 内容或 LightGBM arm，07正确拒绝。
3. **r005/r006 run ID 混淆 CLOSED（仅汇总映射）**：07将 o1913 LightGBM 指向 `M5A_W1_lgbm_t1300fix_r006`，补丁新增 `lgbm_t1300fix_r006_score_post_selection.json`，0.5872038872033637 与汇总一致；旧 r005 0.5854685358907349 仍保留。o1913始终是 `POST_SELECTION_EVALUATION`。

## 15种独立故障注入实际结果
- 正确通过：合法14-run汇总、预测CSV被改→拒绝、config被改→拒绝、data manifest被改→拒绝、config被删除→拒绝、data manifest被删除→拒绝、错误arm→拒绝、score缺actual hash→拒绝、score缺data hash→拒绝；**9种行为符合预期**。
- **6种假通过（均07输出 PASS）**：`score.round` 被改为 `FORGED_EVALUATION`；`score.scale_source` 为 `FORGED_SCALE_SOURCE`；score overall与全部12层数值改成0；score的config hash变为64个0；score的train hash变为64个0；score的predictions改为同run目录下另一个不存在文件的路径。
- 根因：07只检查若干字段是否非空、forecast实际SHA、manifest data/config SHA；不把 `round/scale_source` 与六窗契约核对，不比较 score config/train hash 与实际可信来源，不把 score 的预测文件路径等同于被哈希的 forecast 文件。没有独立冻结的score hash或由07重新评分，所以任意改写 WRMSSE 数值仍能生成PASS。
- **结论：R11的缺失文件门禁已修，但评分来源链尚非不可伪造、不可变，`G3_INDEPENDENT_PASS` 仍不允许。**

## 新证据冲突尚未解决
- v10汇总 **o1857 LightGBM=0.80743584815725**，但同一补丁中的旧独立 evidence `bt_lgbm_t1300fix_o1857_score.json` 仍 **0.8184498373211491**；两者同指 `M5A_W1_BT_lgbm_t1300fix_o1857`。差 **0.011013989164**。须提供新评分、run manifest/预测哈希并改用新 run ID，或明确正当的重新评分迁移。
- o1913已明确 r006，v10 evidence分数与汇总相符，但仅有摘要，缺实际 forecast+manifest 的评审端核验。

## 六窗口正式方法学裁决（限定性）
- v10 summary 的12层均值与overall全部在浮点精度内一致。
- 六个指定历史回测窗口 Mean28 **6/6** overall优于 Naive7；六窗算术平均WRMSSE分别为 **0.8763347491636488 vs 1.0376609295501775**；平均差 **0.16132618038652868**，相对降低 **15.5471%**。
- Level1–5每层5/6胜；Level6–12每层6/6胜；o1801仅7/12层胜。
- 结论 `OBSERVED_IN_SIX_CUSTOM_WINDOWS / SUMMARY_ARITHMETIC_VERIFIED / GENERALIZATION_NOT_CONFIRMED`。六窗相邻、训练历史重叠、权重/尺度分窗重算；不是六个独立自然实验，不证明新时期泛化或库存经济价值。

## 退出条件
1. 07 对 `round=custom_window`、`scale_source=SCALE_SOURCE_SELF_COMPUTED`、各模型身份和run manifest的预注册契约做硬断言；对score config/data/forecast SHA与真实文件分别比较，并明确train/actual如不可取得则 `NOT_RECHECKED`。
2. 预测路径做**完整标准化路径**比对，不再仅检查 `run_id in predictions`。
3. 保存受保护的score SHA索引、使评分工件不可变，或由独立 verifier 用真实预测+原始标签/权重 **重新计算 WRMSSE**；单纯修改 score 数值时不得返回来源验收PASS。
4. 澄清 o1857 分数变化，为新分数绑定唯一 run ID；提供14份运行来源的可审查脱敏manifest与摘要。
5. G3最高等级独立验收必须在实际完整 M5 真实数据上复算两轮的12层/42,840键和所有输入SHA；本轮没有该条件。G0 owner责任签署也未完成。

**R12 STATUS：`PATCH_INTEGRITY_PASS / 35_TESTS_PASS / MISSING_SOURCE_REJECTION_CLOSED / R006_SUMMARY_ID_CLOSED / SCORE_ROUND_SCALE_VALUE_TAMPER_FALSE_PASS / O1857_UNRECONCILED / G3_INDEPENDENT_NO-GO / W2_DESIGN_GO`。**

**评审附带的独立复现材料（本轮生成于模型隔离环境）：** `m5_r12_independent_checks.py`、`m5_r12_independent_results.json`、`m5_r12_review_report.md`。原始 M5 CSV、第三方官方权重和原始预测文件不上传公开 GitHub。
