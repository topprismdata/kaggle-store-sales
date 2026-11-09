# M5 W1 | R15 双轨独立验收（v12 / e954202）

**裁决：A_ENGINEERING_PARTIAL_ACCEPT_FINAL_NO_GO / B_OFFICIAL_INDEPENDENT_REPLAY_NOT_EXECUTED / G3_INDEPENDENT_PASS_NO_GO**

## 1. 补丁与测试

- 本轮上传 `m5-w1-r14-fixes.patch`：**99,139 B**，实测 SHA256 `57c49df22137c76d1372639c90079b295359730521e445bfaf59b74a0188cbd0`，与报告一致。
- 实际是 **7 个**邮件提交（`6de0b39,4fe98a0,648ef8e,b0689cb,8784b90,ad3f5a8,e954202`）；相对 R14 原样工作区只有 `e954202` 新增。仅应用第 7 个，`git apply --check` / `git apply` / compileall 全部 PASS。
- `pytest m5/tests -q` **35/35 PASS**；本次新增提交只修改 `07_six_window_summary.py`，没有新增持久化负向测试文件。
- 复跑生产 `07.main()`，14-run 合成评分/预测/manifest、monkeypatch 源数据加载（真实 M5 大数据未提供）；R14原21项 **17/21** 符合预期，拓展到26项后 **18/26** 符合预期。测试不是原始 Kaggle 数据复算。

## 2. R14 八条错放行中本轮已修复的 4 条

1. 单层 claimed NaN/Inf 不再绕过 abs 比较。
2. 合法枚举但错报 official route、而缺权重 SHA 时拒绝。
3. manifest config SHA 错误非空值被真实文件哈希否决。
4. manifest data SHA 错误非空值被真实文件哈希否决。

这些具体修复均通过 `07.main()` 的负向故障注入。

## 3. 仍存在的 4 条 R14 假通过

| 反例 | R15 实际结果 | 根因 |
|---|---|---|
| 删除 manifest `forecast_sha256` | **PASS（错误）** | `if man.get('forecast_sha256') and ...` 不要求字段必须存在 |
| 14 个 run 共同伪造 score/manifest canonical ID hash | **PASS（错误）** | 只比较工件相互一致，没有与实际 `id_order_sha256(load_id_frame(data_dir))` 的第三锚点对照 |
| 伪造 score 的 `actual_file_sha256` | **PASS（错误）** | 未与实际窗口标签文件真实字节哈希比较 |
| score 的预测路径换成另一个同run文件 | **PASS（错误）** | 用 `run_id in predictions` 子串检查，不是规范路径精确相等 |

## 4. 额外 5 项组合输入：4 项错误 PASS

| 反例 | 07 结果 |
|---|---|
| 删除 manifest `config_sha256` | **PASS（错误）** |
| 删除 manifest `data_manifest_sha256` | **PASS（错误）** |
| 将 score `scale_source=SCALE_SOURCE_OFFICIAL_PARITY` 并填写虚构非空 `weights_file_sha256` | **PASS（错误）** |
| 将 score `scale_source=SCALE_SOURCE_OFFICIAL_WEIGHTS_SELF_SCALES` 并填写虚构非空 `weights_file_sha256` | **PASS（错误）** |
| `SELF_COMPUTED` 但附带虚构权重 SHA | 正确拒绝 |

**正确整改**：对必需 manifest SHA 做存在性及内容完整验证，score / manifest / 实际输入 canonical ID 三方绑定；核验 actual、train、forecast 的真实 SHA 和精确路径；official route 必须读取实际受控权重文件验证 SHA，不是仅要求一个非空字符串。所有反例纳入真实入口 CI。

## 5. 两轨签署

**A 轨工程实现：PARTIAL / FINAL NO-GO。** 多轮修复的预测、scorer 及强基线局部能力可继续使用和研究，但由于 8 个 provenance false-PASS，不能发布为可独立验收的完整链路。后续补丁只需聚焦门禁，无需重做研究。

**B 轨官方独立复算：NOT_EXECUTED。** 本独立审查环境未获得 M5 原始完整销售/价格/日历、双方完整预测工件、权威 reference 全量资产与第二独立实现；不能声称已经从真实 42,840 序列计算 validation/evaluation 两轮。应由**不同于工程师的独立执行人**在经批准的数据环境中冻结 source/code/config/env/reference SHA，再对 validation origin 1913(d1914..1941)、evaluation origin1941(d1942..1969) 执行逐键42,840、逐层12及overall复核，连同所有异常、负向注入与签署输出。

**冻结文件**：该轮对话交付 `M5_W1_R15_AB_Gates_and_Independent_Replay_Freeze_Spec.md`（详细目录、角色、冻结对象、官方精度门槛与逐步规程）及 `M5_W1_G3_freeze_manifest_template.json`（未封存模板）。实际模型/数据不随本报告上传。

## 6. 方法学与放行

Mean28在提交的六个指定历史回测窗口 overall **6/6** 优于 Naive7，平均WRMSSE约 `0.876334749` vs `1.037660930`，差约15.55%；L1-L5每层5/6、L6-L12每层6/6。**限定性描述性归档**，非独立实测新时期成功概率或库存收益。

`W2_DESIGN_GO / W2_EXPLORATORY_GO / W2_CONFIRMATORY_NO_GO`：真正新封存样本、预注册模型/基线/分层/统计检验及G0数据责任签署未完备。

**R15：`PATCH_INTEGRITY_PASS / 7_EMAIL_COMMITS_1_NEW / 35_TESTS_PASS / 18_OF_26_EXPECTED / 8_FALSE_PASSES / A_PARTIAL_NO_GO / B_NOT_EXECUTED / G3_FINAL_NO_GO`。**
