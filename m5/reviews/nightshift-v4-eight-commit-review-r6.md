# M5 Hands-on｜v4 八提交 R6 独立复审

**裁决：CHANGES_REQUESTED / DO_NOT_MERGE / 原 Dollar_Sales P0 CLOSED / 新 Reference-Weight P0 OPEN。**

**评审范围**：用户上传 `m5-w1-g3-review-v4.patch`；181,944 B；SHA-256 `f4c69a45e0deafc183c2b0765a23b210a3fae9566d244b756c874c06a3090874`。共 **8** 个提交：`21a31cb` → `ef54be1` → `3fce96c` → `02b2da9` → `0b383ba` → `22a3027` → `686b61d` → `846341a`。

本报告不声称评审环境获得完整 M5 训练 CSV，或重新跑了 LightGBM。所有真实数据的模型数值为 **ENGINEER_REPORTED / ARTIFACT_ARITHMETIC_CHECKED**，而非独立复训。

## 1. 本次执行与可复算结果

- 末次增量 commit `846341a` 对已应用前七提交的隔离工作区成功 `git apply --check`、`git apply`。
- `python -m compileall` PASS；`python -m pytest m5/tests -q` **22/22 PASS**。评审环境 Python 3.13.5，不是本机 Windows 3.11.15 的环境锁。
- 自行执行合成故障注入，验证 `_crosscheck_weights()` 中正确金额 `OK`、错误金额（max_rel≈0.98999）`FAIL`；`04.main()` 在 `FAIL` 时已有 `SystemExit` 硬中止。**R5 指名的 Dollar_Sales 容差错误 P0 确认关闭。**
- **新发现 P0：** 正确金额但将官方 `weight` 全部改成 `2.0` 时，`_crosscheck_weights()` 仍返回 `OK`；`_weights_from_official()` 将每层权重和 `2.0` 原样送入评分。更进一步，直接构造 12 层最小合成输入调用 **`04_score_wrmsse.main()`**，真实结果：`exit_code=0`、`status=PASS`、`reference_check_status=OK`，而结果被错误放大一倍。新增重复的聚合键也被 dict comprehension 静默覆盖。
- **新发现 P0 的另一条同源路径：** `wrmsse_from_levels()` 对 NaN 参考权重未做有限性检查，直接产生 NaN 层分数；校验元数据不足以把这个情况判为拒绝评分。上述情况不代表官方源文件存在错误，而是评测完整性门禁不健全。

## 2. R5 整改项逐项结论

| 项目 | 代码位置 | R6 结论 |
|---|---|---|
| Dollar_Sales 大差异阻断 | `04:108-116, 240-268` | **CLOSED**：大误差返回 `FAIL` 并中止 |
| 限定 G1 28 天权重窗口 | `01b:85-89` | **OPEN**：`window_weeks` 仍使用 `d_num > origin-28` 而没有 `<= origin`；随后仅对 price 记录裁至起点周，不能修复已选入的未来周 |
| 逐周价格 / 文件边界 | `01b:94-134` | **NOT ACCEPTED**：窗口错选导致报告“13周”“11614–11621为文件边界”。`file_last_week` 是**过滤后**价格表的 max，不能用于推断官方原始 `sell_prices.csv` 真正末周 |
| Zero-heavy 抽样 | `global_lgbm.py:274-279` | **OPEN**：`pd.qcut(recent,3,labels=[0,1,2],duplicates="drop")` 仍可在 70% 零销量合成样本抛 `ValueError`；所谓“退化回单一层”的修复不在本补丁里 |
| Lag 历史越界与夹具 | `global_lgbm.py:117-123`、`test_feature_alignment.py:11-12` | **CLOSED（针对原负索引缺陷）**：min_index 防护存在，fixture origin=740 |
| Confirmatory hash 一致 | `03b:78-90,173-175` | **PARTIAL PASS**：模型参数+fixed-rounds 的门禁与 manifest 使用相同 hash；但未把 origin、代码 SHA、数据 manifest SHA 纳入冻结判据，仍允许已开封 o1913 被标为 confirmatory |
| Round/origin 矛盾 | `04:66-69` | **CLOSED（针对该冲突）**：不一致 origin 中止 |
| Actual/train 身份对齐 | `04:172-181` | **PARTIAL PASS**：两个文件相互同序时通过；**未再与 `load_id_frame()` 基准顺序比对**。构造两文件同时交换行，仍通过本检查，却可能与固定汇总组及预测行顺序不一致 |
| 六窗汇总归属 | `07:63-77` | **PARTIAL PASS**：检查 origin 及预测路径包含 run_id，但未验证 model/round/scale source/data SHA；路径字符串子串并非强证据 |
| 六窗“simple-only 第一名” | `07:82-88` | **OPEN**：仍按包括 LightGBM 的整体第一名判断，记录 **5**，而 simple-only Mean28 相对 Naive7 的真实窗口数应为 **6** |
| G1 event 与销量守恒 | `01b:70-80, 145-147` | **OPEN**：group-size 总和不等于逐层销售额数值守恒；`E_event_columns_present=True` 依旧硬编码 |
| 完整 13 份 score 产物 | v4 patch 中只见两份 LightGBM `score.json` 摘要时间字段调整和一个六窗汇总变化 | **SOURCE_LIMITED**：本机 `runs/` 下的 13 份新 score 未随附件上传，不能独立断言其全部 origin/run 字段正确 |

**严重风险：** G3 硬门禁不能只验证 `Dollar_Sales`，还必须检查 `weight`。官方数据即便真实正确，评估器也不能在注入错误权重或重复参考键时仍返回 PASS。

## 3. G1 Price audit 的具体根因

现有 `window_weeks = calendar[calendar.d_num > origin-28]` 从 d1886 一直选到完整 calendar 的 d1969，而不是 d1886..d1913。随后 `prices = prices[prices.wm_yr_wk <= origin_week]` 只会令未来周在 `per_week_missing` 里全部显示缺价。当前 `content_audit.json` 的 `D_weeks_in_weight_window=13`、`D_window_weeks_upper_bound=11621`、`D_file_last_week=11613` 正是这一组合产生的**伪文件边界**，不能用于解释官方原始数据的末周或缺价机制。

与评分器本身形成直接矛盾：`bottom_weights_from_sales_value()` 已正确取 `(origin-28,origin]` 并在 o1913 得到 **11609..11613 五个周号**；但 `01b` 错误报告 **13周**。

修复要求：

```python
window_cal = calendar[
    (calendar["d_num"] > origin - 28)
    & (calendar["d_num"] <= origin)
]
window_weeks = sorted(window_cal["wm_yr_wk"].unique())
file_last_week = int(raw_prices["wm_yr_wk"].max())  # 截断前
```

缺少商品-门店-周售价不总是数据损坏：官方 guide 说明无售价可能意味着该周未售卖；需要根据同周非零实际销量判断缺价是否影响价值权重，再分类记账。不可笼统要求所有 30,490 商品-店组合每周均有售价。

## 4. 六窗口方法学裁决（当前报告范围）

仅对 Mean28 vs Naive7 的提交工件作**有条件的描述性裁决**：

| Origin | Naive7 WRMSSE | Mean28 WRMSSE | Mean28 对比 |
|---:|---:|---:|---|
| 1773 | 1.411421 | 1.204080 | 更好 |
| 1801 | 1.112062 | 1.066091 | 更好 |
| 1829 | 1.021062 | 0.851218 | 更好 |
| 1857 | 0.888939 | 0.717579 | 更好 |
| 1885 | 0.922781 | 0.658369 | 更好 |
| 1913 | 0.869701 | 0.760672 | 更好 |

- 6/6 轮 Mean28 整体低于 Naive7；六窗 Overall 平均分别 0.876335 与 1.037661；算术平均差 0.161326，较 Naive7 六窗均值低约 15.55%。
- **不是全部层级 6/6 胜出**：o1801 仅 7/12 层 Mean28 胜；L1–L5 在六窗中各胜 5/6，L6–L12 各胜 6/6。
- 六窗预测区间互不重叠，但训练历史重叠、需求机制相关、权重/尺度每窗重算；不能把 6/6 当成六个独立自然实验或总体胜出概率。没有完整原始 CSV 独立回算，也没有基于依赖结构的置信区间。
- **研究结论**：`MEAN28 > NAIVE7, OBSERVED_IN_SIX_CUSTOM_WINDOWS`；`GENERALIZATION_NOT_CONFIRMED`。可以将 Mean28 作为强简单基线继续使用；不批准普遍胜出或经济效用声明。
- LightGBM 修复版 o1857=0.81845 仍差于 Mean28=0.71758；o1913=0.58547 优于 Mean28=0.76067，但此窗口已开封，只能是 `POST_SELECTION_EVALUATION`；本轮不对 LightGBM 做稳定性裁决。

## 5. R6 退出条件（先堵误报，再 W2）

**P0：官方 reference weight integrity**

1. 正确记录 ref file hash/round；拒绝重复或缺失的 42,840 reference key。
2. 全部 `Dollar_Sales`、`weight` 为有限非负；每层的 weight sum≈1。
3. 每键依据参考 Dollar_Sales 重算 `round(12 * Dollar_Sales / sum(all_12_level_dollars),12)`，在预先冻结的精度容差内匹配 `weight`。
4. 用 3 类负向测试保证 **0.99 的参考金额偏差、2×权重但正确金额、重复 reference key** 全部在 **`04.main()` 入口级非零失败**；不能只测试 helper。

**P1：** 完整修复 `01b` 的时间上界、价格真文件边界与事件/销量守恒；`qcut` 可退化样本；baseline scorer 的三方 ID 对齐；`07` 的 simple-only 排名、模型/轮次/source/hash；确认性冻结应含 origin/数据/源码并封锁曾开封的 o1913。

**治理：** Owner 仍待登记 G0 责任人；只能在新的未开封样本中对 W2 候选方法做严格确认性评价。对本次所有结果保留原 run、异常、哈希与失败记录；不修改结果以达门槛。

**本轮状态：** `PATCH_SHA_AND_8_COMMITS_PASS / 22_OF_22_TESTS_PASS / DOLLAR_MISMATCH_P0_CLOSED / NEW_REFERENCE_WEIGHT_P0_OPEN / PRICE_AUDIT_AND_QCUT_P1_OPEN / SIX_WINDOWS_DESCRIPTIVE_ACCEPTED / DO_NOT_MERGE`。

**材料说明：** 仅提交原创审计报告到 review 分支；v4 完整补丁包含尚未关闭的阻塞项和本地路径，不能冒充已审核通过的代码直接合并。
