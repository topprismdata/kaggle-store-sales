# M5 Hands-on｜v3 七提交补丁独立复审 R5

**裁决：CHANGES_REQUESTED / DO_NOT_MERGE**  
**核心变化：原零尺度掩蔽 P0 已关闭；Dollar_Sales 交叉核验“错误仍 PASS”的另一项 P0 未关闭。**  
**资料状态：** 官方 sNaive 数值与证据 JSON 一致性强，但本评审没有获得 450 MB 完整 M5 训练数据并重新跑完整模型。

## 1. 输入完整性与提交链

- 附件：`m5-w1-g3-review-v3.patch`
- 实测字节：**156,379**
- 实测 SHA-256：`23a9a93ec0c01e3c67b34c87b31956cf9e3fe9bd07a333e32d0721bb58708ca0`
- 七提交：`21a31cb` → `ef54be1` → `3fce96c` → `02b2da9` → `0b383ba` → `22a3027` → `686b61d`
- 将补丁拆分为 7 份并逐一 `git apply --check`、`git apply`；全部通过。前六提交与 v2 版代码 diff 一致（个别邮件尾部空白不同）。
- 在隔离工作区结合规范分支的基础 Python 模块与测试契约，`python -m compileall` 通过，`pytest` **22/22 PASS**。评审运行环境为 Python 3.13.5，不是用户 Windows 3.11.15；不冒称复现 Windows 运行环境。
- 附加独立合成反例（评分 main()、错误金额、零销量分层抽样、冻结配置哈希），不包含在 22 项测试中。

## 2. 主要变更 686b61d：原评分门禁 P0 关闭

**确认修复**：

- `scoring.py:203–223` 增加 `assert_scoring_ready(scales, weights)`，在计算前检查 `(~isfinite(scale) or scale <= 0) & (weight > 0)` 并报告对应行。
- `04_score_wrmsse.py:114–126` 在向任何无效行填 0 **之前**调用门禁，仅对通过门禁的零权重无尺度行填零。
- `wrmsse_from_levels()` 继续对正权重无效 RMSSE 做二次检查。
- 合成输入中正权重尺度 `NaN`、0、-1、`inf` 全部抛 `ValueError`；零权重无效尺度按声明策略允许，返回异常计数。
- **入口级独立复现**：模拟 12 层基础模型输入，直接调用 `04_score_wrmsse.main()`，设置 Level1 `weight=1.0, scale=NaN`，现在中止并报错；旧版会产生错误的 `status=PASS`。
- `test_scoring.py` 中新增两项双向测试。本次 22/22 为原工程测试全部通过，不表示不存在没有覆盖的路径。

**裁决：`P0_ZERO_SCALE_MASKING = CLOSED`。**

## 3. 仍未关闭的评分 P0：Dollar_Sales 交叉核验错误也返回 OK

`04_score_wrmsse.py:208–239` 的 `_crosscheck_weights()` 逐层计算 `dollar_sales_max_rel_diff`，但没有误差阈值或失败分支；始终 `return {"status": "OK", ...}`。`main()` 随后固定生成 `"status":"PASS"`。

**独立故障注入**：构造 12 层各一条的 synthetic key，计算 Dollar_Sales=10，故意把“官方参考”设置为 999；程序实得 `dollar_sales_max_rel_diff = 0.98998998998999`、`status=OK`。因此实际参考错约 **99% 也能通过**。

**最低整改**：对官方 reference 与重算金额逐键校验时，以冻结阈值硬中止；同时验证文件的 42,840 唯一聚合键、验证/评价轮独立文件 SHA、权重非负有限与每层归一化。增加 1% 和 99% 错误注入的 CLI 入口级负向测试。若官方文件不含 `Dollar_Sales`，不得默认以 `SKIPPED` 仍对外宣称完整校验通过。

**裁决：`P0_REFERENCE_MISMATCH_FALSE_PASS = OPEN`。**

## 4. 尚未关闭的 P1 与研究治理项

| 位置 | 现状/复验 | 验收要求 |
|---|---|---|
| `01b_content_audit.py:85` | 权重窗口仍使用 `d_num > origin-28` 而无 `<= origin`；误选未来日，报告 13 周并非真正最后 28 天 | 封闭上下界、核验所有所需 (item,store,week)；事件/SNAP 字段真实检测 |
| `01b:70–80` | group size 总数守恒不等于销量数值守恒 | 12 层独立销售聚合加总核验 |
| `global_lgbm.py:269–279` | 零销量高占比时 `pd.qcut(..., labels=[0,1,2], duplicates="drop")` 仍可崩；独立 90 条高零样本触发 `ValueError: Bin labels must be one fewer than the number of bin edges` | 可退化的活跃度分层与正负向用例 |
| `test_feature_alignment.py:11–12,70–71` | fixture 首个目标 t=15，却读取 `t-364=-349`；NumPy 回绕为未来列 51，测试同样读取负下标形成自证正确 | 将起点提高至 >= 730 并另测历史不足时拒绝负 lag |
| `03b_global_lgbm.py:79–87,169–170` | confirmatory 期望哈希包括 `fixed_rounds:1300`，manifest 的 `config_hash` 不含轮数；独立计算二者分别为 `50307cf63c... ` 和 `ece84bd5c3...`，明确不一致 | 统一规范配置对象/hash，冻结数据与代码 SHA、来源、origin；d1913 不得再称 untouched-confirmatory |
| `04_score_wrmsse.py:66–83` | `--round evaluation --origin 1913` 自相矛盾组合仍可进入评分 | 拒绝 round-origin 交叉冲突 |
| `04:162–169,172–205` | actuals 按 CSV 行位置读取；官方权重 lookup 未主动拒绝重复键/多余键 | 三方 ID、序列顺序、42,840 完整权重键校验 |
| `07_six_window_summary.py:54–78` | 汇总不验证 score 内 origin/model/reference SHA；“mean28_first_of_simple” 实际用 overall 包括 LGBM 的排名 | 每个 run 逐项核验身份和准确命名指标 |
| `m5/evidence/*.json` | 存在本机 E: 绝对路径 | 公共提交前脱敏，仅保留 hash / 相对路径 / 轮次 |

上述 P1 不得通过提高模型分数或增加测试总数抵消。

## 5. 02b2da9 与后续：数值证据复核

- 官方评价轮 sNaive：`0.8470173528438624`（此前已从官方表独立核对）。
- `g3_snaive_reproduction_full_precision.json`：重算 12 层 mean `0.8470173528345969`，最大逐层差 `1.4909784518124525e-10`，内部计算一致；实际 42,840 序列的逐键重跑本轮尚未进行。
- `04` 的 `--round validation|evaluation` 轮次路径与完整精度导出功能存在；与 `--origin` 冲突未封锁、actual ID 对齐仍是开放缺陷。
- 六窗口自定义回放：Mean28 6/6 窗低于 Naive7。所有 12 层平均与提交报告算术一致；尚不是六个独立样本或官方竞赛六次评分。
- 修复后 `lgbm_tweedie1300`：o1857 `0.8184498373211491`，仍差于 Mean28 `0.7175785836`；o1913 `0.5854685358907349`，优于 Mean28 `0.7606716964`，但属于 `POST_SELECTION_EVALUATION`；该 run 的全量数据未由评审环境重训。
- 不能将 0.93579→0.81845 或 0.91638→0.58547 完全归因于 calendar 修复，因为 P0 批次同时修改价格可见性及固定轮数 365 日 refit；需单独消融 C/P/R。

## 6. 剩余验收门与后续指令

1. **先修 P0**：`_crosscheck_weights` 大误差必须真正抛错/非零退出；官方 reference 缺失、键重复和 round 错配不能静默产生 PASS；补充错误注入回归。
2. 同步修上述 P1：价格日期窗口、抽样 qcut、特征测试负索引、冻结哈希一致、scorer 轮次/ID 校验、六窗口 run 身份检查。
3. 仅将审计通过的、脱敏后的原创代码及结果摘要进入 PR，保留本地完整 run、旧失败和日常日志；不要重新分发官方权重/评分 R 原件。
4. d1913 仅保留后选择描述性证据；W2 未封存新的确认性样本前，不得以其作为最终增量证据。
5. G0 规则接受人和数据保管人仍待 Owner 签署。G3 应记 `LOCAL_NUMERIC_PARITY_STRONGLY_SUPPORTED / SCORER_DEFENSIVE_GATES_FAIL`，不能升级 `G3_INDEPENDENT_PASS`。
6. 这次七提交完整补丁用于隔离审计，**不把整个存在 P0 的实现直接合并**到 `research/m5-study-v1`。

**最终裁决：** `HASH_AND_7_COMMITS_VERIFIED / 22_TESTS_PASS / ZERO_SCALE_P0_CLOSED / DOLLAR_REFERENCE_P0_OPEN / CHANGES_REQUESTED`。
