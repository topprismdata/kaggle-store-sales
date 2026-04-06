# Store Sales - Time Series Forecasting

> Kaggle 竞赛项目：使用 Agentic AI 架构完成 Store Sales 时间序列预测竞赛
> 竞赛链接：https://www.kaggle.com/competitions/store-sales-time-series-forecasting

## 项目概述

本项目使用**完全 Agentic 的竞赛方法论**参与 Kaggle Store Sales 时间序列预测竞赛。核心思路是：

1. **Agent 协作架构** — 多个专业化 Agent（特征工程、模型训练、集成学习）分工协作
2. **NotebookLM 智囊** — 上传 TOP notebooks + 论坛资料到 Google NotebookLM，作为 AI 智囊实时咨询
3. **两轮迭代** — Round 1 建立 baseline pipeline，Round 2 多模型集成优化

## 竞赛背景

- **任务**: 预测厄瓜多尔 Corporación Favorita 连锁超市 54 家门店 × 33 个商品家族在未来 16 天的销售额
- **评估指标**: RMSLE (Root Mean Squared Logarithmic Error)
- **数据规模**: 训练集 ~300 万行 (2013-01-01 至 2017-08-15)
- **预测目标**: 2017-08-16 至 2017-08-31 的 28,512 个 (store, family, date) 组合

## 项目结构

```
kaggle-store-sales/
├── README.md                    # 本文件 — 完整教学文档
├── Makefile                     # 快捷命令
├── requirements.txt             # Python 依赖
├── data/
│   └── raw/                     # Kaggle 原始数据 (gitignored)
├── src/
│   ├── config.py                # 全局配置 (路径、超参数)
│   ├── data/
│   │   └── loader.py            # 数据加载 + 多表合并
│   ├── features/
│   │   ├── builder.py           # 特征构建流水线
│   │   ├── time_features.py     # 时间/日历特征
│   │   ├── lag_features.py      # Lag/Rolling/EWM 特征
│   │   └── external_features.py # 外部数据特征 (油价/节假日/交易)
│   ├── models/
│   │   └── gbdt.py              # GBDT 模型 (LightGBM/XGBoost/CatBoost)
│   ├── ensemble/
│   │   └── blender.py           # 集成策略 (加权平均/Hill Climbing)
│   └── utils/
│       └── metrics.py           # RMSLE 评估指标
├── scripts/
│   ├── run_baseline.py          # Round 1: LightGBM baseline
│   └── run_ensemble.py          # Round 2: 多模型集成
├── outputs/
│   ├── submissions/             # 提交文件
│   └── models/                  # 模型文件
└── docs/
    └── notebooklm-sources.md    # NotebookLM 智囊搭建指南
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据
kaggle competitions download -c store-sales-time-series-forecasting
unzip store-sales-time-series-forecasting.zip -d data/raw/

# Round 1: Baseline (LightGBM)
python scripts/run_baseline.py

# Round 2: 集成 (LightGBM + XGBoost + CatBoost)
python scripts/run_ensemble.py

# 提交到 Kaggle
kaggle competitions submit -c store-sales-time-series-forecasting \
    -f outputs/submissions/submission_ensemble.csv -m "ensemble v1"
```

## 技术详解

### 1. 数据处理 (`src/data/loader.py`)

**核心挑战**: 7 个 CSV 文件需要正确合并

| 文件 | 行数 | 关键字段 | 合并方式 |
|------|------|---------|---------|
| train.csv | 3,001,088 | store_nbr, family, date, sales, onpromotion | 主表 |
| test.csv | 28,512 | 同上 | 预测目标 |
| stores.csv | 54 | store_nbr, city, state, type, cluster | left join on store_nbr |
| oil.csv | 1,219 | date, dcoilwtico | left join on date |
| holidays_events.csv | 351 | date, type, locale, locale_name | 复杂合并 (见下文) |
| transactions.csv | 83,488 | store_nbr, date, transactions | left join on (store_nbr, date) |

**节假日合并的难点**:

节假日分为三个层级，需要根据 `locale` 字段分别处理：
- `National`: 所有门店生效
- `Regional`: 只对匹配 `state` 的门店生效
- `Local`: 只对匹配 `city` 的门店生效

**油价数据处理**:
- 周末和节假日没有油价数据 → forward-fill + back-fill
- 衍生滚动均值、百分比变化等特征

### 2. 特征工程 (`src/features/`)

最终构建了 **69 个特征**，分为四大类：

#### 2.1 时间/日历特征 (22个) — `time_features.py`

```
基础日历: year, month, day, day_of_week, day_of_year, week_of_month,
          week_of_year, quarter, is_weekend, is_month_start, is_month_end
厄瓜多尔特色: payday (每月15号和月末发薪), earthquake (2016-04-16地震恢复期),
              school_start (9月开学季)
周期编码: sin/cos(day_of_week), sin/cos(month), sin/cos(day_of_year)
```

**为什么需要周期编码？**
- `day_of_week=6` 和 `day_of_week=0` 实际上是相邻的（周六和周日）
- 但数值上 6 和 0 差距最大
- 用 sin/cos 变换可以保持这种周期性关系

#### 2.2 Lag/Rolling/EWM 特征 (28个) — `lag_features.py`

```
Lag特征: sales_lag_{1,7,14,28,365}        (5个)
Rolling: sales_roll_{mean,std}_{7,14,28,60} (8个)
EWM:    sales_ewm_{7,14,28}                (3个)
促销Lag: onpromotion_lag_{1,7,14}          (3个)
促销Rolling: onpromotion_roll_mean_7       (1个)
```

**关键设计决策**:
- 所有 rolling/ewm 特征都先做 `shift(1)` 防止数据泄露
- 按 `(store_nbr, family)` 分组计算，因为每个门店-商品组合有独立的销售模式
- `rolling_windows=[7,14,28,60]` 覆盖周、双周、月、双月粒度

#### 2.3 外部数据特征 (11个) — `external_features.py`

```
油价: oil_roll_mean_{7,14,28}, oil_pct_change_{7,28}, oil_diff_7
交易: transactions_lag_1, transactions_lag_7, transactions_roll_mean_7
统计: store_family_sales_mean, store_family_sales_median, store_family_sales_std
```

#### 2.4 类别编码 (8个)

```
family → category code
city → category code
state → category code
store_type → category code
cluster → numeric
```

### 3. OOM Bug 修复记录 — 教学重点

这是本项目最有教学价值的技术问题。

#### 问题现象

运行 `run_baseline.py` 时，进程占用 7.4GB RAM，最终被系统 OOM Kill。

#### 根因分析

```python
# 有问题的代码 (lag_features.py)
shifted = group["sales"].shift(1)
for w in rolling_windows:
    for stat in rolling_stats:
        # 问题: .values 触发了完整的内存拷贝
        df[col_name] = shifted.rolling(w, min_periods=1).mean().values
```

**为什么 `.values` 导致 OOM?**

1. `group["sales"].shift(1)` 返回一个 GroupBy 操作的 lazy 对象
2. `.rolling(w)` 创建另一个 lazy 对象
3. `.mean()` 计算每个组的滚动均值
4. **`.values` 强制将所有中间结果一次性加载到内存**
5. 对于 300 万行 × 54门店 × 33品类 × 4窗口 × 4统计 = 约 540 万个中间 Series 拷贝
6. 每个拷贝 ~1KB → 总计 ~5.4GB

#### 修复方案

```python
# 修复后的代码
shifted = group["sales"].shift(1)
for w in rolling_windows:
    for stat in rolling_stats:
        # 直接赋值 pandas Series，不调用 .values
        df[col_name] = shifted.rolling(w, min_periods=1).mean()
```

**修复原理**: 去掉 `.values` 后，pandas 的索引对齐机制自动处理赋值，避免了全量内存拷贝。

#### 额外优化

```python
# config.py — 减少 rolling 统计数量
rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std"])
# 之前是 ["mean", "std", "min", "max"] — 4个统计量减到2个
```

这同时减少了内存使用和特征数量（从77特征降到69特征），而 min/max 对 RMSLE 的贡献有限。

### 4. 模型训练 (`src/models/gbdt.py`)

#### 4.1 时间序列交叉验证

传统 K-Fold CV 会造成未来数据泄露。本项目使用**滚动时间窗口 CV**：

```
Fold 1: Train [----] | Val [16天]
Fold 2: Train [------] | Val [16天]
Fold 3: Train [--------] | Val [16天]
...
每次验证集从 train_end 往前取 16 天（与竞赛预测窗口一致）
```

#### 4.2 LightGBM Baseline

```
超参数: learning_rate=0.005, num_leaves=64, min_child_samples=30
        subsample=0.8, colsample_bytree=0.8
训练: n_estimators=3000, early_stopping=100
结果: CV RMSLE = 0.3751
```

#### 4.3 XGBoost

```
超参数: learning_rate=0.005, max_depth=8, min_child_weight=30
        subsample=0.8, colsample_bytree=0.8, tree_method=hist
```

#### 4.4 CatBoost

```
超参数: learning_rate=0.005, depth=8, l2_leaf_reg=0.1
        bootstrap_type=Bernoulli, subsample=0.8
```

#### 4.5 log1p 变换

所有模型训练前对 `sales` 做 `log1p` 变换：

```python
target = np.log1p(sales)  # 训练
pred = np.expm1(pred)     # 预测后反变换
```

**为什么？**
- RMSLE = `sqrt(mean((log(1+y_true) - log(1+y_pred))^2))`
- 用 log1p 变换后，MSE 目标 ≈ RMSLE 目标
- 同时缓解了 sales 的右偏分布

### 5. 集成策略 (`src/ensemble/blender.py`)

#### 5.1 加权平均

```python
blend = w1 * pred_lgb + w2 * pred_xgb + w3 * pred_catboost
```

#### 5.2 网格搜索

对 2-3 个模型的情况，直接遍历权重空间：

```python
for w0 in np.linspace(0, 1, 51):
    for w1 in np.linspace(0, 1-w0, 51):
        w2 = 1 - w0 - w1
        score = rmsle(y_true, w0*pred0 + w1*pred1 + w2*pred2)
```

#### 5.3 Hill Climbing

从空集成开始，每次贪心地给某个模型加微小权重，如果 RMSLE 降低则保留：

```python
while not converged:
    随机选一个模型 i
    随机选步长方向 (±0.01)
    如果新权重组合的 RMSLE < 当前最优: 保留
    否则: 回退
```

这是 Kaggle 中最简单有效的权重搜索方法之一。

### 6. 后处理

```python
# 接近零的预测直接设为0
ensemble_preds[ensemble_preds < 0.1] = 0
```

某些 store-family 组合几乎无销售（如特定门店的 "LADYWEAR"），直接归零可以降低 RMSLE。

## Agentic 竞赛方法论

### 架构设计

```
                    ┌──────────────┐
                    │  Team Lead   │
                    │  (主Agent)   │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │ Feature Eng  │ │  Model   │ │  Ensemble    │
    │   Agent      │ │  Trainer │ │  Specialist  │
    └──────────────┘ └──────────┘ └──────────────┘
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                    ┌──────────────┐
                    │ NotebookLM   │
                    │  智囊 (RAG)  │
                    └──────────────┘
```

### 两轮迭代流程

**Round 1: Pipeline Baseline**
1. 数据加载 → 多表合并
2. 特征工程 → 69 特征
3. LightGBM 训练 → 5-fold CV
4. 生成提交 → Kaggle 上传

**Round 2: 多模型集成优化**
1. 添加 XGBoost + CatBoost
2. Hill Climbing / 网格搜索最优权重
3. 后处理优化
4. 集成提交

### NotebookLM 智囊使用

这是本项目最独特的方法论 — 将竞赛知识外部化到 Google NotebookLM：

1. **Phase 0**: 收集 TOP notebooks + 论坛讨论 + 获奖方案
2. **上传到 NotebookLM**: 12 个资料源，涵盖 EDA、特征工程、建模、可视化
3. **遇到问题时主动咨询**: "lag特征怎么构建？"、"油价数据怎么处理？"
4. **RAG 质量验证**: 4/4 查询通过验证

详见 `docs/notebooklm-sources.md`。

## 成绩记录

| 版本 | 模型 | CV RMSLE | 公开榜 (Public LB) | 说明 |
|------|------|---------|-------------------|------|
| v1 | LightGBM (NaN) | 0.3751 | 2.67192 | Round 1 Baseline — test lag特征NaN级联 |
| v2 | LightGBM (sales=0) | 0.3750 | 2.83107 | sales填0，lag_1全为0 → 更差 |
| v3 | **LightGBM (ffill)** | **0.3750** | **1.90248** | **ffill修复 → 大幅改善** |
| v4 | LGB+XGB+CB Ensemble | 待定 | — | Round 2 集成 |

## 技术栈

- **语言**: Python 3.x
- **GBDT**: LightGBM 4.6.0, XGBoost 3.2.0, CatBoost 1.2.10
- **数据处理**: pandas, numpy
- **AI 框架**: Claude Code (Agentic), Google NotebookLM (RAG)
- **竞赛工具**: kaggle CLI
- **版本控制**: Git

## 许可

本项目仅用于学习和教学目的。
