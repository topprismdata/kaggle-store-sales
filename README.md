<p align="center">
  <img src="https://raw.githubusercontent.com/topprismdata/.github/main/assets/brand/topprism-repo-header.png" alt="TopPrism dual-prism visual" width="100%" />
</p>

# Kaggle Store Sales --- Capability Formation Case

> **Language / 语言:** English primary · 中文概览如下。
>
> ### 中文概览
> 使用公开时间序列竞赛作为可度量训练环境的纵向学习项目，沉淀预测实验和可复用机器学习知识。


**A longitudinal learning project for TopPrism's Cultivating ML Agent,
using a public time-series competition as a measurable training
environment.**

`LEARNING PROJECT` · `TIME SERIES` · `KAGGLE BENCHMARK`

> This is **not a customer product**. Its value is the project
> experience, failure analysis, and reusable ML knowledge crystallized
> for later work.

------------------------------------------------------------------------

## Why this project matters

The public competition provides:

-   a fixed dataset;
-   a measurable metric;
-   temporal validation challenges;
-   feature engineering;
-   model comparison;
-   operational debugging.

That makes it useful as an environment where an ML agent can learn and
leave behind reusable skills.

------------------------------------------------------------------------

## Competition task

Forecast 16 days of sales for 54 stores × 33 product families using the
public Favorita Store Sales dataset.

------------------------------------------------------------------------

## Most important learning

The repository should emphasize **what became reusable knowledge**, not
every implementation detail.

Examples already visible in the project:

-   lag / rolling features must be shifted to avoid leakage;
-   temporal validation should use rolling / walk-forward logic;
-   memory behavior in large pandas feature pipelines matters;
-   public-LB failure can reveal train/test feature-generation mismatch;
-   external knowledge sources are useful only when validated against
    the actual pipeline.

------------------------------------------------------------------------

## Preserved failure history

The project records a large CV vs public-LB mismatch and the iterative
fixes that followed. That history is preserved verbatim in
[`docs/competition-log.md`](docs/competition-log.md) because Native AI
needs to learn from failure, not only from final scores.

The reusable pattern in this project is:

``` text
Project
-> failure
-> root cause
-> corrected method
-> reusable skill / principle
-> later reuse
```

The v1 -> v3 (LightGBM ffill fix) and v3 -> v4 (Round 2 multi-model
ensemble) jumps in this project both follow this pattern.

------------------------------------------------------------------------

## Evidence

Competition scores are classified as **learning evidence**, not as
product evidence. Long feature-by-feature tutorials and code-debug
narratives live in [`docs/competition-log.md`](docs/competition-log.md)
and are not duplicated in the README.

This README records:

-   task: forecast 16 days of sales for 54 stores x 33 product families;
-   best reproducible result: LightGBM with ffill fix (public LB
    ~1.90248 per the preserved log);
-   most important failures: v1 -> v2 NaN-vs-zero lag-fill regression;
-   skills crystallized: time-series walk-forward validation,
    feature-leakage detection, AutoML-first tabular baseline;
-   relationship to `cultivating-ml-agent`: this is one of the
    longitudinal capability-formation cases the agent ran through.

------------------------------------------------------------------------

## TopPrism metadata

``` yaml
topprism:
  purpose: learning-project
  capability: time-series-ml
  maturity: learning
  evidence:
    type: kaggle-benchmark
  parent:
    - cultivating-ml-agent
```
