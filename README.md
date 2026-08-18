# Kaggle Store Sales --- Capability Formation Case

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

## Preserve failure history

Do not rewrite the project as if the first approach was correct.

The repository records large CV ↔ public-LB mismatch and iterative
fixes. That history is valuable because Native AI needs to learn from
failure, not only final scores.

Recommended section:

### What the agent learned later

For every major mistake:

``` text
Observation
→ root cause
→ corrected method
→ reusable skill / principle
→ later project where it was reused
```

------------------------------------------------------------------------

## Evidence

Keep competition scores, but classify them as **learning evidence**, not
product evidence.

Move long feature-by-feature tutorials and code-debug narratives into
`docs/`.

README should contain:

-   task;
-   best reproducible result;
-   most important failures;
-   skills crystallized;
-   relationship to `cultivating-ml-agent`.

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
