# M5 W1 G3 / Global LightGBM — Independent Code Review R1

**Decision: CHANGES_REQUESTED — DO_NOT_MERGE**

**Review date:** 2026-11-09  
**Base:** `research/m5-study-v1`  
**Inspected artifact:** chat-uploaded `m5-w1-g3-review.patch` (SHA-256 `fee523e29e4f28362740c5ebc63d357d8c0619ac59dc623cb3228b60aba489b5`), containing **only** local commit `21a31cb` (20 changed files). The two uploaded files are byte-identical.  
**Superseding bundle:** `https://dpaste.org/7aWhY/raw` was provided, reportedly adds `3fce96c`, but it could not be fetched in the review environment. Thus **the six-window summary script from the second commit has not been inspected**. This branch is a review record, **not a mirror of the latest engineer patch**.

## 0. Evidence taxonomy

- `SOURCE_VERIFIED`: companion official M5 results and 12-level sNaive evaluation-round score independently checked.
- `PATCH_INSPECTED`: source of local commit `21a31cb` read and inspected.
- `SYNTHETIC_REPRODUCED`: calendar misalignment verified on an original two-series/365-day synthetic fixture in the review runtime.
- `ENGINEER_REPORTED`: full M5 scores, six-window numeric data, weight parity, elapsed time and memory estimate. No raw Kaggle dataset was available to this review process.
- `NOT_INSPECTED`: local commit `3fce96c`, its six-window scripts/output, and complete high-precision G3 score output.

## 1. P0 — training calendar/event/SNAP are paired with wrong labels

**File:** `m5/src/m5research/models/global_lgbm.py`, `build_feature_table`.

Training labels and lag/rolling/static features are **series-major** (`sub_sales[:, targets].reshape(-1)`), while calendar values use `np.repeat(day_df[...].to_numpy(), n_series, axis=0)`, **day-major**. The same mismatch affects the event flag and state SNAP arrays.

**Independent synthetic check:** For two series whose correct first target weekdays are `[2,3,4,5,6]`, the first five training rows have `[2,2,3,3,4]`. Labels follow consecutive target days, so training samples are misaligned. Any resulting LightGBM predictions are contaminated by a deterministic feature-construction defect.

**Fix:** Either build all features in series-major order via `np.tile(day_df[...].to_numpy(), (n_series, 1))` and `np.tile(event_flag, n_series)` with correctly keyed SNAP, or build every array day-major consistently. Add an explicit two-series × seven-day alignment regression test for weekdays, event, SNAP, price and label.

**Gate:** Reject the W1 LightGBM model interpretation until this is fixed and all affected runs rerun. Existing scores remain historical debugging artifacts, not confirmatory evidence.

## 2. P0 — future prices and future fallback price leak into main experiment

**File:** `m5/src/m5research/models/global_lgbm.py`, `recursive_forecast`.

The function reads target-window `price_map` from the whole `sell_prices.csv` and builds `last_known_price` with `prices.sort_values(...).groupby(...).tail(1)`, without restricting to `wm_yr_wk` available at the forecast origin.

The frozen research contract requires the **main arm to exclude unknown future price information**. In the public M5 competition future price fields may be provided as covariates, but that constitutes a different, explicitly named arm; it cannot be silently conflated with a prospective decision-availability-safe main experiment.

**Fix:** Introduce an explicit `future_price_policy` enum. Main arm = no target-window future price; optional arm = competition-provided future prices, clearly labeled, with matching train/forecast feature logic. Never use a fallback derived from weeks after the origin. Add an invariance test: modifying future-week prices must not alter the main-arm predictions.

## 3. P0 — fixed 1,300-round mode still trains on only 337 target days

**File:** `m5/scripts/03b_global_lgbm.py`.

`train_rows` excludes the last 28 target days unconditionally. In `--fixed-rounds` mode early stopping is disabled but `dtrain` is still built from the truncated 337-day slice. This violates the frozen 365-day training target contract and confounds design comparison.

**Fix:** Use an internal pre-origin validation holdout for **design selection only**; once the round count is frozen, refit the model on all 365 eligible target days for the report run. Record `train_target_days_per_series`, `validation_days`, and `refit=true/false` in the manifest. Separate `phase=design` from `phase=report` in CLI, restrict allowable origins/objectives by phase and hash the frozen design.

## 4. P0 — official-reference scoring route still stub; undefined scales may silently disappear

**Files:** `m5/scripts/04_score_wrmsse.py`, `m5/src/m5research/scoring.py`.

`_load_reference()` always exits, and `_crosscheck()` is `PENDING_REFERENCE_LAYOUT`. Even after obtaining official weights, the CLI has no implementation that can produce `SCALE_SOURCE_OFFICIAL_PARITY`.

More importantly, `wrmsse_from_levels()` calls `np.nansum(...)`. If a **positive-weight** series has invalid scale, its contribution can silently vanish. This contradicts the strict undefined-scale gate in the existing `metrics/wrmsse.py`.

**Fix:** Implement keyed official-reference comparison or rename the flag until implemented. Before any weighted sum, explicitly assert all positive-weight series have finite positive scales and finite RMSSE; otherwise fail with a per-series diagnostic. Tests must include a positive-weight zero-scale case and a zero-weight undefined-scale case with an explicitly documented policy.

## 5. P0 — evaluation-round `origin=1941` cannot be reproduced by the submitted scorer as written

**Files:** `m5/src/m5research/scoring.py` `load_sales_matrix()` and `m5/scripts/04_score_wrmsse.py` `_load_actuals()`.

Both use the `sales_train_validation.csv` / `sales_train_evaluation.csv` scheme for the public validation window. `load_sales_matrix(origin=1941)` tries to fetch d_1914–d_1941 from `sales_train_validation.csv`, where they do not exist. Separately the evaluation labels d_1942–d_1969 are not columns in `sales_train_evaluation.csv`; they come from official post-competition test/actual data.

**Fix:** Add `round=validation|evaluation` with explicit train/actual/reference file contracts, or use a validated combined actuals matrix with strict key alignment. Ensure the evaluation scorer's inputs and manifest show exactly which released evaluation targets were used.

**Evidence discrepancy:** The submitted `m5/evidence/g3_snaive_reproduction.json` rounds each `reproduced` level to six decimals and reports `max_abs_diff=4.881938939771402e-07`, whereas nightshift text claims `1.5e-10`. The latter cannot be established from this patch. Preserve unrounded 12-level output and run SHA before claiming strict parity.

## 6. P1 — G1 price audit selects future weeks and checks only ANY-week coverage

**File:** `m5/scripts/01b_content_audit.py`.

`window_weeks = calendar[calendar["d_num"] > args.sample_days - 28]` has **no upper bound `<= sample_days`**; with sample_days=1913 this includes up to d_1969. The reported “13 weeks” follows from this overbroad selection and is not the 28-day WRMSSE weight window.

Also `pairs - window_pairs` only verifies that a pair has a price in at least one selected week, not in every week/day where it is needed. `E_event_columns_present` is hardcoded true. The `C_partition_ok` group-size check does not establish numeric aggregation conservation or exact official key mapping.

**Fix:** Freeze last-28-training-day calendar bounds; cross join actual required `(store,item,week)` keys and count missing; verify event/SNAP columns rather than hardcoding status; independently test numeric sales preservation at all levels.

## 7. P1 — design selection/confirmation segregation is comments, not enforceable workflow

**Files:** `m5/scripts/03b_global_lgbm.py`, `m5/src/m5research/models/global_lgbm.py`.

`OBJECTIVE_OPTIONS` gives arbitrary user choice at any origin, with unrestricted `--fixed-rounds`. No `phase` gate, no designated frozen spec hash, no prohibition on using report-window labels for later changes. The engineer admits one exploratory peek at d1913; retaining that log is correct, but it must prevent this same interval from being treated as **untouched confirmatory evidence**.

**Fix:** Explicit `EXPLORATORY_CONTAMINATED` label for the prematurely inspected d1913 run; mark any subsequent d1913 result as **post-selection evaluation** unless an untouched later holdout is reserved. Use a genuinely untouched origin (or later live period) for strict confirmation; document how much information from d1913 was available during method selection.

## 8. Six-window baseline summary — provisionally useful, not independent proof

Engineer-reported overall WRMSSE, per-window self-computed weights/scales:

| Origin | Naive7 | Mean28 | Tweedie-LGBM (reported) |
|---|---:|---:|---:|
| 1773 | 1.41142 | 1.20408 | — |
| 1801 | 1.11206 | 1.06609 | — |
| 1829 | 1.02106 | 0.85122 | — |
| 1857 | 0.88894 | 0.71758 | 0.93579 |
| 1885 | 0.92278 | 0.65837 | — |
| 1913 | 0.86970 | 0.76067 | 0.91638 |

Mean28 < Naive7 in all six windows. This supports **within-six-window rank stability**, not a universal win probability or cross-category superiority. The new six-window aggregation/weight code in superseding commit `3fce96c` has not been accessed here; thus the mean scores remain `ENGINEER_REPORTED`.

The LightGBM cells should not be used to make architectural conclusions until P0 feature, refit, and price issues are fixed and the impacted runs rerun.

## 9. What may be merged when defects are fixed

**Potentially acceptable after review:** standalone data fingerprint/manifest schemas, corrected content-audit rules, safe baseline generator, immutable run-ID contract, non-redistributed provenance summaries, official weights crosscheck.

**Block until repaired:** Global LightGBM features/training/recursive inference, scorers with NaN suppression or incorrect evaluation round, and any status promoting provisional scores to `OFFICIAL_PARITY` automatically.

**Requested handoff:** newest two-commit patch (`21a31cb` + `3fce96c`) as a conversation attachment or accessible GitHub branch; raw source/test + sanitized summary JSON only. Never submit Kaggle source CSV, official weight files, credentials or large prediction files.

**Gates:** G0 owner signatures pending; G1/G2 local reported and partial source inspected; G3 numerical agreement engineer-reported but **strict parity and official-reference branch require code/evidence repair**; G4 Global LGBM is `BLOCKED_INVALID_IMPLEMENTATION`; six-window simple baselines are `PROVISIONAL`.

