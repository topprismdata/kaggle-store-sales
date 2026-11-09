# M5 W1 G3 and Global LightGBM implementation contract — V1.1

Status: APPROVED_FOR_IMPLEMENTATION / G3_SCORER_NOT_VERIFIED

## 1. Evidence classes

- Windows preflight 6/6 and pytest 12/12: ENGINEER_REPORTED_PASS pending stdout / run manifests.
- Data CSV dimensions / five files and the 12 hierarchy *counts*: ENGINEER_REPORTED_PASS. The current 01_audit_data.py checks file dimensions; the current 02_verify_hierarchy.py checks group counts. Neither fully verifies ID uniqueness, validation/evaluation historical equality, price join validity, aggregate conservation, or per-key sale alignment. These remain outstanding G1/G2 tests.
- Official companion file access: SOURCE_VERIFIED. Dataset validation / evaluation weights and reference R scripts located, with precise source identifiers in [reference/](../reference/README.md).
- G3 official WRMSSE equivalence: NOT_VERIFIED. Running a dimensional audit, helper RMSSE unit tests, or validating weight file structure is NOT a pass.

## 2. Official G3 test order

1. Download Dataset/weights_validation.csv, Dataset/weights_evaluation.csv, and optionally validation/weights_validation.csv to an authorized LOCAL directory. Keep originals immutable. Fingerprint every file.
2. Run scripts/05_validate_reference_weights.py and ensure 42,840 keyed rows, level weights sum approximately 1, sales totals agree, and compact/key alignment passes. Primary validation has columns Dollar_Sales + weight; compact validation has only Weight.
3. Implement twelve-level daily summation with strict key mappings. Prove for every forecast day that Level1 total equals the sum of the 30,490 bottom series, and cross-level aggregation is correct.
4. Independently derive bottom-level last-28-day revenue and aggregate through all levels; compare official Dollar_Sales and reference weight per key. First normalize scale and cost conventions, then compare precision. Do not align by CSV row number.
5. Derive RMSSE scale from each grouped training history starting at first strictly positive sale. For zero-scale / missing active history, fail with explicit status; separate zero-weight from zero-scale cases.
6. Implement the official scorer's per-series numerator (28-day mean squared forecast error), per-level weighted RMSSE, and equal average across twelve levels. Compare on published benchmark forecasts against official reference outputs; verify both overall and all twelve levels before G3 PASS.
7. Use the official validation round with train sales through d_1913. Do not use evaluation weights for public-validation scoring; final evaluation uses training through d_1941 and is a separate experiment.

## 3. Global LightGBM — Owner-authorized initial model contract

| Contract | Frozen rule |
|---|---|
| Model | One global bottom-series one-step LightGBM, recursively predict F1–F28; CPU only initially |
| Train target window | Per series last 365 eligible days <= forecast origin; for the first origin d_1913 this is d_1549 through d_1913 |
| Approximate row count | 30,490 x 365 = 11,128,850 *before invalid-history/exposure filters*; not a guaranteed 11.13M rows |
| Primary lag features | lag 7, 14, 28, 56, 364 |
| Rolling | mean/std over predeclared window(s), computed from shift(1) or equivalent strictly past values |
| Known-future features | weekday, month, known event/calendar and SNAP information |
| Static features | item, dept, category, store, state, encoded deterministically |
| Price | Main causal-availability-safe arm excludes unknown future prices; explicit separate sensitivity arm only when advance visibility has been demonstrated |
| Training schedule | Time-origin based, future sales embargoed, recursive features recomputed from predicted history only |
| Smoke | 3,000 bottom-series stratified sample (seed frozen, selection hashed), then full 30,490 series |
| Determinism | Python 3.11.15, fixed code/data/config hashes, LightGBM deterministic=true, force_row_wise=true, num_threads=28 on Windows dev, fixed all seeds |
| Scoring | G3-verified official scorer only for official WRMSSE; smoke scores clearly labelled partial / non-comparable |

One-step dynamics might benefit from lag 1 and other short lags, but **do not silently insert them** into the frozen primary contract. Create a separate predeclared ablation with lag 1 if desired. Same for future-price features, rolling window variants, and lag364 removal.

### Recursive inference information contract

For horizon 1, features use only information known through origin d_1913. After predicting d_1914, add the predicted value (not true d_1914 sales) to the within-series simulation state, then forecast the next day. Recompute lag and rolling features at every step. The full horizon must never read future actual sales as features. Preplanned known-future calendar fields may be read, but not hidden labels.

### Hard limitations on performance estimates

The engineer's 'minutes-scale full training' and '<8GB peak RAM' are **not verified**. 11M rows plus category encodings, lag/rolling arrays, duplicated dataframe copies and histogram workspace can exceed the estimate. Measure peak resident memory and wall clock in smoke, and stop/escalate if the full-run budget or memory cap is exceeded. Do not promise a wall-time deadline.

## 4. Six rolling windows / metrics

Frozen prediction origins: d_1773, d_1801, d_1829, d_1857, d_1885, d_1913, each with 28-day horizon. Every window must output:

- overall and Level1–Level12 WRMSSE;
- active series group: all-zero, recently active, intermittent, high-volume, long-tail (categories and thresholds frozen on training slice);
- store, category and state splits; rank reversals, dominant-series weight contributions;
- model, sample, feature, code SHA, data manifest SHA and environment ID;
- separate Windows development and Linux canonical result namespaces.

## 5. Review-ready engineering deliverable

Submit a Git diff that includes production hierarchy aggregation/scorer, synthetic and official-reference alignment tests, Naive7/SeasonalMean28, and LightGBM runner with smoke diagnostics. Include a reference comparison report with per-level error maxima and nonzero-scale exception counts. No pasted opaque overall WRMSSE alone. Any unresolved G3 mismatch remains SCORER_NOT_VERIFIED; can continue model coding but cannot claim official scores.

References: [official M5-methods repository](https://github.com/Mcompetitions/M5-methods), [companion folder](https://drive.google.com/drive/folders/1D6EWdVSaOtrP1LEFh1REjI3vej6iUS_4), [reference source index](../reference/README.md).