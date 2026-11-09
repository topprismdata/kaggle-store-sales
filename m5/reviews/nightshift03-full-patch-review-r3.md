# M5 Hands-on W1 — Full Patch Audit R3

**Verdict:** `CHANGES_REQUESTED / DO_NOT_MERGE`  
**Source:** conversation-uploaded `m5-w1-g3-review(2).patch`  
**SHA-256 (independently measured):** `cbc13a08d5d286fda2d2e983161fdb90efc7fbca80974a6f304d3fadc93d8936`  
**Size:** 122,522 bytes  
**Scope:** 4 patch emails, **not 3**:
1. `21a31cbabadba963875270662297c0de2fba363f` — initial G3/scorer/model/code/evidence.
2. `ef54be102a3c7ef92b09718ca29d6397176bf5b3` — six-window summary.
3. `3fce96ccb25e1cd0c00cd640ea7b337c8a222d31` — explicit run-ID mapping.
4. `02b2da92b4e030af1a3be1a62569fba61d9955a1` — round-parametric 03/04 and full precision evidence.

## A. Review method and what was actually tested

- SHA-256 and patch commit boundaries checked on the **actual uploaded bytes**.
- Recreated the first patch's source snapshot; applied commits 2, 3, 4 separately: `git apply --check` and `git apply` all passed.
- Isolated review worktree included a minimal local `metrics.wrmsse` shim solely because the earlier base-repository module was absent from the extracted patch snapshot; this is **not** full original-repository end-to-end test. The unchanged base's `03_run_baselines.py` entrypoint line `raise SystemExit(main())` was restored in the worktree (the patch does not remove it).
- `compileall` succeeded after reconstructing that unchanged base context.
- Executed **5/5 tests from the supplied `m5/tests/test_scoring.py`** against the isolated snapshot. This **does not verify** the claimed `20/20` in the Windows repository, because the full base scaffold was not checked out in this isolated worktree.
- Three additional original synthetic reproductions were run: calendar alignment, future price sensitivity and positive-weight undefined RMSSE; one additional zero-heavy stratified sampler test was run.
- Independently validated internal arithmetic of every window's 12 levels and recomputed all 12 official-layer differences stored in `g3_snaive_reproduction_full_precision.json`.
- Actual M5 CSV contents and complete Windows run logs were unavailable; reported end-to-end scores are `ENGINEER_REPORTED` supported by artifacts but not independent raw-data replays.

## B. The new patch does NOT contain the reported four P0 repairs

A search of the **entire four-commit uploaded patch** shows no `m5/tests/test_feature_alignment.py`, no `_visible_prices()`, no `--phase` or enforced confirmatory hash gate. The original offending lines remain in source.

| P0 | Source location | Actual uploaded code | Proof / remediation |
|---|---|---|---|
| **P0-1 Calendar/event/SNAP label misalignment** | `m5/src/m5research/models/global_lgbm.py:117–122` | `np.repeat` day-major against series-major flattened `y` | Original two-series × 365-target-day fixture: **312 / 365** wrong weekday rows for the first series. Replace with series-major aligned `tile` / explicitly keyed matrix, test every feature. |
| **P0-2 Future price leakage** | `global_lgbm.py:199–201,215–238` | full future prices read for recursive prediction; fallback chosen via `tail(1)` on all weeks | Synthetic booster using `sell_price` output: perturbing only future-week prices changed forecasts by **100 units**. Main arm must cut at origin; separate competition-covariate arm if needed. |
| **P0-3 365-day refit not performed** | `m5/scripts/03b_global_lgbm.py:99–115` | 28-day `tail_rows` removed unconditionally; fixed-rounds still trains **337 of 365 targets** | In fixed-round mode, refit on all 365 after frozen round selection. Keep `train_target_days_per_series` and `refit=true` in manifest. |
| **P0-4 NaN masking** | `m5/src/m5research/scoring.py:204–207` | `np.nansum(level_rmsse * weights)` | Synthetic **positive weight 1.0 + NaN RMSSE** returned `0.0` instead of failing. Must throw with level and series keys for invalid positive-weight entries. |
| **P0-5 Report/design isolation remains non-enforced** | `m5/scripts/03b_global_lgbm.py:60–69` | CLI has `--objective`, `--fixed-rounds`, `--origin`; **no `--phase` or frozen hash** | `d1913` was already viewed; later `d1913` runs remain post-selection. Source-level controls and truly untouched confirmation required. |

**Additional independent reproducible issue:** `global_lgbm.py:250–262` calls `pd.qcut(recent, 3, labels=[0,1,2], duplicates="drop")`. A zero-heavy synthetic sample raised `ValueError: Bin labels must be one fewer than the number of bin edges`. Add a deterministic fallback for collapsed activity bins and test the smoke selection counts.

## C. What WAS repaired in 02b2da9

1. `m5/scripts/03_run_baselines.py` adds `--train-file`, allowing `sales_train_evaluation.csv` with `origin=1941`; appropriate idea. It still relies on caller to choose a history source with the correct cutoff.
2. `m5/scripts/04_score_wrmsse.py:29–83` adds `--round validation|evaluation`, resolves origin, historical labels, actual labels and weights per round. Correct improvement over original stub.
3. `m5/scripts/04_score_wrmsse.py:141–145` persists `per_level_full_precision` and a six-decimal display-only version. This resolves the earlier artifact-rounding disagreement.
4. `m5/scripts/04_score_wrmsse.py:208–239` recalculates Dollar_Sales; however, **`_crosscheck_weights` returns `status: OK` without threshold-enforcing a failure** even if there is a large mismatch. Add an error/exit gate. Do not make `SCALE_SOURCE_OFFICIAL_PARITY` automatic.
5. `m5/scripts/04_score_wrmsse.py:65–74` allows `--round` and `--origin` to override each other without a consistency check, enabling mismatched weights and window labels; reject inconsistent combinations.
6. `m5/scripts/04_score_wrmsse.py:162–169` loads actual CSV by row position without checking identifier alignment with the training series. Reference files may be sorted consistently, but robust G3 requires explicit series-key validation and provenance.

**Important:** Status in `04` is `SCALE_SOURCE_OFFICIAL_WEIGHTS_SELF_SCALES`, not proof of independently matched official scales. It is honest to retain that distinction.

## D. New full-precision G3 evidence: internal consistency verified

- `g3_snaive_reproduction_full_precision.json` contains **12 unrounded layer values**.
- Recomputed max `abs(reproduced_layer - reference_layer)`: **1.4909784518124525e-10**, exactly the JSON value.
- Recomputed average of the 12 reproduced layers: **0.8470173528345969**, matching the artifact.
- Independent companion `Scores and Ranks.xlsx` evaluation-round official sNaive: **0.8470173528438624**; absolute overall difference ≈ **9.27e-12**.
- `g3_snaive_o1941_score_evaluation.json` correctly labels `round=evaluation`, `origin=1941`, points to evaluation weights and released test actuals; it still contains machine-specific absolute paths and was not replayed on actual raw files here.
- `g3_weights_crosscheck.json` reports validation-round max abs weight difference zero after 12-digit rounding; the checker `06_g3_weights_crosscheck.py:117–119` allows tolerance `1e-9`, so its PASS gate **does not enforce bit equality** despite reported observed zero difference.

**G3 decision:** `LOCAL_NUMERIC_PARITY_STRONGLY_SUPPORTED / INDEPENDENT_SOURCE_AND_RAW_DATA_EQUIVALENCE_PENDING`. Model training remains blocked by other P0 defects.

## E. Six-window evidence — internal arithmetic verified, source not replayed on M5 data

| Origin | Naive7 | Mean28 | Frozen-LGBM in artifact |
|---:|---:|---:|---:|
| 1773 | 1.411421 | 1.204080 | — |
| 1801 | 1.112062 | 1.066091 | — |
| 1829 | 1.021062 | 0.851218 | — |
| 1857 | 0.888939 | 0.717579 | 0.935785 (INVALID MODEL CODE) |
| 1885 | 0.922781 | 0.658369 | — |
| 1913 | 0.869701 | 0.760672 | 0.916384 (POST_SELECTION + INVALID MODEL CODE) |

- 12 layer average matched reported overall to numerical floating-point precision in every displayed entry.
- `Mean28 < Naive7` in **all six windows**. Mean overall across windows: Naive7 **1.0376609296**, Mean28 **0.8763347492**. This is provisional within-window stability, not inference over independent markets.
- `m5/scripts/07_six_window_summary.py:54–67` reads `score.json` by hardcoded run-ID without validating that each score artifact actually contains the requested `origin` or correct model label; potential misattribution remains.
- `07:73–76` uses the overall ranking to compute a field called `windows_with_mean28_first_of_simple`; if LGBM later wins overall, this will not measure “first of simple” as named.
- Per-origin weights/scales are recomputed per run *according to the engineer's code/report*; official dataset was not available here for independent recomputation.

## F. Other per-file review findings and release matrix

| Files | Review result |
|---|---|
| `src/m5research/models/seasonal_naive.py`, `tests/test_scoring.py` | **APPROVABLE_AFTER_TESTS**. Seasonal7 and Mean28 model formulas are coherent. Five synthetic tests passed; insufficient to claim full 20 test suite. |
| `scripts/03_run_baselines.py` | **NEEDS_P1**. Train history override added, but ID/order integrity and automatic round/cutoff checks absent. |
| `scripts/04_score_wrmsse.py`, `src/m5research/scoring.py` | **BLOCK_P0**. NaN masking, missing mandatory fail on weight/Dollar_Sales mismatch, row-order assumptions and unchecked round overrides. |
| `scripts/06_g3_weights_crosscheck.py` | **NEEDS_P1**. Reports abs/relative differences; tolerance does not guarantee bitwise equality, lacks strict key completeness/duplicate check. |
| `scripts/01b_content_audit.py` | **NEEDS_P1**. `window_weeks` filter `d_num > origin-28` has no `d_num <= origin`; claimed “13-week window” is an audit bug. Pair coverage tests any week, not each required week. Event columns status hard-coded True. Numeric aggregation conservation not verified. |
| `scripts/07_six_window_summary.py` | **NEEDS_P1**. Validate origin/model/weight scheme in each score artifact and compute simple-only ranking correctly. |
| `src/m5research/models/global_lgbm.py`, `scripts/03b_global_lgbm.py` | **BLOCK_P0**. Four previously identified P0 not repaired; `qcut` collapsed activity bins can crash smoke selection. |
| `evidence/data_manifest.json`, `reference_sha256.json` | **PROVENANCE_READY_AFTER_SANITIZATION**. Hashes and filenames only; absolute E: machine paths should be stripped for public release. |
| `evidence/g3_*_full_precision.json`, `g3_weights_crosscheck.json` | **REPORTABLE_AS_LOCAL_EVIDENCE**, include derivation/program SHA and round; not proof that original Kaggle data was independently replayed here. |
| `evidence/baseline_*.json`, `six_window_baseline_summary.json` | **PROVISIONAL_SUMMARIES**. Internal arithmetic consistent; real-data origin-specific scoring cannot be independently re-executed here. |
| `evidence/bt_lgbm_*.json` | **QUARANTINE_EXPLORATORY** until fixed model code and reruns. |

## G. Remediation exit tests

1. Add the **actual** `test_feature_alignment.py` under the committed patch; assert all features/labels in multi-series fixture. Check no `np.repeat(day_df` remains.
2. Perturb post-origin prices and prove main-arm forecasts are invariant, including fallback.
3. Show `fixed_rounds=1300` refit has `n_series * 365` eligible target rows, not `n_series * 337`.
4. Positive-weight zero RMSSE must raise; zero-weight undefined must follow one clearly documented policy.
5. Make `--phase` and `--round` mutually consistent with frozen config, origin and input hashes. No rebranding post-selection d1913 as untouched confirmation.
6. Confirm unrounded 12-layer benchmark and keyed weights/scale comparisons on validation/evaluation rounds with source hashes.
7. Add ID-key validation for actuals/train/forecasts, and tests for zero-heavy stratified sampling and origin integrity in six-window summary.
8. Re-run affected model windows and issue a **new** unambiguous patch; do not delete old exploratory runs.

**Final:** `HASH_VERIFIED / FOUR_PATCHES_APPLY / G3_EVIDENCE_INTERNALLY_CONSISTENT / FOUR_P0_UNFIXED / DO_NOT_MERGE`.
