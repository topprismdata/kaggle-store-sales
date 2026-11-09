# M5 W1 G3 — Rework R2 Intake & Evidence Decision

**Review status:** `REPORTED_REPAIR / NEW_PATCH_NOT_RETRIEVED / NOT_MERGEABLE`  
**Base:** `research/m5-study-v1`  
**Previous inspected patch:** `21a31cb`, SHA-256 of original attached patch `fee523e29e4f28362740c5ebc63d357d8c0619ac59dc623cb3228b60aba489b5`  
**Latest reported local commit:** `02b2da9` (after `21a31cb` and `3fce96c`)  
**Current source accessibility:** `https://dpaste.org/1cEdU/raw` supplied in user report; raw URL was inaccessible through available network fetch methods (page fetch cache miss; container DNS unavailable). The previous attached patches are **only** the initial `21a31cb` patch. **No new code diff from `02b2da9` was independently inspected or applied.**

## 1. R2 claims and external evidence check

| Item | Engineer-reported rework | Independently verified here | Status |
|---|---|---|---|
| `04_score_wrmsse.py` | `--round validation|evaluation`, origin / actuals / training/reference data resolved per round | New source bytes unavailable | `REPORTED_NOT_CODE_VERIFIED` |
| `03_run_baselines.py` | `--train-file` selects validation/evaluation history | New source bytes unavailable | `REPORTED_NOT_CODE_VERIFIED` |
| Validation Naive7 | 0.869700702 | Consistent with previously reported 0.8697; run not reproduced here | `ENGINEER_REPORTED` |
| Evaluation sNaive | 0.8470173528345969 | Official `Scores and Ranks.xlsx`: **0.8470173528438624**. Absolute overall difference: **9.265477274311706e-12** | `OFFICIAL_REFERENCE_CONFIRMED / LOCAL_REPRODUCTION_REPORTED` |
| G3 per-level parity | Full precision max absolute difference 1.4909784518124525e-10 | Full precision JSON / code cannot yet be inspected | `ENGINEER_REPORTED` |
| Dollar_Sales | max relative diff 4.32e-16 | Detailed per-key file missing | `ENGINEER_REPORTED` |
| Memory peak | RSS 4.76 GB, psutil 0.25s sampling; train 129.6s, inference 13.9s | No measurement log accessible here | `ENGINEER_REPORTED` |

The overall evaluation-round error is consistent with the claimed maximum per-level discrepancy; both can be true. This **does not establish code equivalence** in the absence of raw 12-level and source artifacts.

## 2. Rework acceptance checklist for 03/04

- `--round evaluation` must use training through d_1941 and explicitly released evaluation target d_1942–1969, with immutable file hashes and exact series-ID alignment.
- `--round validation` must use training through d_1913, targets d_1914–1941, and **validation weights**. No use of evaluation-round weights in the validation arm.
- Custom backtest `--origin` mode must use origin-specific training scales and dollar-weights, with explicit row-level price-availability controls. Passing `--train-file` alone does not guarantee absence of future labels.
- `--reference-dir` must resolve by exact expected filename **and verified layout**, with no silent fallback to compact `weights_validation.csv`; compare by `(Level_id, Agg_Level_1, Agg_Level_2)`, never CSV row position.
- Preserve all 12 **unrounded** per-level scores and compare against the official reference *of the same round*. `per_level` may hold display-rounding but must not be used to compute parity.
- Positive-weight undefined/NaN RMSSE must raise rather than vanish in `np.nansum`; zero-weight undefined RMSSE handling must have an explicit documented rule and tests.
- A valid `G3_PARITY` claim requires a complete code SHA, hashes of raw/reference/score files, zero-scale exception counts, 42,840 keyed weighted score inputs, and 12 per-level differences.

## 3. Unresolved earlier P0 findings not addressed by the reported R2 change set

The engineer says latest R2 changes affected 03/04 and evidence only. **Unless the newly delivered patch actually includes additional source changes**, the following issues found in `21a31cb` remain open:

1. **Calendar/event/SNAP feature rows misaligned with series-major training labels** in `global_lgbm.py`.
2. **Future price and post-origin fallback price leakage** in `global_lgbm.py` for the main decision-safe arm.
3. **Fixed 1,300-round refit trained on 337 rather than frozen 365 target days** in `03b_global_lgbm.py`.
4. **`np.nansum` silently drops undefined positive-weight contributions** in `scoring.py` unless separately repaired.
5. **Exploratory look at origin 1913** means later runs at this origin are post-selection, not untouched confirmatory evaluations.

Therefore **do not merge the Global LightGBM scores**, do not declare them confirmatory, and do not start W2 model comparison as if these findings were resolved. The G3 evaluation-round parity can be accepted for *local development continuation* pending code/audit review, not independent final signoff.

## 4. Source handoff needed

Provide the complete **three-commit patch** including `02b2da9` as a conversation file attachment, or push it to a readable GitHub branch. Use only source, tests and sanitized summary JSON; no raw Kaggle data, reference CSV/R original, credentials, user-specific local paths or large predictions. After receiving the actual bytes, apply with `git apply --check` to a throwaway tree on this branch, inspect diffs, run tests, and only then commit accepted changes.

**Final R2 decision:** `REPAIRS_PLAUSIBLE / OFFICIAL_SCORE_VALUE_VERIFIED / NEW_PATCH_NOT_INSPECTED / CHANGES_REQUESTED`.
