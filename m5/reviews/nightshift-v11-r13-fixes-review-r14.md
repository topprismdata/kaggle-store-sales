# M5 W1 — R14 v11 independent acceptance review

**Decision: CHANGES_REQUESTED; G3_INDEPENDENT_PASS = NO-GO; SIX_WINDOW_DESCRIPTIVE_ARCHIVE = ACCEPT; W2_DESIGN = GO; W2_CONFIRMATORY = NO-GO**

## Source integrity and execution

- User patch `m5-w1-r13-fixes.patch`: verified **94,782 bytes**, SHA-256 **888b230fa60d5a845b60e73c519f07a0dfec5388ce6eb7455fb6206c12e0ca34**.
- Actual patch includes **six** email commits: `6de0b39`, `4fe98a0`, `648ef8e`, `b0689cb`, `8784b90`, `ad3f5a8`; the last two are new relative to the prior R13 reviewed snapshot. Applied only those two using `git apply --check` and `git apply`: PASS.
- Python compilation PASS; existing `pytest m5/tests`: **35/35 PASS** in isolated Python 3.13 environment, not the engineering Windows locked environment.
- Directly executed the production `07_six_window_summary.main()` with 14 coherent synthetic run directories and targeted file/data mocks (original M5 files are unavailable). **21 entrypoint scenarios, 13 meeting intended behavior, eight malformed/forged scenarios wrongly accepted**. This verifies production control flow and fail-closed logic, not full Kaggle dataset numerical replay.

## Independent fault-injection ledger

| Fault injected | Actual 07 behavior | R14 |
|---|---|---|
| Valid 14 synthetic runs | PASS | Expected |
| Claimed overall zero, recomputation=2 | REJECT | CLOSED |
| Claimed overall NaN | REJECT | CLOSED |
| All 12 per-level scores forged to zero | REJECT | CLOSED for finite values |
| **One per-level score NaN** | **PASS** | **P0 OPEN** |
| Round `evaluation` at origin1857 or `custom_window` at origin1913 | REJECT | CLOSED |
| Unrecognized `SCALE_SOURCE_FAKE` | REJECT | CLOSED for unknown enumerands |
| Wrong but allowed `SCALE_SOURCE_OFFICIAL_PARITY` for self-computed window | **PASS** | P1 OPEN |
| Manifest forecast SHA modified to invalid non-empty SHA | REJECT | CLOSED |
| **Manifest config SHA modified** | **PASS** | P0 provenance OPEN |
| **Manifest data-manifest SHA modified** | **PASS** | P0 provenance OPEN |
| **Manifest forecast SHA field missing** | **PASS** | P0 provenance OPEN |
| Score train SHA modified | REJECT | CLOSED |
| **Score actual SHA modified** | **PASS** | P1 OPEN |
| **Score predictions path changed to other same-run filename** | **PASS** | P1 OPEN |
| **Score and manifest canonical ID SHA forged identically across all 14 runs** | **PASS** | P0 identity OPEN |
| Forecast bytes changed | REJECT | CLOSED |
| Data manifest or config file missing | REJECT | CLOSED |

### Precise causes and exits

1. `07_six_window_summary.py:282-291`: per-level claimed scores are checked using `abs(float(claimed_level)-recomputed)>1e-6` without first demanding `np.isfinite`. As `NaN > tol` is False, corrupt claimed layer is falsely accepted. Explicitly validate finite for each of the 12 claimed/recomputed levels and use strict JSON serialization with `allow_nan=False`.
2. `07:320-325`: `if man.get('forecast_sha256') and ...` is **not** a mandatory field gate; missing forecast SHA is accepted. Manifest `config_sha256` and `data_manifest_sha256` are not compared against actual config/data bytes, despite score-side hashes being validated. Require score ↔ run manifest ↔ source-file SHA equality and fail on every missing hash.
3. `07:216, 340-342` checks that score/manifest ID SHA match each other and across runs, **not** against the canonical order actually derived from `id_order_sha256(load_id_frame(data_dir))`. Forging all 14 claims consistently passes. Must verify the trusted third anchor.
4. `07:299-308` accepts an unrelated but valid scale-source enum; bind a precise scale-source policy per historical replay. Also verify score actual-file content hash and exact normalized predictions path instead of the `run_id in predictions` substring check.

## o1857 immutable-run claim and retained evidence

- The `RUN_IDS` o1857 LightGBM mapping now references **`M5A_W1_BT_lgbm_t1300fix_o1857_c1`**. New `m5/evidence/lgbm_t1300fix_o1857_c1_score.json` has overall **0.80743584815725**, precisely consistent with six-window summary and 12-level mean **0.8074358481572498**.
- The original `bt_lgbm_t1300fix_o1857_score.json` retains **0.8184498373211491** and old run path. Thus the previous *run-label collision in submitted evidence* is resolved.
- Actual `_c1` run's forecast.csv, run_manifest, immutable-storage evidence and full training log are **not attached**. Cannot independently sign the asserted cross-process determinism or immutable execution. Allowed statement: `RUN_ID_SEPARATION_AND_SUMMARY_ARITHMETIC_PASS / REAL_RUN_IMMUTABILITY_NOT_VERIFIED`.

## Six-window formal methodologic decision

Six custom historical backtest windows: Mean28 overall WRMSSE improves over Naive7 **6/6**. Unweighted window means **0.876334749 vs 1.037660930**, average difference about **0.16132618** (relative decrease **15.55%**); L1-L5 each 5/6 winning windows, L6-L12 each 6/6, o1801 only 7/12 levels. **Approved as limited descriptive evidence**: `HISTORICAL_SIX_WINDOW_OBSERVATION_ACCEPTED / SUMMARY_ARITHMETIC_VERIFIED / GENERALIZATION_NOT_CONFIRMED`. Six windows share historical data and have already been opened; not independent experiments, not a generalization or inventory-benefit claim.

## G3 three-pillar ruling

| Pillar | R14 |
|---|---|
| A. Local official numerical parity | **STRONGLY_SUPPORTED, ENGINEER_REPORTED**: sNaive evaluation 0.8470173528345969 versus official 0.8470173528438624; previous 12-level maximal abs diff about 1.49e-10. Not independently recomputed from raw data this turn. |
| B. Provenance and recomputation gates | **PARTIAL PASS**. 35/35 tests; only 13/21 direct entrypoint cases meet expected rejection semantics. |
| C. Independent full original-data replay | **NOT DONE**. Original 42,840-series M5 input and original 14 true run files unavailable in review environment; no independently coded official WRMSSE end-to-end validation. |

**G3_INDEPENDENT_PASS = NO-GO**. The same-repository 07 rescoring adds engineering evidence but is not an independent second implementation.

## W2 exit criteria

- G0 owner / authorized data custodian / independent evaluator assignments and provenance log.
- Close eight false-PASS classes with persistent production 07.main() adversarial CI. Validate exact hashes of train/actual/forecast/data/config/reference and anchored canonical row order; require exact scale source/origin/round/model provenance.
- Complete independently run original-data reference for validation/evaluation 12 levels and 42,840 series, freeze code/data/reference/weights hashes and results.
- Preregister a truly unopened future period or external independent data, frozen strong Mean28 baseline, fixed model arm, metrics, statistical unit and multiplicity before any confirmed evaluation. Already viewed origins 1773/1801/1829/1857/1885/1913 and official evaluation cannot be rebranded unopened.
- **W2 design/exploratory GO**: predefine calendar alignment, price visibility, 365-day refit C/P/R factor contrasts and Mean28 holiday-versus-noise mechanisms. **W2 confirmatory NO-GO**.

R14 result: `PATCH_SHA_PASS / 6_COMMITS_2_NEW / PYTEST_35_PASS / 8_FALSE_PASSES_IN_21_ENTRYPOINT_PROBES / O1857_RUN_ID_COLLISION_CLOSED / G3_FINAL_NO-GO`.

Complete Chinese report and independent execution files are supplied as conversation artifacts `m5_r14_review_report.md`, `m5_r14_independent_checks.py`, `m5_r14_independent_results.json`, `m5_r14_acceptance_result.json`. No Kaggle original data uploaded to GitHub.
