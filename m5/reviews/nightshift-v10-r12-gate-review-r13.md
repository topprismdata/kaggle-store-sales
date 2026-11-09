# M5 W1 — R13 independent code audit of R12 re-scoring gate

**Decision: CHANGES_REQUESTED / G3_INDEPENDENT_NO-GO / W2_DESIGN_GO**

## Artifact
- Uploaded patch: `m5-w1-r12-fixes.patch`
- Verified size **33,302 B**; SHA-256 `93f8d89b1ad87cdde3939db03193ace5a6bc377a3cf764ecc5ae43b4012af4b6`.
- Contrary to the one-commit description, patch has **2 email commits**: previously applied `648ef8e` and new `b0689cb`. Independently applied only `b0689cb` on R12 snapshot: `git apply --check`, `git apply`, compileall PASS; `pytest` **35/35 PASS**.
- Source/data boundary: the real raw M5 files and original 14 run forecasts/score/manifests were not attached. **No independent full M5 replay occurred.** Review uses production 07.main(), synthetic small-input scoring adapters, and independently constructed adversarial 14-run artifacts.

## What passes
- `07_six_window_summary.py` now reads actual forecast file bytes, loads origin-specific sales/actual, reconstructs weights/scales via `m5research.scoring` and compares claimed overall WRMSSE with a recomputed value, absolute tol 1e-6.
- A legitimate synthetic 14-run collection yields PASS. Replacing a finite claimed overall **2.0 → 0.0** is rejected. Corrupting forecast CSV content without updating the bound content hash is rejected. Missing config/data manifest is rejected. Positive-weight undefined forecast RMSSE is rejected.

## Remaining actual false-PASS paths (entrypoint tests)
All tests below called **real `07.main()`**, using tiny synthetic inputs and mocked M5 scoring loaders only to avoid unavailable raw Kaggle files:

| Tamper | Actual 07 outcome | Decision |
|---|---|---|
| Set claimed overall score to Python-JSON nonstandard `NaN` | **PASS; NaN written to summary** | **P0**: `abs(recomputed - claimed) > 1e-6` is false when claimed NaN; no finite check |
| Set every `per_level_full_precision` score to **0** while keeping overall valid | **PASS; fake per-level values copied** | **P0**: only overall recomputed, not 12 layer scores |
| Change round to `evaluation` for historical custom origin 1913 | **PASS** | **P0/P1**: enum membership is not the origin-specific contract |
| Set `scale_source=SCALE_SOURCE_FAKE` | **PASS** | **P0/P1**: prefix is not exact scoring provenance |
| Set `run_manifest.forecast_sha256` to wrong hash | **PASS** | **P0**: previous manifest-to-file content gate removed |
| Set `run_manifest.config_sha256` to wrong hash | **PASS** | **P0** |
| Set `run_manifest.data_manifest_sha256` to wrong hash | **PASS** | **P0** |
| Set score `train_file_sha256` to wrong hash | **PASS** | **P0**: source train bytes are not rehashed against score |
| Set score predictions path to another file in same run directory | **PASS** | **P1**: only run-ID substring checked |
| Replace forecast with all NaN, and update score forecast hash | **REJECTED** | Gate works on positive-weight nonfinite recomputation |
| Zero out finite claimed overall | **REJECTED** | New replay works for this narrow attack |

### Root cause
New `07` validates `round` by `{'validation','evaluation','custom_window'}` and `scale_source` by string prefix; it checks report forecast SHA against bytes, but no longer compares the run manifest's `forecast_sha256/config_sha256/data_manifest_sha256` to real bytes. Source train/actual file hashes are present but not rechecked by 07. It only calls `_recompute_overall()` and then re-emits the **unverified** `per_level_full_precision` claimed by score. Crucially it does not require finite claimed/recomputed scores, so NaN can pass.

## Evidence conflict still pending
At origin 1857, `six_window_baseline_summary.json` associates run `M5A_W1_BT_lgbm_t1300fix_o1857` with **0.80743584815725**, while the retained independent evidence file `bt_lgbm_t1300fix_o1857_score.json` still says **0.8184498373211491** for that same run. No new run archive was attached to reconcile the difference. Origin 1913 is now labeled r006 with 0.5872038872 and remains `POST_SELECTION_EVALUATION` (old r005 retained).

## R13 required exit
1. Reject NaN/inf in all claimed and recomputed overall/level numbers; serialize released JSON with `allow_nan=False`.
2. Independently recompute and compare **all 12 per-level scores**, validate overall equals mean, and output recomputed trusted values.
3. Require exact `round='custom_window'`, `scale_source='SCALE_SOURCE_SELF_COMPUTED'` for six-window custom replays and pre-registered model/origin identity.
4. Reinstate report↔run_manifest↔actual forecast/config/data/train/actual file SHA checks; exact normalized prediction path, not substring. Add persistent real-entrypoint failure-injection tests.
5. Reconcile 1857 LGBM number under immutable run ID.
6. For **highest-tier independent G3 signoff**, an independently operated environment with licensed real M5 source inputs must recompute official validation/evaluation 12-layer and 42,840-key figures and verify input/reference/code hashes. Same-codebase 07 re-scoring is valuable second-pass evidence, **not an independently coded official WRMSSE scorer**.

## Methodology (already archived, not upgraded)
Six submitted historical custom windows show Mean28 overall **6/6** better than Naive7; mean WRMSSE **0.8763347492** vs **1.0376609296**, relative improvement about 15.55%; L1-L5 each 5/6, L6-L12 each 6/6. **DESCRIPTIVE_ONLY / NOT_UNSEEN_PERIOD_GENERALIZATION**.

**Final evidence taxonomy:** `PATCH_SHA_PASS / 2_EMAIL_COMMITS_ACTUAL / 35_TESTS_PASS / OVERALL_FINITE_TAMPER_REJECTED / NAN_SCORE_FALSE_PASS / LAYER_SCORE_FALSE_PASS / PROVENANCE_FALSE_PASS / LOCAL_NUMERIC_PARITY_STRONGLY_SUPPORTED / INDEPENDENT_RAW_DATA_REPLAY_NOT_DONE / G3_FINAL_NO_GO / W2_DESIGN_GO / CONFIRMATORY_NO_GO`.

Detailed independent test harness and results were saved in the conversation sandbox as `m5_r13_independent_checks.py`, `m5_r13_independent_results.json`, and detailed Chinese notes in `m5_r13_review_report.md`. No official Kaggle raw data was republished.
