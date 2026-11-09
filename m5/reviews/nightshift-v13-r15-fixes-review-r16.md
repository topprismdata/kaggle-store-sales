# M5 W1 · R16 independent v13 audit — A/B two-track decision

**Decision: A_ENGINEERING_FINAL_NO_GO / B_OFFICIAL_INDEPENDENT_REPLAY_NOT_EXECUTED / G3_INDEPENDENT_PASS_NO_GO.**

## Input verification
- Actual upload `m5-w1-r15-fixes.patch`: **71,146 B**, SHA256 `db711c99295942027b3622d4ff5d98a10f8d4a9169b6cad0f98b0072964800cd`; matches claim.
- Contrary to the "9 commits" statement, the exact uploaded `git format-patch` contains **6** mail patches: `648ef8e`, `b0689cb`, `8784b90`, `ad3f5a8`, `e954202`, `3fb7d84`. Only **3fb7d84** is new relative to independently audited R15 worktree; apply it only, `git apply --check` / `git apply` PASS.
- New diff: **only** `m5/scripts/07_six_window_summary.py` changes (49 lines). Python compileall PASS; pytest **35/35 PASS**, no new persistent negative-test file.
- With a coherent 14-run synthetic fixture, invoke **real production 07.main()** and mock only M5 input loading in absence of raw licensed data. Repeat 26 R15 failure-injection cases: **23 expected / 3 false PASS**. Add 5 strict identity/round/path cases: **31 cases, 23 expected / 8 false PASS**. This is NOT independent real-data WRMSSE replay.

## Valid fixes confirmed
- Missing `run_manifest.forecast_sha256/config_sha256/data_manifest_sha256` fail; present-but-wrong SHA fail.
- Common forged canonical hash across 14 artifacts now fails against `id_order_sha256(load_id_frame(data_dir))`, a real third anchor.
- Wrong score filename within same run is rejected; the forecast content SHA check, finite score checks, 12-level recomputation, incorrect arm, and wrong round protections remain effective.
- Valid synthetic 14-run baseline passes.

## Eight current false PASS paths (all exercised via 07.main())

| Failure injection | Finding / location |
|---|---|
| `score.actual_file_sha256` wrong | Still PASS. 07 requires a non-empty field, but does **not** hash the actual source file; see 07:232–234,318–323. |
| `SCALE_SOURCE_OFFICIAL_PARITY` + fabricated nonempty weight SHA, **no --reference-dir** | PASS. `known` empty and `if known and ...` skips verification; 07:387–413. |
| `SCALE_SOURCE_OFFICIAL_WEIGHTS_SELF_SCALES` + fabricated nonempty weight SHA, **no --reference-dir** | PASS. Same bug. |
| Foreign absolute score prediction path ending in `<run_id>/forecast.csv` | PASS. Only `endswith`, no normalized exact path compare; 07:225–231. |
| `run_manifest.forecast_file` names unrelated foreign file | PASS. No exact binding to actually loaded `fc_path`. |
| Custom origin 1857 claims a **genuine existing evaluation** reference SHA with provided --reference-dir | PASS. Reference hash set not linked to origin/round. |
| Custom origin 1885 claims a **genuine existing validation** reference SHA with provided --reference-dir | PASS. Same origin mismatch. |
| Explicit reference directory exists but contains **zero valid reference files**; OFFICIAL claim has fake SHA | PASS. `known=empty` wrongly accepted. |

## Source-level root causes and minimal exit
- `07:391–402`: optional `--reference-dir` and `if known and ...` is not fail-closed. For any official route, require a present, expected **round-specific** authorized weights file and strict content SHA. The six historical custom windows should have a frozen self-computed policy, not accept unrelated official weights hashes. Confirm that recomputation actually uses the claimed route; current `_recompute_overall()` always self-computes.
- `07:225–231`: replace suffix with resolved realpath equality, compare score.predictions ↔ manifest.forecast_file ↔ `runs/<run_id>/forecast.csv`; handle symlinks/relative path explicitly.
- `07`: derive actual file from registered origin/day contract, hash the **real bytes used by the scorer**, and compare score.actual_file_sha256, not only train_file_sha256. Add tamper-rejection tests at entrypoint.
- Add production `07.main()` tests for **all 8** and one official-reference legitimate positive case to CI, freeze only after fresh clean-tree rerun.

## A/B signoff and W2
- **A track:** `PARTIAL_ACCEPT / FINAL_NO_GO`. Many engineering fixes are valid; however 8 provenance attacks still return PASS. Do not merge full patch or mark verified release.
- **B track:** `NOT_EXECUTED`. Independent evaluator has not received approved complete M5 raw data/actual/reference assets and an independently coded official scorer; cannot claim independent 42,840-key validation/evaluation replay. Continue using R15 `M5_W1_R15_AB_Gates_and_Independent_Replay_Freeze_Spec.md`: freeze data/code/env/reference SHA, roles, tolerance and 12-level/per-key outputs before execution.
- **G3:** `LOCAL_NUMERIC_PARITY_STRONGLY_SUPPORTED / SOURCE_GATE_PARTIAL / INDEPENDENT_REPLAY_PENDING / FINAL_NO_GO`.
- **Methodology:** submitted six historical custom windows have Mean28 overall 6/6 ahead of Naive7, summary arithmetic verified. Descriptive only, not independent generalization or inventory-return proof.
- **W2:** design/exploratory GO; confirmatory NO-GO pending genuinely unseen test window, preregistration, permissions, A/B gates.
