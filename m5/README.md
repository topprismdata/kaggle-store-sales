# M5 Hands-on Reproduction

This is the executable research scaffold for the M5 Accuracy / Uncertainty reproduction.

## Status

- Code scaffold and preflight: added.
- Windows development environment: owner-reported, registered.
- Canonical Linux environment: designated, machine specification pending.
- Official companion Drive: access verified. Source links and the two different validation-weight variants are documented in [reference/README.md](reference/README.md). Official third-party files are not redistributed in this repository.
- Kaggle rules acceptor and data custodian: **not yet named**.
- Engineer-reported Windows preflight, pytest, data dimensions and hierarchy counts are provisional until their output artifacts are independently reviewed.\n- Nightshift-02 engineer report: full 42,840-key validation weight parity, 12/12 evaluation sNaive benchmark parity, plus validation baselines (Naive7=0.8697; Mean28=0.7607). These are **ENGINEER_REPORTED** while actual local source/diff and evidence JSON are unavailable here; official sNaive 0.8470173528438624 verified from `Scores and Ranks.xlsx`. [G3 review decision](docs/nightshift02-g3-review.md).

## Quick start — Windows PowerShell

Use Python 3.11.15 x64. Set UTF-8 mode before launching Python:

```powershell
$env:PYTHONUTF8 = "1"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r m5\requirements-py311.txt
python m5\scripts\00_preflight.py
python m5\scripts\fingerprint.py --input E:\m5-data\raw\m5_accuracy --output m5\runs\manifests\accuracy-data.json
```

The repository path is ASCII-only (`m5/`). Keep the raw M5 data outside Git, preferably on the E: data volume. Do not use the nearly full D: volume.

## Current executable checks

```powershell
python -m pytest m5\tests -q
python m5\scripts\01_audit_data.py --data-dir E:\m5-data\raw\m5_accuracy
python m5\scripts\02_verify_hierarchy.py --data-dir E:\m5-data\raw\m5_accuracy
```

These scripts currently check **file dimensions** and **hierarchy counts**, respectively. Passing does NOT prove unique IDs, sales-train historical equality, correct price joins or 12-level aggregation conservation. G1/G2 full audit still requires those tests.\n\nFor G3 source acquisition and strict per-key score checks, see [G3 / LightGBM contract](docs/w1-g3-and-lgbm-contract-v1.1.md). Download official weight files to E:/m5-data/reference (not Git) then run the cross-platform validator: `python m5/scripts/05_validate_reference_weights.py --validation E:/m5-data/reference/weights_validation.csv --evaluation E:/m5-data/reference/weights_evaluation.csv --output m5/runs/reference_weights_audit.json`. This validates weight data only; it is not an official scorer.

## WRMSSE zero-scale policy

The official guide and M5 Accuracy paper specify that the RMSSE scaling denominator is computed after the first non-zero observation. The companion R reference script inspected for this study does not contain an explicit fallback for a series whose active-history denominator is still zero. We therefore **raise an explicit undefined-scale error** rather than inserting epsilon, assigning zero, or silently dropping the series. If this edge case appears in official data, scorer status remains `SCORER_NOT_VERIFIED` until resolved against the official evaluation convention.

## Dependency discipline

`requirements-py311.txt` pins the direct top-level package versions for the initial Windows/Linux Python 3.11 environment. It is not yet a hash-complete transitive lockfile. The canonical Linux environment must generate and commit its own resolved lock after the platform is registered; do not reuse Windows-specific wheel hashes.

See [environment register](docs/environment-register.md), [four-week plan](../M5研究/docs/hands-on-execution-plan-v1.md), and [Week 1 Runbook](../M5研究/docs/week1-runbook.md).
