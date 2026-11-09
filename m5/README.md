# M5 Hands-on Reproduction

This is the executable research scaffold for the M5 Accuracy / Uncertainty reproduction.

## Status

- Code scaffold and preflight: added.
- Windows development environment: owner-reported, registered.
- Canonical Linux environment: designated, machine specification pending.
- Official companion Drive: access verified through the connected Drive integration; the evaluation R scripts, weights, 24 Accuracy benchmark forecasts, and 50 Accuracy submissions are visible/downloadable.
- Kaggle rules acceptor and data custodian: **not yet named**.
- Full scorer equivalence and M5 model runs: **not yet verified / not yet run**.

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

The audit scripts are data-contract checks; passing them does not imply that WRMSSE has been verified against the official reference.

## WRMSSE zero-scale policy

The official guide and M5 Accuracy paper specify that the RMSSE scaling denominator is computed after the first non-zero observation. The companion R reference script inspected for this study does not contain an explicit fallback for a series whose active-history denominator is still zero. We therefore **raise an explicit undefined-scale error** rather than inserting epsilon, assigning zero, or silently dropping the series. If this edge case appears in official data, scorer status remains `SCORER_NOT_VERIFIED` until resolved against the official evaluation convention.

## Dependency discipline

`requirements-py311.txt` pins the direct top-level package versions for the initial Windows/Linux Python 3.11 environment. It is not yet a hash-complete transitive lockfile. The canonical Linux environment must generate and commit its own resolved lock after the platform is registered; do not reuse Windows-specific wheel hashes.

See [environment register](docs/environment-register.md), [four-week plan](../M5研究/docs/hands-on-execution-plan-v1.md), and [Week 1 Runbook](../M5研究/docs/week1-runbook.md).
