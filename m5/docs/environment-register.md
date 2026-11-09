# M5 Environment Register

## Canonical and development environments

| Environment | Role | Status | Notes |
|---|---|---|---|
| Team Linux | Canonical / normative benchmark | Designated; machine specification pending | Its hardware, OS, Python, CUDA/driver, thread count and storage path must be recorded before the first canonical run. |
| Owner Windows 11 workstation | Development, smoke tests, exploratory analysis | Owner-reported; not independently verified | Full configuration below. Its runs must not be pooled with canonical Linux runs. |

## Owner-provided Windows workstation

| Field | Registered value |
|---|---|
| OS | Windows 11 build 26300 |
| CPU | Intel Core i7-14700KF |
| CPU topology | 20 physical cores / 28 logical threads |
| RAM | 64 GB |
| GPU | NVIDIA RTX 3080 Ti |
| GPU VRAM | 20 GB (owner-reported; verify with `nvidia-smi`) |
| NVIDIA driver | 581.94 |
| Python default | 3.13.13 |
| Project Python | 3.11.15 (selected for the initial M5 environment) |
| Kaggle CLI | 2.2.4 |
| Git | 2.55 |
| Data volume | E: (1.76 TB free at registration) |
| Do not use | D: (24.5 GB free at registration) |
| Other free space | C: 628 GB; G: 597 GB |

**Source status:** all workstation details above are Owner-provided, not independently hardware-probed. Capture actual `python --version`, `nvidia-smi`, `Get-ComputerInfo`, disk free space and environment lock in every run manifest.

## Reproducibility rules

1. Use Python 3.11.15 for the initial shared project environment.
2. Set `PYTHONUTF8=1` before starting Python on Windows.
3. Keep the project/code path ASCII-only (`m5/`); store raw data on E:.
4. Use deterministic LightGBM settings: `deterministic=true`, `force_row_wise=true`, fixed `num_threads`, and fixed seeds.
5. Windows development runs and Linux canonical runs use separate hardware IDs, configs, manifests and result namespaces.
6. Never pool scores across hardware environments without an explicit cross-environment reproducibility analysis.
7. The Linux machine's exact thread count and software lock remain open until the team registers its hardware.
