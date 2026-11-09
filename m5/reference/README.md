# M5 official reference assets (source pointers, not redistributed binaries)

The M5-methods companion folder exposes two different files named weights_validation.csv. They have identical 42,840 aggregation keys but different precision, column structure, and row order. They MUST NOT be confused.

| Variant | File link | Rows | Bytes | Header |
|---|---|---:|---:|---|
| Dataset / validation (primary) | [official file](https://drive.google.com/file/d/1-yjyAXJeuUQWXloCi9yC1LNO7Qtl1hcH/view) | 42,840 | 2,242,706 | Level_id, Agg_Level_1, Agg_Level_2, Dollar_Sales, weight |
| Dataset / evaluation | [official file](https://drive.google.com/file/d/1M37c__ioZChCYpY69oXSzBi3bKrKVEkN/view) | 42,840 | 2,257,736 | same as primary |
| validation / compact | [official file](https://drive.google.com/file/d/1KGhwZO23VtCUViyBJMfNZdMIGnPPx5QF/view) | 42,840 | 1,553,220 | Level_id, Agg_Level_1, Agg_Level_2, Weight |

Observed keyed comparison of the two validation files: 42,840 / 42,840 matches, maximum absolute difference about 5e-8. Prefer the primary Dataset variant for strict comparisons, then cross-check the compact variant.

Official R sources (download independently and fingerprint locally):
- [Estimate WRMSSE.R](https://drive.google.com/file/d/1K9nCkHB1n_OClVGSRmKvMqAGY9KhBXFf/view)
- [Estimate weights.R](https://drive.google.com/file/d/1nCRWaxzeSshpPn77Wqvaa3C1xXqtTwM5/view)
- [M5_Evaluation_Accuracy.R](https://drive.google.com/file/d/13uI3k_VtrO6oceTJE2kk0-NF05s2x1OW/view)
- [Companion Drive folder](https://drive.google.com/drive/folders/1D6EWdVSaOtrP1LEFh1REjI3vej6iUS_4)

Independent method summary: for each aggregation series, cut training history at its first positive sale and compute the mean squared first difference for RMSSE scale. Dollar sales are computed for the last 28 training days using daily item-store units times corresponding weekly prices. Across all 12 hierarchy levels the reference weight rule is 12 * series-dollar-sales / sum(all-level-dollar-sales), rounded to 12 decimal places. Per-level weighted RMSSE is summed, then the 12 level scores are averaged.

WARNING: the inspected Estimate WRMSSE.R does not explicitly define a zero-scale fallback for all-zero or constant active history. Keep SCORER_NOT_VERIFIED for such a case until reference-equivalent behavior is established. A zero-weight series is not the same as a zero-scale series.

The current GitHub repository does not establish an explicit license to redistribute complete third-party R files or weights data. Accordingly these are direct source links, not uploads. On the authorized machine, download, record SHA-256, and store only in a local untracked reference directory.

Run locally: python m5/scripts/fingerprint.py --input E:/m5-data/reference --output m5/runs/reference_sha256.json
Then run python m5/scripts/05_validate_reference_weights.py --validation E:/m5-data/reference/weights_validation.csv --evaluation E:/m5-data/reference/weights_evaluation.csv --compact-validation E:/m5-data/reference/weights_validation_compact.csv --output m5/runs/reference_weights_audit.json.

Passing reference integrity checks does NOT mean G3 has passed. Official-scoring parity requires keyed per-series scale and weight verification and benchmark forecast score reproduction.