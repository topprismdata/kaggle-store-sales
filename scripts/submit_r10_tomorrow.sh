#!/bin/bash
# R10 day-specific model submissions for 2026-04-11 (UTC)
echo "=== R10 Day-Specific Model Submissions ==="

SUB=/Users/guohongbin/projects/kaggle-store-sales/outputs/submissions

echo "[1/5] r10_day_specific (raw, mean=432 - expect best)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r10_day_specific.csv" \
  -m "R10: Day-specific 16 LGB models (1st place approach), mean=432, CV=0.407"

sleep 15

echo "[2/5] r10_dayspec_geo_01_99 (mean=338)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r10_dayspec_geo_01_99.csv" \
  -m "R10: Day-specific + geo 1/99 TE, mean=338"

sleep 15

echo "[3/5] r10_dayspec_geo_02_98 (mean=326)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r10_dayspec_geo_02_98.csv" \
  -m "R10: Day-specific + geo 2/98 TE, mean=326"

sleep 15

echo "[4/5] r10_dayspec_geo_05_95 (mean=292)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r10_dayspec_geo_05_95.csv" \
  -m "R10: Day-specific + geo 5/95 TE, mean=292"

sleep 15

echo "[5/5] r10_dayspec_geo_10_90 (mean=247)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r10_dayspec_geo_10_90.csv" \
  -m "R10: Day-specific + geo 10/90 TE, mean=247"

echo "=== Done ==="
kaggle competitions submissions -c store-sales-time-series-forecasting 2>&1 | head -10
