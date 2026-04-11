#!/bin/bash
# Priority submission queue for 2026-04-10 (UTC)
# Fine-tuning the geometric mean blend ratio
echo "=== Round 8 Fine-Tuning Submissions ==="

SUB=/Users/guohongbin/projects/kaggle-store-sales/outputs/submissions

echo "[1/5] geo_01_99 (mean=343 - expect best)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r8_geo_01_99.csv" \
  -m "R8: Geo 1/99 model+TE, mean=343"

sleep 10

echo "[2/5] geo_02_98 (mean=335)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r8_geo_02_98.csv" \
  -m "R8: Geo 2/98 model+TE, mean=335"

sleep 10

echo "[3/5] geo_03_97 (mean=326)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r8_geo_03_97.csv" \
  -m "R8: Geo 3/97 model+TE, mean=326"

sleep 10

echo "[4/5] geo_05_95 (mean=311)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r8_geo_05_95.csv" \
  -m "R8: Geo 5/95 model+TE, mean=311"

sleep 10

echo "[5/5] geo_07_93 (mean=296)..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f "$SUB/submission_r8_geo_07_93.csv" \
  -m "R8: Geo 7/93 model+TE, mean=296"

echo "=== Done ==="
kaggle competitions submissions -c store-sales-time-series-forecasting 2>&1 | head -15
