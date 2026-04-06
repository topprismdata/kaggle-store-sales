.PHONY: setup data features train ensemble submit clean

setup:
	pip install -r requirements.txt

data:
	python -c "from src.data.loader import load_raw_data, merge_all_tables; load_raw_data()"

features:
	python -m src.features.builder

train:
	python -m src.models.gbdt

ensemble:
	python -m src.ensemble.blending

submit:
	kaggle competitions submit -c store-sales-time-series-forecasting -f outputs/submissions/submission.csv -m "agentic submission"

clean:
	rm -rf data/processed/* outputs/models/* outputs/figures/*
