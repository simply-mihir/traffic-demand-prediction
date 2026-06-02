.PHONY: install train ensemble test lint clean all

all: install train

install:
	pip install -r requirements.txt

train:
	python src/solution.py

ensemble:
	jupyter nbconvert --to notebook --execute notebooks/02_stacked_ensemble.ipynb --output 02_stacked_ensemble.ipynb

test:
	python tests/make_synth_and_check.py

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

clean:
	rm -f submission.csv data/train.csv data/test.csv data/sample_submission.csv
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

help:
	@echo "Available targets:"
	@echo "  install   - Install dependencies"
	@echo "  train     - Run single-model pipeline → submission.csv"
	@echo "  ensemble  - Run the stacked-ensemble notebook"
	@echo "  test      - Run CI smoke test on synthetic data"
	@echo "  lint      - Run ruff + mypy"
	@echo "  clean     - Remove generated files"
