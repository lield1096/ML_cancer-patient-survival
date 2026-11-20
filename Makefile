.PHONY: install train compare test lint format serve docker-build docker-run clean

install:        ## Install package + dev tools in editable mode
	pip install -e ".[dev]"

train:          ## Tune, select and evaluate models; writes models/ and reports/
	survival-predictor train

compare:        ## Measure how much the original notebook's leakage inflated results
	python scripts/leakage_comparison.py

test:           ## Run test suite with coverage
	pytest --cov=survival_predictor --cov-report=term-missing

lint:           ## Static checks
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

serve:          ## Run the API locally at http://localhost:8000/docs
	uvicorn survival_predictor.api:app --reload

docker-build:
	docker build -t survival-predictor .

docker-run:
	docker run --rm -p 8000:8000 survival-predictor

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov **/__pycache__
