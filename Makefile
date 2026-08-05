.PHONY: setup ingest features train serve test eval clean docker-up docker-down

# Setup environment
setup:
	pip install poetry
	poetry install
	docker-compose up -d redis prometheus grafana

# Data pipeline
ingest:
	python -m orbit.ingestion.download_yoochoose
	python -m orbit.ingestion.enrich_data

features:
	python -m orbit.features.engineering

# Training
train:
	python -m orbit.models.train --model xgboost
	python -m orbit.models.train --model multitask

# Causal uplift
causal:
	python -m orbit.causal.uplift_engine --train

# Serving
serve:
	uvicorn orbit.serving.api:app --host 0.0.0.0 --port 8000 --reload

# Testing
test:
	pytest tests/ -v --cov=orbit --cov-report=html --cov-report=term

# Evaluation
eval:
	python -m orbit.evaluation.run_benchmark

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ htmlcov/ .coverage

# Full pipeline
pipeline: ingest features train causal eval
	@echo "Full pipeline complete!"
