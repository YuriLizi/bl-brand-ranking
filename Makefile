.PHONY: help install ingest train-test train-prod serve loadtest test docker-build up down clean

help:
	@echo "install     create the venv and install dependencies (needs Python 3.12+)"
	@echo "ingest      load bl_full_data.csv into the local Delta table"
	@echo "train-test  evaluation run (no artifacts written)"
	@echo "train-prod  production run (registers a new @champion version)"
	@echo "serve       run the ranking endpoint locally"
	@echo "loadtest    simulate production traffic and report p50/p95/p99"
	@echo "test        run the test suite"
	@echo "up / down   start / stop the full Docker Compose stack"

install:
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements-torch.txt
	.venv/bin/pip install -r requirements.txt

ingest:
	PYTHONPATH=src python -m bl_ranker.data.ingest

train-test:
	PYTHONPATH=src python -m bl_ranker.training.run --mode train_test

train-prod:
	PYTHONPATH=src python -m bl_ranker.training.run --mode production

serve:
	PYTHONPATH=src uvicorn bl_ranker.serving.app:app --host 0.0.0.0 --port 8000

loadtest:
	PYTHONPATH=src python loadtest/simulate.py --requests 500 --concurrency 1 4 8 16

test:
	PYTHONPATH=src pytest -q

docker-build:
	docker compose --profile build build base
	docker compose build

up:
	docker compose up -d mlflow
	docker compose up -d serving scheduler

down:
	docker compose down

clean:
	rm -rf data/delta mlruns artifacts loadtest/results.json
