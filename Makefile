PYTHON ?= python
PIP := $(PYTHON) -m pip

.PHONY: install test lint build-graphs train evaluate compare-baselines clean

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	PYTHONPATH=src pytest --cov=src/gbd_ctu src/gbd_ctu/tests

lint:
	PYTHONPATH=src flake8 src

build-graphs:
	PYTHONPATH=src $(PYTHON) -m gbd_ctu build-graphs --data-root data/ctu13 --output-dir artifacts/graphs

train:
	PYTHONPATH=src $(PYTHON) -m gbd_ctu train --graph-dir artifacts/graphs --checkpoint artifacts/checkpoints/gnn_model.pt --epochs 20

evaluate:
	PYTHONPATH=src $(PYTHON) -m gbd_ctu evaluate --graph-dir artifacts/graphs --checkpoint artifacts/checkpoints/gnn_model.pt --output artifacts/reports/gnn_metrics.csv

compare-baselines:
	PYTHONPATH=src $(PYTHON) -m gbd_ctu compare-baselines --graph-dir artifacts/graphs --report-dir artifacts/reports --gnn-report artifacts/reports/gnn_metrics.csv

clean:
	rm -rf .coverage .pytest_cache build dist *.egg-info
	rm -rf artifacts/reports artifacts/checkpoints artifacts/graphs
