# One-command reproducibility. Run `make help` to see targets.
# Uses a plain Python venv + pip (no conda).
.PHONY: help setup reproduce smoke test lint check clean

VENV := .venv
PY := $(VENV)/bin/python

help:
	@echo "make setup      - create .venv and install requirements"
	@echo "make smoke      - dry run on small synthetic data (no download needed)"
	@echo "make reproduce  - full run on UNSW-NB15, writing results/ (THE grading command)"
	@echo "make check      - run the P5 grading checks on the results you already have"
	@echo "make test       - run the unit tests"
	@echo "make lint       - byte-compile check"
	@echo "make clean      - remove generated results"

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# The single command a grader runs. Must recreate your reported results.
reproduce:
	$(PY) -m src.p5 --config config.yaml

smoke:
	$(PY) -m src.p5 --config config.yaml --synthetic

check:
	$(PY) grading/grade.py --skip-reproduce

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m py_compile src/*.py tests/*.py

clean:
	rm -f results/*.png results/*.json
