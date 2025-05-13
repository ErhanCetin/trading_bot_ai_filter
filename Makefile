.PHONY: format lint check all

format:
	black .
	isort .

lint:
	flake8 .

check:
	black . --check
	isort . --check-only
	flake8 .

all: format lint
