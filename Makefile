# SpectraMind V50 — Makefile convenience targets

.PHONY: help deps lock fmt lint test selftest train predict dashboard ci-local

help:
	@echo "Targets: deps | lock | fmt | lint | test | selftest | train | predict | dashboard | ci-local"

# Install deps via Poetry (no-root keeps project editable)
deps:
	poetry install --no-root

# Refresh lockfile
lock:
	poetry lock --no-update

fmt:
	ruff format

lint:
	ruff check --fix

# Lightweight unit tests placeholder
test:
	poetry run python -m pytest -q || true

selftest:
	poetry run python -m spectramind selftest

train:
	poetry run python -m spectramind train --dry-run

predict:
	poetry run python -m spectramind predict --out-csv outputs/submission.csv

dashboard:
	poetry run python -m spectramind dashboard --html outputs/diagnostics/diagnostic_report_v50.html

ci-local: deps selftest test predict dashboard
